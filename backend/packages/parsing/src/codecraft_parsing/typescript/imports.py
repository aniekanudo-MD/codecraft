"""Read imports, re-exports and dynamic imports from a TypeScript syntax tree.

Only grammar nodes produce references. Text inside comments or string literals
is not an import. Positions are characters, converted from the grammar's byte
columns.
"""

import posixpath
import re
from dataclasses import dataclass
from typing import Self

from codecraft_core.model import EdgeKind, ImportCondition, ParseFailure
from tree_sitter import Node, Tree

MAX_REFERENCE_TEXT = 120
_TYPE_ONLY_IMPORT = re.compile(r"^import\s+type(?=\s|\{|\*)")
_TYPE_ONLY_EXPORT = re.compile(r"^export\s+type(?=\s|\{|\*)")
_LOW_SURROGATE_ESCAPE = re.compile(r"\\u([dD][c-fC-F][0-9a-fA-F]{2})")
_ASSET_EXTENSIONS = frozenset(
    {
        ".json",
        ".css",
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".avif",
        ".ico",
        ".bmp",
    }
)


@dataclass(frozen=True)
class SourceText:
    """A decoded file, split into lines for character positions."""

    text: str
    lines: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(text=text, lines=tuple(text.split("\n")))

    def position(self, node: Node) -> tuple[int, int]:
        row, byte_column = node.start_point
        if row >= len(self.lines):
            return row + 1, byte_column + 1
        try:
            prefix = self.lines[row].encode("utf-8")[:byte_column].decode("utf-8")
        except UnicodeDecodeError:
            return row + 1, byte_column + 1
        return row + 1, len(prefix) + 1


@dataclass(frozen=True)
class RawReference:
    """One literal reference, or a non-literal dynamic import, before resolution."""

    reference: str
    kind: EdgeKind
    line: int
    column: int
    condition: ImportCondition | None
    asset: bool
    is_re_export: bool
    failure_reason: str | None = None


def read_references(tree: Tree, source: SourceText) -> list[RawReference]:
    found: list[RawReference] = []
    stack: list[Node] = [tree.root_node]
    while stack:
        node = stack.pop()
        match node.type:
            case "import_statement":
                reference = _module_reference(node, source)
                if reference is not None:
                    found.append(reference)
            case "export_statement":
                reference = _re_export(node, source)
                if reference is not None:
                    found.append(reference)
            case "call_expression":
                reference = _dynamic_import(node, source)
                if reference is not None:
                    found.append(reference)
            case _:
                pass
        stack.extend(reversed(node.named_children))
    return found


def first_syntax_problem(root: Node, source: SourceText) -> ParseFailure | None:
    if not root.has_error:
        return None
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            line, column = source.position(node)
            reason = "syntax error"
            return ParseFailure(reason=reason, line=line, column=column)
        stack.extend(reversed(node.children))
    return None


def _module_reference(node: Node, source: SourceText) -> RawReference | None:
    source_node = node.child_by_field_name("source")
    if source_node is None:
        return None
    line, column = source.position(node)
    text = _node_text(node)
    type_only = _TYPE_ONLY_IMPORT.match(text.lstrip()) is not None
    specifier = _string_value(source_node)
    if specifier is None:
        return RawReference(
            reference=_node_text(source_node),
            kind=EdgeKind.IMPORT,
            line=line,
            column=column,
            condition=None,
            asset=False,
            is_re_export=False,
            failure_reason="module specifier cannot be decoded",
        )
    asset = is_asset(specifier)
    return RawReference(
        reference=specifier,
        kind=(
            EdgeKind.ASSET_IMPORT
            if asset
            else (EdgeKind.TYPE_ONLY_IMPORT if type_only else EdgeKind.IMPORT)
        ),
        line=line,
        column=column,
        condition=None,
        asset=asset,
        is_re_export=False,
    )


def _re_export(node: Node, source: SourceText) -> RawReference | None:
    source_node = node.child_by_field_name("source")
    if source_node is None:
        return None
    line, column = source.position(node)
    text = _node_text(node)
    type_only = _TYPE_ONLY_EXPORT.match(text.lstrip()) is not None
    specifier = _string_value(source_node)
    if specifier is None:
        return RawReference(
            reference=_node_text(source_node),
            kind=EdgeKind.RE_EXPORT,
            line=line,
            column=column,
            condition=None,
            asset=False,
            is_re_export=True,
            failure_reason="module specifier cannot be decoded",
        )
    return RawReference(
        reference=specifier,
        kind=EdgeKind.RE_EXPORT,
        line=line,
        column=column,
        condition=ImportCondition.TYPE_ONLY if type_only else None,
        asset=is_asset(specifier),
        is_re_export=True,
    )


def _dynamic_import(node: Node, source: SourceText) -> RawReference | None:
    function = node.child_by_field_name("function")
    if function is None or function.type != "import":
        return None
    arguments = node.child_by_field_name("arguments")
    if arguments is None or not arguments.named_children:
        return None
    first = arguments.named_children[0]
    line, column = source.position(node)
    if first.type == "string":
        specifier = _string_value(first)
        if specifier is None:
            return RawReference(
                reference=_node_text(node),
                kind=EdgeKind.DYNAMIC_IMPORT,
                line=line,
                column=column,
                condition=None,
                asset=False,
                is_re_export=False,
                failure_reason="module specifier cannot be decoded",
            )
        asset = is_asset(specifier)
        return RawReference(
            reference=specifier,
            kind=EdgeKind.ASSET_IMPORT if asset else EdgeKind.DYNAMIC_IMPORT,
            line=line,
            column=column,
            condition=None,
            asset=asset,
            is_re_export=False,
        )
    text = _node_text(node)
    if len(text) > MAX_REFERENCE_TEXT:
        text = f"{text[:MAX_REFERENCE_TEXT]}..."
    return RawReference(
        reference=text,
        kind=EdgeKind.DYNAMIC_IMPORT,
        line=line,
        column=column,
        condition=None,
        asset=False,
        is_re_export=False,
        failure_reason="non-literal dynamic import",
    )


def is_asset(specifier: str) -> bool:
    """Whether a relative specifier names an asset, ignoring ?query and #fragment."""
    if not specifier.startswith((".", "/")):
        return False
    return has_asset_extension(specifier)


def has_asset_extension(specifier: str) -> bool:
    path = specifier.split("?", 1)[0].split("#", 1)[0]
    return posixpath.splitext(path)[1].lower() in _ASSET_EXTENSIONS


def _node_text(node: Node) -> str:
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8")


def _string_value(node: Node) -> str | None:
    raw = _node_text(node)
    if len(raw) < 2 or raw[0] not in "'\"" or raw[-1] != raw[0]:
        return None
    return _decode_js_string(raw[1:-1])


def _decode_js_string(value: str) -> str | None:
    out: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        index += 1
        if index >= len(value):
            return None
        escape = value[index]
        index += 1
        simple = {
            "\\": "\\",
            "'": "'",
            '"': '"',
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "b": "\b",
            "f": "\f",
            "v": "\v",
            "0": "\0",
        }
        if escape in simple:
            out.append(simple[escape])
            continue
        if escape == "\n":
            continue
        if escape == "\r":
            if index < len(value) and value[index] == "\n":
                index += 1
            continue
        if escape == "x":
            digits = value[index : index + 2]
            if len(digits) != 2 or not all(c in "0123456789abcdefABCDEF" for c in digits):
                return None
            out.append(chr(int(digits, 16)))
            index += 2
            continue
        if escape == "u":
            if index < len(value) and value[index] == "{":
                end = value.find("}", index + 1)
                if end == -1:
                    return None
                digits = value[index + 1 : end]
                if not digits or not all(c in "0123456789abcdefABCDEF" for c in digits):
                    return None
                codepoint = int(digits, 16)
                if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                    return None
                out.append(chr(codepoint))
                index = end + 1
                continue
            digits = value[index : index + 4]
            if len(digits) != 4 or not all(c in "0123456789abcdefABCDEF" for c in digits):
                return None
            codepoint = int(digits, 16)
            index += 4
            if 0xD800 <= codepoint <= 0xDBFF:
                low = _LOW_SURROGATE_ESCAPE.match(value, index)
                if low is None:
                    return None
                codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (int(low[1], 16) - 0xDC00)
                index = low.end()
            elif 0xDC00 <= codepoint <= 0xDFFF:
                return None
            out.append(chr(codepoint))
            continue
        out.append(escape)
    return "".join(out)
