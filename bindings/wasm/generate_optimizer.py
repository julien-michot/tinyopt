# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from pathlib import Path


SCALAR_TYPES = {
    "char": (1, 1, "i8"),
    "signed char": (1, 1, "i8"),
    "unsigned char": (1, 1, "u8"),
    "short": (2, 2, "i16"),
    "unsigned short": (2, 2, "u16"),
    "int": (4, 4, "i32"),
    "unsigned int": (4, 4, "u32"),
    "float": (4, 4, "f32"),
    "double": (8, 8, "f64"),
    "_Bool": (1, 1, "u8"),
    "bool": (1, 1, "u8"),
    "size_t": (4, 4, "u32"),
}
HEAP_LAYOUTS = {
    "i8": ("HEAP8", None),
    "u8": ("HEAPU8", None),
    "i16": ("HEAP16", 1),
    "u16": ("HEAPU16", 1),
    "i32": ("HEAP32", 2),
    "u32": ("HEAPU32", 2),
    "f32": ("HEAPF32", 2),
    "f64": ("HEAPF64", 3),
}


def align(offset, alignment):
    return (offset + alignment - 1) // alignment * alignment


def camel_case(name):
    return re.sub(r"_([a-z0-9])", lambda match: match.group(1).upper(), name)


def strip_comments(source):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)


def extract_braced(source, start):
    opening = source.find("{", start)
    if opening < 0:
        raise ValueError("expected a struct or union body")
    depth = 1
    for index in range(opening + 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index], index + 1
    raise ValueError("unterminated struct or union declaration")


def declarations(body):
    result = []
    start = 0
    depth = 0
    for index, char in enumerate(body):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == ";" and depth == 0:
            declaration = " ".join(body[start:index].split())
            if declaration:
                result.append(declaration)
            start = index + 1
    trailing = " ".join(body[start:].split())
    if trailing:
        raise ValueError(f"unterminated field declaration: {trailing}")
    return result


def parse_field(declaration, typedefs, enum_types):
    union_match = re.fullmatch(r"union\s*\{(.*)\}\s*(\w+)", declaration)
    if union_match:
        members, name = union_match.groups()
        member_layouts = [
            parse_field(member, typedefs, enum_types)
            for member in declarations(members)
        ]
        union_alignment = max(field["alignment"] for field in member_layouts)
        union_size = align(max(field["size"] for field in member_layouts), union_alignment)
        return {"name": name, "size": union_size, "alignment": union_alignment, "kind": "pointer"}

    function_pointer = re.fullmatch(r".*?\(\s*\*\s*(\w+)\s*\)\s*\(.*\)", declaration)
    if function_pointer:
        name = function_pointer.group(1)
        return {"name": name, "size": 4, "alignment": 4, "kind": "pointer"}

    pointer = re.fullmatch(r".+?\*\s*(\w+)", declaration)
    if pointer:
        name = pointer.group(1)
        return {"name": name, "size": 4, "alignment": 4, "kind": "pointer"}

    scalar = re.fullmatch(r"(.+?)\s+(\w+)", declaration)
    if scalar is None:
        raise ValueError(f"unsupported field declaration: {declaration}")
    type_name, name = scalar.groups()
    type_name = type_name.strip()
    if type_name in enum_types:
        size, alignment, kind = 4, 4, "i32"
    elif type_name in typedefs:
        size, alignment, kind = typedefs[type_name]
    else:
        raise ValueError(f"unsupported type {type_name!r} for field {name}")
    return {"name": name, "size": size, "alignment": alignment, "kind": kind}


def parse_struct(source, struct_name, typedefs, enum_types):
    match = re.search(rf"typedef\s+struct\s+{re.escape(struct_name)}\s*\{{", source)
    if match is None:
        raise ValueError(f"could not find struct {struct_name}")
    body, _ = extract_braced(source, match.start())

    fields = []
    offset = 0
    max_alignment = 1
    for declaration in declarations(body):
        field = parse_field(declaration, typedefs, enum_types)
        offset = align(offset, field["alignment"])
        field["offset"] = offset
        fields.append(field)
        offset += field["size"]
        max_alignment = max(max_alignment, field["alignment"])
    return fields, align(offset, max_alignment)


def parse_layouts(common_header, double_header):
    common = strip_comments(common_header)
    double = strip_comments(double_header)
    source = common + "\n" + double
    typedefs = dict(SCALAR_TYPES)
    typedefs.update(
        {
            name: (4, 4, "pointer")
            for name in re.findall(
                r"typedef\s+[^;]*\(\s*\*\s*(\w+)\s*\)\s*[^;]*;", source, flags=re.DOTALL
            )
        }
    )
    enum_types = set(
        re.findall(
            r"typedef\s+enum(?:\s+\w+)?\s*\{.*?\}\s*(\w+)\s*;",
            source,
            flags=re.DOTALL,
        )
    )
    layouts = {}
    for key, name in (
        ("options", "tinyopt_options_t"),
        ("summary", "tinyopt_summary_t"),
        ("params", "tinyopt_params_t"),
        ("problem", "tinyopt_problem_t"),
    ):
        layouts[key] = parse_struct(source, name, typedefs, enum_types)
    return layouts


def field_aliases(name):
    alias = camel_case(name)
    return [name] if alias == name else [alias, name]


def layout_js(layouts):
    lines = ["export const LAYOUT = {"]
    for struct_name, (fields, size) in layouts.items():
        lines.append(f"  {struct_name}: {{")
        lines.append(f"    size: {size},")
        for field in fields:
            for alias in field_aliases(field["name"]):
                lines.append(f"    {alias}: {field['offset']},")
        lines.append("  },")
    lines.append("};")
    return "\n".join(lines)


def option_fields_js(fields):
    lines = ["const OPTION_FIELDS = {"]
    for field in fields:
        if field["kind"] not in HEAP_LAYOUTS:
            continue
        heap, shift = HEAP_LAYOUTS[field["kind"]]
        shift_value = "null" if shift is None else shift
        for alias in field_aliases(field["name"]):
            lines.append(
                f"  {alias}: [{field['offset']}, '{heap}', {shift_value}],"
            )
    lines.append("};")
    return "\n".join(lines)


def option_names_js(fields):
    lines = ["const OPTION_NAMES = {"]
    for field in fields:
        for alias in field_aliases(field["name"]):
            lines.append(f"  {alias}: '{camel_case(field['name'])}',")
    lines.append("};")
    return "\n".join(lines)


def generate(template, common_header, double_header):
    layouts = parse_layouts(common_header, double_header)
    for marker, replacement in (
        ("export const LAYOUT = null;", layout_js(layouts)),
        ("const OPTION_FIELDS = null;", option_fields_js(layouts["options"][0])),
        ("const OPTION_NAMES = null;", option_names_js(layouts["options"][0])),
    ):
        if template.count(marker) != 1:
            raise ValueError(f"expected exactly one {marker!r} generation marker")
        template = template.replace(marker, replacement, 1)
    return template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--common-header", type=Path, required=True)
    parser.add_argument("--double-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = generate(
        args.template.read_text(),
        args.common_header.read_text(),
        args.double_header.read_text(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)


if __name__ == "__main__":
    main()
