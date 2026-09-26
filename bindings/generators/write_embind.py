#!/usr/bin/env python3
"""
Generate Emscripten Embind bindings from parsed C++ structs and enums.
"""

import os
import re
import sys
from pathlib import Path


def write_embind_structs(output_path, header_includes, structs, enums_global=None, include_dir=None):
    """Generate Emscripten Embind bindings for Options and Output structs."""
    out = []
    out.append('// Generated Embind bindings - do not edit by hand')
    out.append('#ifndef TINYOPT_GENERATED_EMBIND_BINDINGS')
    out.append('#define TINYOPT_GENERATED_EMBIND_BINDINGS 1')
    out.append('#include <emscripten/bind.h>')
    out.append('#include <emscripten/val.h>')
    for h in header_includes:
        if include_dir:
            rel = Path(h).relative_to(include_dir) if Path(h).is_absolute() else Path(h)
            out.append(f'#include <{rel}>')
        else:
            out.append(f'#include "{h}"')
    out.append('')
    out.append('namespace em = emscripten;')
    out.append('using namespace tinyopt;')
    out.append('')

    whitelist = set(os.environ.get('GEN_STRUCT_WHITELIST','Output,Options').split(','))

    # Collect enums to emit
    enums_to_emit = []
    if enums_global:
        nested_enum_names = set()
        for entry in [s for s in structs if s[1] in whitelist]:
            _, _, _, enums, _ = entry
            for (enum_name, _) in enums:
                nested_enum_names.add(enum_name)

        for fq, short, enumerators, src in enums_global:
            if short in whitelist or short in nested_enum_names:
                enums_to_emit.append((fq, short, enumerators))

    # Emit enums
    seen_enum_shorts = set()
    for fq, short, enumerators in enums_to_emit:
        if short in seen_enum_shorts:
            continue
        seen_enum_shorts.add(short)
        out.append(f'  // Enum: {short}')
        out.append(f'  em::enum_<{fq}>("{short}")')
        for (e_name, e_val) in enumerators:
            out.append(f'    .value("{e_name}", {fq}::{e_name})')
        out.append('    ;')
        out.append('')

    # Emit structs
    to_emit = []
    for s in structs:
        cpp_n, py_n = s[0], s[1]
        if py_n in whitelist:
            to_emit.append(s)
        parent = cpp_n.split('::')[-2] if '::' in cpp_n and len(cpp_n.split('::')) > 1 else None
        if parent and parent in whitelist:
            to_emit.append(s)

    # Sort by nesting depth (deepest first)
    to_emit.sort(key=lambda e: e[0].count('::'), reverse=True)

    # Deduplicate
    seen = set()
    unique_emit = []
    for entry in to_emit:
        cpp_n = entry[0]
        if cpp_n in seen:
            continue
        seen.add(cpp_n)
        unique_emit.append(entry)

    for entry in unique_emit:
        cpp_name, py_name, fields, enums, callbacks = entry

        # Skip Output struct - we use convert_output_to_js instead
        if py_name == 'Output':
            out.append(f'  // Skipping Output struct - using custom conversion function')
            continue

        out.append(f'  // Bind struct: {cpp_name}')
        out.append(f'  em::class_<{cpp_name}>("{py_name}")')
        out.append(f'    .constructor<>()')

        # Bind fields as properties
        for (ftype, fname) in fields:
            # Skip types that Embind can't handle
            if 'TimePoint' in ftype or 'time_point' in ftype or 'chrono' in ftype:
                continue
            if 'std::function' in ftype or 'function<' in ftype:
                continue
            if 'std::vector' in ftype:
                continue
            if 'std::array' in ftype:
                continue
            if 'std::variant' in ftype:
                continue
            if 'Matrix' in ftype or 'SparseMatrix' in ftype or 'Eigen' in ftype:
                continue
            # Bind the property
            out.append(f'    .property("{fname}", &{cpp_name}::{fname})')

        out.append('    ;')
        out.append('')

    try:
        Path(output_path).write_text('\n'.join(out))
    except Exception as e:
        print(f'Failed to write Embind bindings to {output_path}: {e}', file=sys.stderr)


def write_embind_conversions(output_path, header_includes, structs, enums_global=None, include_dir=None):
    """
    Generate helper functions to convert C++ structs to JavaScript objects.
    This creates a convert_output_to_js() function and similar helpers.
    """
    out = []
    out.append('// Generated Embind conversion helpers - do not edit by hand')
    out.append('#ifndef TINYOPT_GENERATED_EMBIND_CONVERSIONS')
    out.append('#define TINYOPT_GENERATED_EMBIND_CONVERSIONS 1')
    out.append('#include <emscripten/val.h>')
    for h in header_includes:
        if include_dir:
            rel = Path(h).relative_to(include_dir) if Path(h).is_absolute() else Path(h)
            out.append(f'#include <{rel}>')
        else:
            out.append(f'#include "{h}"')
    out.append('')
    out.append('namespace em = emscripten;')
    out.append('using namespace tinyopt;')
    out.append('')

    whitelist = set(os.environ.get('GEN_STRUCT_WHITELIST','Output,Options').split(','))

    # Find Output struct
    output_entry = next((s for s in structs if s[1] == 'Output'), None)

    if output_entry:
        cpp_name, py_name, fields, enums, callbacks = output_entry

        out.append(f'// Convert {cpp_name} to JavaScript object')
        out.append(f'em::val convert_{py_name.lower()}_to_js(const {cpp_name}& out) {{')
        out.append('  em::val js_obj = em::val::object();')

        for (ftype, fname) in fields:
            # Debug: print the actual type string
            if fname == 'start_time':
                with open('/tmp/embind_debug.txt', 'a') as f:
                    f.write(f"DEBUG: start_time has ftype = '{ftype}'\n")

            # Handle different field types
            if 'std::string' in ftype:
                out.append(f'  js_obj.set("{fname}", em::val(out.{fname}));')
            elif 'std::vector' in ftype:
                # Skip vectors - would need array conversion
                out.append(f'  // Skipping vector field: {fname}')
            elif 'std::array' in ftype:
                # Skip arrays for now
                out.append(f'  // Skipping array field: {fname}')
            elif 'std::function' in ftype or 'std::variant' in ftype:
                # Skip complex types
                out.append(f'  // Skipping complex field: {fname}')
            elif 'StopReason' in ftype:
                # Convert enum to int
                out.append(f'  js_obj.set("{fname}", static_cast<int>(out.{fname}));')
            elif 'TimePoint' in ftype or 'time_point' in ftype or 'chrono' in ftype:
                # Skip time_point - Embind can't convert it
                out.append(f'  // Skipping TimePoint field: {fname}')
            elif 'Matrix' in ftype or 'SparseMatrix' in ftype or 'Eigen' in ftype:
                # Skip Eigen types
                out.append(f'  // Skipping Eigen field: {fname}')
            elif 'bool' in ftype or 'int' in ftype or 'uint' in ftype or 'float' in ftype or 'double' in ftype or 'size_t' in ftype:
                # Primitive types
                out.append(f'  js_obj.set("{fname}", out.{fname});')
            else:
                # Try to handle nested structs
                m = re.search(r'([A-Za-z_]\w*(::[A-Za-z_]\w*)*)$', ftype)
                if m:
                    base = m.group(1).split('::')[-1]
                    struct_short_names = set([s[1] for s in structs])
                    if base in struct_short_names:
                        out.append(f'  // TODO: Convert nested struct {fname} of type {base}')
                    else:
                        out.append(f'  js_obj.set("{fname}", out.{fname});')
                else:
                    out.append(f'  js_obj.set("{fname}", out.{fname});')

        out.append('  return js_obj;')
        out.append('}')
        out.append('')

    out.append('void bind_generated_embind_conversions() {')
    if output_entry:
        out.append(f'  em::function("convert_output_to_js", &convert_output_to_js);')
    out.append('}')
    out.append('')
    out.append('#endif // TINYOPT_GENERATED_EMBIND_CONVERSIONS')

    try:
        Path(output_path).write_text('\n'.join(out))
    except Exception as e:
        import sys
        print(f'Failed to write Embind conversions to {output_path}: {e}', file=sys.stderr)


def write_embind(output_path, header_includes, structs, enums_global=None, include_dir=None):
    """
    Main function that generates both struct bindings and conversion helpers.
    This combines both into a single file with two functions:
    - bind_generated_embind_structs()
    - bind_generated_embind_conversions()
    """
    out = []
    out.append('// Generated Embind bindings - do not edit by hand')
    out.append('#ifndef TINYOPT_GENERATED_EMBIND_BINDINGS')
    out.append('#define TINYOPT_GENERATED_EMBIND_BINDINGS 1')
    out.append('#include <emscripten/bind.h>')
    out.append('#include <emscripten/val.h>')
    for h in header_includes:
        if include_dir:
            rel = Path(h).relative_to(include_dir) if Path(h).is_absolute() else Path(h)
            out.append(f'#include <{rel}>')
        else:
            out.append(f'#include "{h}"')
    out.append('')
    out.append('namespace em = emscripten;')
    out.append('using namespace tinyopt;')
    out.append('')

    whitelist = set(os.environ.get('GEN_STRUCT_WHITELIST','Output,Options').split(','))

    # === Part 1: Conversion functions ===
    output_entry = next((s for s in structs if s[1] == 'Output'), None)

    if output_entry:
        cpp_name, py_name, fields, enums, callbacks = output_entry

        out.append(f'// Convert {cpp_name} to JavaScript object')
        out.append(f'em::val convert_{py_name.lower()}_to_js(const {cpp_name}& out) {{')
        out.append('  em::val js_obj = em::val::object();')

        for (ftype, fname) in fields:
            # Skip complex types first
            if fname == 'start_time':  # TimePoint parsed incorrectly as 'int' by libclang
                out.append(f'  // Skipping TimePoint field: {fname}')
                continue
            if fname == 'final_hessian':  # std::variant<monostate, MatX, SparseMat>
                out.append(f'  // Skipping variant<Hessian> field: {fname}')
                continue
            if fname in ['errs', 'deltas2', 'successes']:  # std::vector fields
                out.append(f'  // Skipping vector field: {fname} (not yet converted)')
                continue
            if 'TimePoint' in ftype or 'time_point' in ftype or 'chrono' in ftype:
                out.append(f'  // Skipping time_point field: {fname}')
                continue
            if 'Matrix' in ftype or 'Hessian' in fname or 'SparseMatrix' in ftype or 'Eigen' in ftype:
                out.append(f'  // Skipping matrix field: {fname}')
                continue
            if 'std::function' in ftype or 'std::variant' in ftype or 'function<' in ftype:
                out.append(f'  // Skipping complex field: {fname}')
                continue

            # Handle simple types
            if 'std::string' in ftype:
                out.append(f'  js_obj.set("{fname}", em::val(out.{fname}));')
            elif 'std::vector' in ftype:
                # Debug vectors
                with open('/tmp/embind_debug.txt', 'a') as f:
                    f.write(f"Vector field '{fname}' has ftype = '{ftype}'\n")
                # Convert vectors to JavaScript arrays
                if 'double' in ftype or 'float' in ftype or 'int' in ftype or 'bool' in ftype:
                    out.append(f'  if (!out.{fname}.empty()) {{')
                    out.append(f'    js_obj.set("{fname}", em::val::array(out.{fname}.begin(), out.{fname}.end()));')
                    out.append(f'  }}')
                else:
                    out.append(f'  // Skipping complex vector field: {fname}')
            elif 'std::array' in ftype:
                out.append(f'  // Skipping array field: {fname}')
            elif 'StopReason' in ftype:
                out.append(f'  js_obj.set("{fname}", static_cast<int>(out.{fname}));')
            elif 'Cost' in ftype:
                # Convert Cost to double (it's a wrapper around double)
                out.append(f'  js_obj.set("{fname}", static_cast<double>(out.{fname}));')
            elif 'bool' in ftype or 'int' in ftype or 'uint' in ftype or 'float' in ftype or 'double' in ftype or 'size_t' in ftype:
                # Primitive types
                out.append(f'  js_obj.set("{fname}", out.{fname});')
            else:
                m = re.search(r'([A-Za-z_]\w*(::[A-Za-z_]\w*)*)$', ftype)
                if m:
                    base = m.group(1).split('::')[-1]
                    struct_short_names = set([s[1] for s in structs])
                    if base in struct_short_names:
                        out.append(f'  // Skipping nested struct field: {fname}')
                    else:
                        out.append(f'  js_obj.set("{fname}", out.{fname});')
                else:
                    out.append(f'  // Skipping unknown field: {fname}')

        out.append('  return js_obj;')
        out.append('}')
        out.append('')

    # === Part 2: Struct bindings ===
    out.append('void bind_generated_embind_structs() {')

    # Collect enums to emit
    enums_to_emit = []
    if enums_global:
        nested_enum_names = set()
        for entry in [s for s in structs if s[1] in whitelist]:
            _, _, _, enums, _ = entry
            for (enum_name, _) in enums:
                nested_enum_names.add(enum_name)

        for fq, short, enumerators, src in enums_global:
            if short in whitelist or short in nested_enum_names:
                enums_to_emit.append((fq, short, enumerators))

    # Emit enums
    seen_enum_shorts = set()
    for fq, short, enumerators in enums_to_emit:
        if short in seen_enum_shorts:
            continue
        seen_enum_shorts.add(short)
        out.append(f'  // Enum: {short}')
        out.append(f'  em::enum_<{fq}>("{short}")')
        for (e_name, e_val) in enumerators:
            out.append(f'    .value("{e_name}", {fq}::{e_name})')
        out.append('    ;')
        out.append('')

    # Emit structs
    to_emit = []
    for s in structs:
        cpp_n, py_n = s[0], s[1]
        if py_n in whitelist:
            to_emit.append(s)
        parent = cpp_n.split('::')[-2] if '::' in cpp_n and len(cpp_n.split('::')) > 1 else None
        if parent and parent in whitelist:
            to_emit.append(s)

    to_emit.sort(key=lambda e: e[0].count('::'), reverse=True)

    seen = set()
    unique_emit = []
    for entry in to_emit:
        cpp_n = entry[0]
        if cpp_n in seen:
            continue
        seen.add(cpp_n)
        unique_emit.append(entry)

    for entry in unique_emit:
        cpp_name, py_name, fields, enums, callbacks = entry

        out.append(f'  // Bind struct: {cpp_name}')
        out.append(f'  em::class_<{cpp_name}>("{py_name}")')
        # Don't bind constructor for Output to avoid initializing TimePoint members
        if py_name != 'Output':
            out.append(f'    .constructor<>()')

        for (ftype, fname) in fields:
            # Skip types that Embind can't handle
            # Check for time_point types (can appear as TimePoint or full std::chrono::time_point<...>)
            if 'TimePoint' in ftype or 'time_point' in ftype or 'chrono' in ftype or fname == 'start_time':
                continue
            if 'std::function' in ftype or 'function<' in ftype or 'callback' in fname:
                continue
            if 'std::vector' in ftype:
                continue
            if 'std::array' in ftype:
                continue
            if 'std::variant' in ftype:
                continue
            if 'Matrix' in ftype or 'SparseMatrix' in ftype or 'Eigen' in ftype or 'Hessian' in fname:
                continue
            out.append(f'    .property("{fname}", &{cpp_name}::{fname})')

        out.append('    ;')
        out.append('')

    out.append('}')
    out.append('')

    # === Part 3: Register conversion functions ===
    out.append('void bind_generated_embind_conversions() {')
    if output_entry:
        out.append('  em::function("convert_output_to_js", &convert_output_to_js);')
    out.append('}')
    out.append('')

    out.append('#endif // TINYOPT_GENERATED_EMBIND_BINDINGS')

    try:
        Path(output_path).write_text('\n'.join(out))
    except Exception as e:
        import sys
        print(f'Failed to write Embind bindings to {output_path}: {e}', file=sys.stderr)


def main(argv):
    """Main entry point when called as a script"""
    if len(argv) < 4:
        print('Usage: write_embind.py <header1> [<header2> ...] <output_cpp> <include_dir>')
        return 2

    from pathlib import Path
    from common_parser import try_configure_libclang, generate_from_clang, HAVE_CLANG

    headers = [Path(p) for p in argv[1:-2]]
    out_cpp = Path(argv[-2])
    include_dir = Path(argv[-1])

    # Add stop_reasons header if it exists
    stop_reasons = include_dir / 'tinyopt' / 'stop_reasons.h'
    if stop_reasons.exists() and stop_reasons not in headers:
        headers.append(stop_reasons)

    if not HAVE_CLANG:
        print('Error: libclang not available. Install python-clang.', file=sys.stderr)
        return 1

    configured = try_configure_libclang()
    try:
        structs, enums_global = generate_from_clang(headers, include_dir)
    except Exception as e:
        print('Error: libclang parsing failed:', e, file=sys.stderr)
        return 2

    header_names = [str(h) for h in headers]
    write_embind(out_cpp, header_names, structs, enums_global=enums_global, include_dir=include_dir)
    print(f'Wrote Embind bindings to {out_cpp} with {len(structs)} struct(s)')
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv))
