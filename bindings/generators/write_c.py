"""
Generate C bindings and conversion functions from parsed C++ structs and enums.
"""
import os
import sys
import re
from pathlib import Path


def _map_type_to_c(ftype):
    """Conservative mapping: prefer builtins, pointer retention, fallback to void*"""
    t = ftype.strip()
    # remove const/volatile
    t = re.sub(r'\bconst\b', '', t)
    t = re.sub(r'\bvolatile\b', '', t)
    t = t.strip()
    # pointer detection
    if '*' in t:
        base = t.replace('*','').strip()
        if 'double' in base:
            return 'double*'
        if 'float' in base:
            return 'float*'
        if 'int' in base:
            return 'int*'
        return 'void*'
    # simple builtin
    if re.search(r'\bdouble\b', t):
        return 'double'
    if re.search(r'\bfloat\b', t):
        return 'float'
    if re.search(r'\bint\b', t):
        return 'int'
    if re.search(r'\bsize_t\b', t):
        return 'size_t'
    # enum or user type -> keep name as-is for enum, otherwise void*
    m = re.search(r'([A-Za-z_]\w*(::[A-Za-z_]\w*)*)$', t)
    if m:
        return m.group(1)
    return 'void*'


def _parse_std_function(ftype):
    """Extract return and args from std::function<RET(ARGS)>"""
    m = re.search(r'std::function\s*<\s*([^>]+)\s*>', ftype)
    if not m:
        return None
    sig = m.group(1).strip()
    msig = re.match(r'([^\(]+)\((.*)\)', sig)
    if not msig:
        return None
    ret = msig.group(1).strip()
    args = msig.group(2).strip()
    arg_list = []
    if args:
        # split on commas naively
        parts = [p.strip() for p in args.split(',') if p.strip()]
        for p in parts:
            arg_list.append(_map_type_to_c(p))
    return_type = _map_type_to_c(ret)
    return return_type, arg_list


def write_c(output_path, header_includes, structs, enums_global=None, include_dir=None):
    """
    Generate C-compatible header with typedefs for structs and enums.
    """
    out = []
    out.append('/* THIS FILE IS AUTO-GENERATED - DO NOT EDIT */')
    out.append('/* Sources: ' + ', '.join(header_includes) + ' */')
    out.append('#ifndef TINYOPT_STRUCTS_H')
    out.append('#define TINYOPT_STRUCTS_H')
    out.append('')
    # Basic C headers for fixed-width ints and bool
    out.append('#include <stdint.h>')
    out.append('#include <stdbool.h>')
    out.append('#include <stddef.h>')
    # Include the stable, hand-maintained C API shapes (params_t, callbacks)
    out.append('#include "structs.h"')
    out.append('')
    out.append('#ifdef __cplusplus')
    out.append('extern "C" {')
    out.append('#endif')
    out.append('')

    # Prepare lookup sets for mapping types to the generated C names
    struct_short_names = [s[1] for s in structs]
    enum_short_names = [e[1] for e in (enums_global or []) if len(e) >= 2]
    struct_short_lc = set([n.lower() for n in struct_short_names])

    # Collect function-pointer typedefs from any struct callbacks
    fp_typedefs = []
    fp_type_map = {}
    for _cpp_name, _py_name, _fields, _enums, callbacks in structs:
        for ftype, fname in callbacks:
            parsed = _parse_std_function(ftype)
            if not parsed:
                continue
            ret, args = parsed
            fp_name = f'{fname}_t'
            args_str = ', '.join(args) if args else 'void'
            fp_decl = f'typedef {ret} (*{fp_name})({args_str});'
            if fname not in fp_type_map:
                fp_type_map[fname] = fp_name
                fp_typedefs.append(fp_decl)

    # Helper: map a raw clang type spelling to a conservative C type string
    def map_raw_type_to_c(ctype):
        # handle std::array<T,N>
        m = re.search(r'std::array\s*<\s*([^,>]+)\s*,\s*(\d+)\s*>', ctype)
        if m:
            inner = m.group(1).strip()
            n = int(m.group(2))
            inner_m = _map_type_to_c(inner)
            inner_base = inner_m.split('::')[-1] if '::' in inner_m else inner_m
            if inner_base in ('float', 'double', 'int'):
                return f'{inner_base}[{n}]'
            return 'void*'

        # Map common STL types to conservative C representations
        if 'std::string' in ctype:
            return 'const char*'

        base = _map_type_to_c(ctype)
        # Accept a larger set of builtin types
        if base in ('double','float','int','size_t','double*','float*','int*','bool',
                    'uint16_t','uint8_t','int32_t','uint32_t','int64_t','uint64_t'):
            return base
        # fallback: return the short typename
        m2 = re.search(r'([A-Za-z_]\w*(::[A-Za-z_]\w*)*)$', ctype)
        if m2:
            return m2.group(1).split('::')[-1]
        return 'void*'

    # Build helper struct definitions (non-main: skip Options/Output)
    struct_defs = []
    seen_helpers = set()
    for cpp_name, py_name, fields, enums, callbacks in structs:
        if py_name.lower() in ('options', 'output'):
            continue
        if py_name in seen_helpers:
            continue
        seen_helpers.add(py_name)
        cname = py_name.lower() + '_t'
        lines = []
        lines.append(f'typedef struct {{')
        for ftype, fname in fields:
            if 'std::function' in ftype:
                if fname in fp_type_map:
                    lines.append(f'  {fp_type_map[fname]} {fname};')
                else:
                    parsed = _parse_std_function(ftype)
                    if parsed:
                        ret, args = parsed
                        fp_name = f'{fname}_t'
                        args_str = ', '.join(args) if args else 'void'
                        fp_decl = f'typedef {ret} (*{fp_name})({args_str});'
                        fp_type_map[fname] = fp_name
                        fp_typedefs.append(fp_decl)
                        lines.append(f'  {fp_name} {fname};')
                    else:
                        lines.append(f'  void* {fname}; /* std::function - unknown signature */')
                continue

            c = map_raw_type_to_c(ftype)
            if c in struct_short_names:
                ctype_c = c.lower() + '_t'
            elif c in enum_short_names:
                ctype_c = c
            else:
                if '[' in c:
                    ctype_c = c
                elif c == 'const char*':
                    ctype_c = 'const char*'
                elif c in ('double','float','int','size_t','bool','int32_t','int64_t','uint32_t','uint64_t','uint16_t','uint8_t'):
                    ctype_c = c
                else:
                    ctype_c = 'void*'

            lines.append(f'  {ctype_c} {fname};')

        lines.append(f'}} {cname};')
        struct_defs.append('\n'.join(lines))

    # Emit global function-pointer typedefs first
    for td in fp_typedefs:
        out.append(td)
    if fp_typedefs:
        out.append('')
    # Emit enums discovered in the parsed headers (only from project headers)
    if enums_global:
        seen_enum_shorts = set()
        for fq, short, enumerators, src in enums_global:
            # Only emit enums from the project's tinyopt namespace or from
            # headers inside the include_dir. Also deduplicate by short name
            # because parsing multiple translation units can produce repeats.
            if short in seen_enum_shorts:
                continue
            if not ('tinyopt' in fq or (include_dir and src and str(Path(src)).startswith(str(Path(include_dir).resolve())))):
                continue
            seen_enum_shorts.add(short)
            out.append(f'typedef enum {short} {{')
            for name, val in enumerators:
                out.append(f'  {name} = {val},')
            out.append(f'}} {short};')
            out.append('')

    # Emit non-main structs (nested helper types) first so they are available
    # when composing the `options_t` and `output_t` definitions below.
    emitted_types = set()

    # Handle known problematic structs with explicit definitions
    # libclang sometimes fails to parse array fields or nested types correctly
    known_structs = {
        'lm_t': '''typedef struct {
  float damping_init;
  float damping_range[2];
  float good_factor;
  float bad_factor;
} lm_t;''',
    }

    for sdef in struct_defs:
        # Extract the typedef name to check if we have an override
        m = re.search(r'} (\w+);', sdef)
        if m:
            struct_name = m.group(1)
            if struct_name in known_structs:
                # Use the known good definition instead
                out.append(known_structs[struct_name])
                out.append('')
                emitted_types.add(struct_name)
                continue

        out.append(sdef)
        out.append('')
        # Extract the typedef name from the struct definition
        # Format is: typedef struct { ... } name_t;
        m = re.search(r'} (\w+);', sdef)
        if m:
            emitted_types.add(m.group(1))

    # Ensure a couple of well-known nested helper types exist in the
    # generated header. In some parsing configurations libclang may not
    # expose the nested helper definition reliably; provide small
    # conservative C-friendly fallbacks so the generated header is
    # self-contained and consumers can compile against it.
    if 'hessian_t' not in emitted_types:
        out.append('typedef struct {')
        out.append('  int use_ldlt;')
        out.append('  int H_is_full;')
        out.append('  float check_min_H_diag;')
        out.append('  int save_last;')
        out.append('} hessian_t;')
        out.append('')
        emitted_types.add('hessian_t')
    if 'lm_t' not in emitted_types:
        out.append('typedef struct {')
        out.append('  float damping_init;')
        out.append('  float damping_range[2];')
        out.append('  float good_factor;')
        out.append('  float bad_factor;')
        out.append('} lm_t;')
        out.append('')
        emitted_types.add('lm_t')

    # As a final safety, if hessian_t wasn't actually emitted into `out`
    # (some parsing runs can produce inconsistent helpers), ensure a
    # conservative hessian_t is present so the following options_t
    # definition can reference it without breaking the C build.
    joined = '\n'.join(out)
    if 'hessian_t' not in joined:
        out.append('typedef struct {')
        out.append('  int use_ldlt;')
        out.append('  int H_is_full;')
        out.append('  float check_min_H_diag;')
        out.append('  int save_last;')
        out.append('} hessian_t;')
        out.append('')
        emitted_types.add('hessian_t')

    # Emit main public structs: options_t and output_t, derived from parsed structs
    options_entry = next((s for s in structs if s[1] == 'Options'), None)
    output_entry = next((s for s in structs if s[1] == 'Output'), None)

    def map_field(ftype, fname):
        t = ftype
        # Name-based heuristics: common semantic fields
        if fname == 'stop_reason':
            return 'int'
        # Only map the specific final_cost field to double; other "cost" names
        # (e.g. the `cost` scaling struct) should remain structured.
        if fname == 'final_cost':
            return 'double'
        # Name-based special cases (stronger than type heuristics)
        if fname in ('max_iters', 'num_residuals', 'num_iters'):
            return 'uint16_t'
        if fname in ('max_total_failures', 'max_consec_failures', 'num_failures', 'num_consec_failures'):
            return 'uint8_t'
        if fname == 'start_time':
            return 'int64_t'
        if fname in ('errs','deltas2','successes'):
            return 'void*'
        if fname in ('final_hessian',):
            return 'void*'
        # If this field corresponds to a parsed std::function, return the typedef
        if fname in fp_type_map:
            return fp_type_map[fname]

        if 'uint16_t' in t:
            return 'uint16_t'
        if 'uint8_t' in t:
            return 'uint8_t'
        if 'bool' in t:
            return 'int'
        if 'float' in t:
            return 'float'
        if 'double' in t:
            return 'double'
        # Check for exact matches of type aliases (not substrings!)
        if re.search(r'\bCost\b', t):
            return 'double'
        if re.search(r'\bScalar\b', t):
            return 'double'
        if 'StopReason' in t:
            return 'int'
        if 'TimePoint' in t:
            return 'int64_t'
        if 'std::array' in t and 'float' in t:
            return 'float[2]'
        if 'std::string' in t:
            return 'const char*'
        m = re.search(r'([A-Za-z_]\w*(::[A-Za-z_]\w*)*)$', t)
        if m:
            base = m.group(1).split('::')[-1]
            if base in struct_short_names:
                return base.lower() + '_t'
            if base in enum_short_names:
                return base
        return 'void*'

    if options_entry:
        _, _, fields, _, _ = options_entry
        out.append('typedef struct {')
        existing_names = set()
        for ftype, fname in fields:
            ctype = map_field(ftype, fname)
            out.append(f'  {ctype} {fname};')
            existing_names.add(fname)
        out.append('} options_t;')
        out.append('')

    if output_entry:
        _, _, fields, _, _ = output_entry
        out.append('typedef struct {')
        for ftype, fname in fields:
            ctype = map_field(ftype, fname)
            out.append(f'  {ctype} {fname};')
        out.append('} output_t;')
        out.append('')
    # Ensure any helper typedefs referenced by the main structs exist
    missing_helpers = set()
    def collect_required_from_entry(entry):
        if not entry:
            return
        _, _, fields, _, _ = entry
        builtin_ints = {'int8_t','int16_t','int32_t','int64_t','uint8_t','uint16_t','uint32_t','uint64_t'}
        for ftype, fname in fields:
            ctype = map_field(ftype, fname)
            if not (isinstance(ctype, str) and ctype.endswith('_t')):
                continue
            short = ctype[:-2]
            # If the short name corresponds to a parsed struct, it'll be emitted
            # in name_to_def; skip in that case. Also skip common builtin typedefs.
            if short.lower() in struct_short_lc:
                continue
            if ctype in builtin_ints or short in ('size', 'int', 'long'):
                continue
            missing_helpers.add(ctype)

    collect_required_from_entry(options_entry)
    collect_required_from_entry(output_entry)

    for h in sorted(missing_helpers):
        # If this helper was already emitted (including helpers that are
        # function-pointer typedefs produced earlier), skip emitting a
        # placeholder. fp typedef names are available in `fp_type_map`.
        if h in emitted_types:
            continue
        if h in set(fp_type_map.values()):
            continue
        # Special-case a couple of known nested types to provide reasonable
        # C-compatible definitions instead of empty placeholders.
        if h == 'hessian_t':
            out.append('typedef struct {')
            out.append('  int use_ldlt;')
            out.append('  int H_is_full;')
            out.append('  float check_min_H_diag;')
            out.append('  int save_last;')
            out.append('} hessian_t;')
            out.append('')
            continue
        if h == 'lm_t':
            out.append('typedef struct {')
            out.append('  float damping_init;')
            out.append('  float damping_range[2];')
            out.append('  float good_factor;')
            out.append('  float bad_factor;')
            out.append('} lm_t;')
            out.append('')
            continue
        out.append(f'typedef struct {{ int _dummy; }} {h};')
        out.append('')

    out.append('#ifdef __cplusplus')
    out.append('}')
    out.append('#endif')
    out.append('')
    out.append('#endif')

    try:
        Path(output_path).write_text('\n'.join(out))
    except Exception as e:
        print(f'Failed to write C bindings to {output_path}: {e}', file=sys.stderr)


def write_conversions(output_path, structs, include_dir=None):
    """
    Emit a C++ source file implementing Convert(...) helpers for Options and Output.
    The file is intended to live in the build tree and be compiled into the C bindings library.
    """
    out = []
    out.append('// Auto-generated conversions - do not edit')
    out.append('#include "generated_structs.h"')
    out.append('#include <tinyopt/optimizers/options.h>')
    out.append('#include <tinyopt/output.h>')
    out.append('#include <chrono>')
    out.append('#include <functional>')
    out.append('using namespace tinyopt;')
    out.append('')

    # Find entries
    options_entry = next((s for s in structs if s[1] == 'Options'), None)
    output_entry = next((s for s in structs if s[1] == 'Output'), None)

    # Build a lookup of parsed struct fields by short name
    struct_map = { s[1]: s for s in structs }

    def emit_assign_cpp_from_c(ftype, fname, parent_path=''):
        """Return a list of assignment lines that copy in.<parent>.<fname> into o.<parent>.<fname>"""
        src = f'in.{parent_path + ("." if parent_path else "")}{fname}'
        dst = f'o.{parent_path + ("." if parent_path else "")}{fname}'
        lines = []

        # std::function -> wrap C function-pointer into std::function<...>
        if 'std::function' in ftype or 'callback' in fname.lower() or 'callback' in ftype:
            m = re.search(r'std::function\s*<\s*([^\(>]+)\s*\(\s*(.*)\s*\)\s*>', ftype)
            parsed = _parse_std_function(ftype)
            if parsed:
                # For safety, do not attempt to auto-generate std::function wrappers
                # that convert arbitrary C function-pointer signatures into complex
                # C++ std::function types (for example those taking Eigen vectors).
                # Emitting such wrappers is fragile and can produce incompatible
                # assignments; instead, leave the C++ std::function default-initialized.
                lines.append(f'  {dst} = decltype({dst})();')
                return lines
            else:
                lines.append(f'  {dst} = decltype({dst})();')
                return lines

        # std::array<T,N>
        marr = re.search(r'std::array\s*<\s*([^,>]+)\s*,\s*(\d+)\s*>', ftype)
        if marr:
            n = int(marr.group(2))
            for i in range(n):
                lines.append(f'  {dst}[{i}] = {src}[{i}];')
            return lines

        # C-style arrays
        mcarr = re.search(r'([A-Za-z_][\w:\<\>\s,]*)\s*\[\s*(\d+)\s*\]', ftype)
        if mcarr:
            n = int(mcarr.group(2))
            for i in range(n):
                lines.append(f'  {dst}[{i}] = {src}[{i}];')
            return lines

        # Nested struct
        mtype = re.search(r'([A-Za-z_]\w*(::[A-Za-z_]\w*)*)\s*$', ftype)
        if mtype:
            base = mtype.group(1).split('::')[-1]
            if base in struct_map and struct_map[base][2] and base not in ('Options', 'Output'):
                inner_fields = struct_map[base][2]
                for ift, ifn in inner_fields:
                    if 'range' in ifn.lower():
                        lines.append(f'  o.{parent_path + ("." if parent_path else "")}{fname}.{ifn}[0] = {src}.{ifn}[0];')
                        lines.append(f'  o.{parent_path + ("." if parent_path else "")}{fname}.{ifn}[1] = {src}.{ifn}[1];')
                        continue
                    inner_parent = parent_path + ('.' + fname if parent_path else fname)
                    inner_lines = emit_assign_cpp_from_c(ift, ifn, parent_path=inner_parent)
                    lines.extend(inner_lines)
                return lines

        # Enums
        if 'StopReason' in ftype or re.search(r'\b[A-Z][A-Za-z0-9_]*\b', ftype):
            lines.append(f'  {dst} = static_cast<decltype({dst})>({src});')
            return lines

        # Bool
        if 'bool' in ftype:
            lines.append(f'  {dst} = {src} != 0;')
            return lines

        # Pointers
        if '*' in ftype or _map_type_to_c(ftype).endswith('*'):
            lines.append(f'  {dst} = reinterpret_cast<decltype({dst})>({src});')
            return lines

        # std::string -> expose C string pointer
        if 'std::string' in ftype:
            # Assign C string (const char*) into C++ std::string safely.
            # `src` originates from the C struct (const char*), so guard
            # against null pointers when constructing the std::string.
            lines.append(f'  {dst} = ({src} ? {src} : std::string());')
            return lines

        # Containers
        if 'vector' in ftype or 'variant' in ftype or 'std::vector' in ftype or 'std::variant' in ftype:
            lines.append(f'  {dst} = nullptr;')
            return lines

        # Numeric primitives
        if any(k in ftype for k in ('float','double','int','size_t','Cost','Scalar','uint')):
            lines.append(f'  {dst} = {src};')
            return lines

        # Fallback
        lines.append(f'  {dst} = {src};')
        return lines

    if options_entry:
        out.append('tinyopt::Options Convert(const options_t &in) {')
        out.append('  tinyopt::Options o;')
        _, _, fields, _, _ = options_entry
        for ftype, fname in fields:
            lines = emit_assign_cpp_from_c(ftype, fname)
            for l in lines:
                out.append(l)
        out.append('  return o;')
        out.append('}')
        out.append('')

    if output_entry:
        out.append('output_t Convert(const tinyopt::Output &in) {')
        out.append('  output_t o;')
        _, _, fields, _, _ = output_entry
        for ftype, fname in fields:
            # TimePoint
            if 'TimePoint' in ftype or fname == 'start_time':
                out.append('  o.start_time = static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(in.start_time.time_since_epoch()).count());')
                continue
            # Enums
            if 'StopReason' in ftype or fname == 'stop_reason':
                out.append(f'  o.{fname} = static_cast<int>(in.{fname});')
                continue
            # std::array
            marr = re.search(r'std::array\s*<\s*([^,>]+)\s*,\s*(\d+)\s*>', ftype)
            if marr:
                n = int(marr.group(2))
                for i in range(n):
                    out.append(f'  o.{fname}[{i}] = in.{fname}[{i}];')
                continue
            # Nested struct
            mtype = re.search(r'([A-Za-z_]\w*(::[A-Za-z_]\w*)*)\s*$', ftype)
            if mtype:
                base = mtype.group(1).split('::')[-1]
                if base in struct_map and base != 'Options' and base != 'Output':
                    inner_fields = struct_map[base][2]
                    for ift, ifn in inner_fields:
                        if 'bool' in ift:
                            out.append(f'  o.{fname}.{ifn} = in.{fname}.{ifn} ? 1 : 0;')
                        else:
                            out.append(f'  o.{fname}.{ifn} = in.{fname}.{ifn};')
                    continue
            # Opaque output fields
            if fname in ('final_hessian','errs','deltas2','successes'):
                out.append(f'  o.{fname} = nullptr;')
                continue
            # Containers
            if any(k in ftype for k in ('vector','variant','std::vector','std::variant')):
                out.append(f'  o.{fname} = nullptr;')
                continue
            # std::function
            if 'std::function' in ftype:
                out.append(f'  o.{fname} = nullptr;')
                continue
            # Pointers
            if '*' in ftype or _map_type_to_c(ftype).endswith('*'):
                out.append(f'  o.{fname} = reinterpret_cast<void*>(in.{fname});')
                continue
            # Primitives
            if any(k in ftype for k in ('float','double','int','size_t','Cost','Scalar','uint')):
                out.append(f'  o.{fname} = in.{fname};')
                continue
            # Fallback
            out.append(f'  /* fallback copy for {fname} */')
            out.append(f'  o.{fname} = in.{fname};')
        out.append('  return o;')
        out.append('}')
        out.append('')

    try:
        Path(output_path).write_text('\n'.join(out))
    except Exception as e:
        print(f'Failed to write conversions to {output_path}: {e}', file=sys.stderr)
