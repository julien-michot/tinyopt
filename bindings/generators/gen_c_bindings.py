#!/usr/bin/env python3
"""
Generate C bindings and conversions from C++ headers.
Usage: gen_c_bindings.py <header1> [<header2> ...] <output_c_header> [<output_conversions>] <include_dir>
"""
import sys
from pathlib import Path

# This script is in generators/, add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common_parser import try_configure_libclang, generate_from_clang, HAVE_CLANG
from write_c import write_c, write_conversions


def main(argv):
    if len(argv) < 4:
        print('Usage: gen_c_bindings.py <header1> [<header2> ...] <output_c_header> [<output_conversions>] <include_dir>')
        return 2

    # Detect whether we have conversions output: check if second-to-last arg looks like a file
    possible_conv = argv[-2]
    if '.' in possible_conv or possible_conv.endswith('generated_conversions.cpp'):
        headers = [Path(p) for p in argv[1:-3]]
        out_c = Path(argv[-3])
        out_conv = Path(argv[-2])
        include_dir = Path(argv[-1])
    else:
        headers = [Path(p) for p in argv[1:-2]]
        out_c = Path(argv[-2])
        out_conv = None
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
    write_c(out_c, header_names, structs, enums_global=enums_global, include_dir=include_dir)

    if out_conv:
        write_conversions(out_conv, structs, include_dir=include_dir)
        print(f'Wrote C bindings to {out_c} and conversions to {out_conv} with {len(structs)} struct(s)')
    else:
        print(f'Wrote C bindings to {out_c} with {len(structs)} struct(s)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
