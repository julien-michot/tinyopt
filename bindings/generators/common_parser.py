#!/usr/bin/env python3
"""
Common clang-based C++ header parser shared by all binding generators.
"""

import sys
import re
from pathlib import Path
import os
import traceback

try:
    from clang import cindex
    HAVE_CLANG = True
except ImportError:
    HAVE_CLANG = False


def try_configure_libclang():
    """Attempt to configure libclang from environment or common paths."""
    if not HAVE_CLANG:
        return False
    try:
        lib_path = os.environ.get('LIBCLANG_PATH')
        if lib_path:
            try:
                cindex.Config.set_library_file(lib_path)
                print(f"Configured libclang from LIBCLANG_PATH={lib_path}")
                return True
            except Exception as e:
                print(f"Failed to set libclang from LIBCLANG_PATH={lib_path}: {e}", file=sys.stderr)

        # Try a few common install locations; allow shell-style globs so we
        # can match versioned library names.  ``os.path.exists`` doesn't
        # expand globs, so use ``glob`` here.
        import glob
        common_paths = [
            # macOS paths
            '/opt/homebrew/opt/llvm/lib/libclang.dylib',
            '/usr/local/opt/llvm/lib/libclang.dylib',
            # Linux paths (Ubuntu, Fedora, Debian)
            '/usr/lib/llvm-*/lib/libclang.so*',
            '/usr/lib/x86_64-linux-gnu/libclang.so*',
            '/usr/lib/libclang.so*',
            '/usr/local/lib/libclang.so*'
        ]
        for pattern in common_paths:
            for p in glob.glob(pattern):
                if not p:
                    continue
                try:
                    cindex.Config.set_library_file(p)
                    print(f"Configured libclang from {p}")
                    return True
                except Exception as e:
                    print(f"Failed to set libclang from {p}: {e}", file=sys.stderr)

        # As a last resort try using ctypes to locate the library.  This often
        # returns a soname like 'libclang.so.1' which may be sufficient.
        try:
            from ctypes.util import find_library
            lib = find_library('clang')
            if lib:
                try:
                    cindex.Config.set_library_file(lib)
                    print(f"Configured libclang from ctypes.find_library: {lib}")
                    return True
                except Exception as e:
                    print(f"Failed to set libclang from ctypes.find_library({lib}): {e}", file=sys.stderr)
        except Exception:
            pass
    except Exception as e:
        print(f"Unexpected error configuring libclang: {e}", file=sys.stderr)
    return False


def generate_from_clang(header_paths, include_dir):
    """
    Parse C++ headers using libclang and extract structs and enums.

    Returns: (structs, enums_global)
        structs: List of tuples (cpp_name, py_name, fields, enums, callbacks)
        enums_global: List of tuples (fully_qualified, short_name, enumerators, source_file)
    """
    index = cindex.Index.create()
    args = ['-x', 'c++', '-std=gnu++20', '-I', str(include_dir)]
    extra = os.environ.get('CLANG_ARGS')
    if extra:
        args.extend(extra.split())

    structs = []
    enums_global = []
    header_set = set([str(Path(h).resolve()) for h in header_paths])

    def visit(node, namespace=''):
        qprefix = namespace
        try:
            if node.kind == cindex.CursorKind.NAMESPACE:
                ns = node.spelling
                if ns:
                    qprefix = f"{namespace}::{ns}" if namespace else ns
                for c in node.get_children():
                    visit(c, qprefix)
                return

            if node.kind == cindex.CursorKind.ENUM_DECL:
                ename = node.spelling
                if ename:
                    # Only collect enums declared in the input header files
                    src_file = str(node.location.file) if node.location and node.location.file else ''
                    if src_file:
                        try:
                            if str(Path(src_file).resolve()) not in header_set:
                                return
                        except Exception:
                            return
                    fq = f"{qprefix}::{ename}" if qprefix else ename
                    enumerators = []
                    for e in node.get_children():
                        if e.kind == cindex.CursorKind.ENUM_CONSTANT_DECL:
                            enumerators.append((e.spelling, e.enum_value))
                    src = src_file
                    enums_global.append((fq, ename, enumerators, src))

            if node.kind in (cindex.CursorKind.STRUCT_DECL, cindex.CursorKind.CLASS_DECL):
                name = node.spelling
                if not name:
                    return
                if not re.match(r'^[A-Za-z_]\w*$', name):
                    return
                src_file = str(node.location.file) if node.location and node.location.file else ''
                if src_file:
                    try:
                        if str(Path(src_file).resolve()) not in header_set:
                            return
                    except Exception:
                        return
                cpp_name = f"{qprefix}::{name}" if qprefix else name
                py_name = name
                fields = []
                enums = []
                callbacks = []
                for c in node.get_children():
                    if c.kind == cindex.CursorKind.FIELD_DECL:
                        ftype = c.type.spelling
                        fname = c.spelling
                        fields.append((ftype, fname))
                        if 'std::function' in ftype:
                            callbacks.append((ftype, fname))
                    elif c.kind == cindex.CursorKind.ENUM_DECL:
                        enum_name = c.spelling
                        enumerators = []
                        for e in c.get_children():
                            if e.kind == cindex.CursorKind.ENUM_CONSTANT_DECL:
                                enumerators.append((e.spelling, e.enum_value))
                        enums.append((enum_name, enumerators))
                structs.append((cpp_name, py_name, fields, enums, callbacks))

                child_prefix = f"{qprefix}::{name}" if qprefix else name
                for c in node.get_children():
                    visit(c, child_prefix)
                return
        except Exception as e:
            # Catch unexpected libclang exceptions (e.g. unknown template kinds)
            # IMPORTANT: avoid accessing properties on `node` that call into
            # libclang (for example `node.kind`) because those can raise the
            # same ValueError we're handling. Probe attributes inside
            # guarded try/except blocks so we can emit a concise warning
            # without triggering another exception or printing a huge stack.
            loc = None
            try:
                loc = node.location
            except Exception:
                loc = None
            try:
                loc_file = getattr(loc, 'file', 'unknown')
                loc_line = getattr(loc, 'line', '?')
            except Exception:
                loc_file = 'unknown'
                loc_line = '?'
            try:
                node_name = node.spelling
            except Exception:
                node_name = '<unknown>'
            # Don't access node.kind directly; it may raise ValueError.
            node_kind = '<unknown>'
            # Short diagnostic message instead of full traceback
            print(f"Warning: skipping AST node {node_name} ({node_kind}) at {loc_file}:{loc_line}: {e}", file=sys.stderr)
            # If this appears frequently it often indicates a libclang /
            # python-clang ABI/version mismatch. Suggest a diagnostic hint.
            print("Hint: 'Unknown template argument kind' warnings often mean your python 'clang' package and system libclang.so are mismatched. Try installing a matching 'clang' pip package or set LIBCLANG_PATH to the libclang that matches the package.", file=sys.stderr)
            return

        for c in node.get_children():
            visit(c, namespace=qprefix)

    for header in header_paths:
        tu = index.parse(str(header), args=args)
        visit(tu.cursor, namespace='')

    return structs, enums_global
