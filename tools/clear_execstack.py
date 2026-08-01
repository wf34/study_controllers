#!/usr/bin/env python3
"""Clear the executable-stack flag on venv shared libraries that request it.

drake's bundled libmosek64 is marked PT_GNU_STACK=RWE (a false positive from
assembly objects built without a .note.GNU-stack section). Linux >= 6.x refuses
the mprotect that glibc issues to honour that request, so dlopen fails with:

    ImportError: libmosek64-...so.10.1: cannot enable executable stack as
    shared object requires: Invalid argument

Clearing PF_X on the PT_GNU_STACK program header fixes it. Re-run this after
any `uv sync --reinstall` / drake upgrade, which restores the pristine wheel.

    uv run python tools/clear_execstack.py
"""
import struct
import sys
from pathlib import Path

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def clear(path: Path) -> bool:
    with open(path, "r+b") as f:
        data = bytearray(f.read())
        if data[:4] != b"\x7fELF" or data[4] != 2:
            return False
        end = "<" if data[5] == 1 else ">"
        (e_phoff,) = struct.unpack_from(end + "Q", data, 0x20)
        e_phentsize, e_phnum = struct.unpack_from(end + "HH", data, 0x36)

        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            (p_type,) = struct.unpack_from(end + "I", data, off)
            if p_type != PT_GNU_STACK:
                continue
            (p_flags,) = struct.unpack_from(end + "I", data, off + 4)
            if not p_flags & PF_X:
                return False
            struct.pack_into(end + "I", data, off + 4, p_flags & ~PF_X)
            f.seek(0)
            f.write(data)
            print(f"patched {path} (PT_GNU_STACK {p_flags:#x} -> {p_flags & ~PF_X:#x})")
            return True
    return False


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [Path(sys.prefix) / "lib"]
    patched = 0
    for root in roots:
        targets = [root] if root.is_file() else sorted(root.rglob("*.so*"))
        for lib in targets:
            if lib.is_file() and not lib.is_symlink() and clear(lib):
                patched += 1
    print(f"{patched} librar{'y' if patched == 1 else 'ies'} patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
