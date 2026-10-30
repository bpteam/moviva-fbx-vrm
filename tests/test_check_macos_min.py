import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
from check_macos_min import MIN, is_macho, version  # noqa: E402


def test_every_mach_o_kind_is_scanned_and_nothing_else(tmp_path):
    kinds = {"thin64": "cffaedfe", "thin32": "cefaedfe", "fat": "cafebabe", "fat64": "cafebabf",
             "fat_swapped": "bebafeca", "fat64_swapped": "bfbafeca"}
    for name, magic in kinds.items():
        (tmp_path / name).write_bytes(bytes.fromhex(magic) + b"\0" * 12)
    (tmp_path / "script.py").write_text("print('hi')")
    assert {p.name for p in tmp_path.iterdir() if is_macho(p)} == set(kinds)


def test_both_load_commands_give_the_minimum_macos():
    out = ("cmd LC_BUILD_VERSION\n  cmdsize 32\n platform 1\n    minos 14.0\n"
           "cmd LC_VERSION_MIN_MACOSX\n  cmdsize 16\n  version 10.13\n")
    assert [version(a or b) for a, b in MIN.findall(out)] == [(14, 0), (10, 13)]
