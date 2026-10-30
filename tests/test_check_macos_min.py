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


def test_a_file_otool_cannot_read_counts_as_unverified(tmp_path, monkeypatch):
    import subprocess

    import check_macos_min
    runs = {"bad": subprocess.CompletedProcess([], 1, "", "not an object file"),
            "bare": subprocess.CompletedProcess([], 0, "Load command 0\n cmd LC_SEGMENT_64\n", ""),
            "ok": subprocess.CompletedProcess([], 0, "cmd LC_BUILD_VERSION\n minos 11.0\n", "")}
    monkeypatch.setattr(check_macos_min.subprocess, "run", lambda cmd, **kw: runs[Path(cmd[-1]).name])
    for name in runs:
        (tmp_path / name).write_bytes(bytes.fromhex("cffaedfe") + b"\0" * 12)
    assert [check_macos_min.minimum(tmp_path / n) for n in runs] == [None, None, (11, 0)]
    assert check_macos_min.main(str(tmp_path), "12.0") == 1
    (tmp_path / "bad").unlink()
    (tmp_path / "bare").unlink()
    assert check_macos_min.main(str(tmp_path), "12.0") == 0
