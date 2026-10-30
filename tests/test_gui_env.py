import sys

from moviva_fbx_to_vrm.gui import system_library_path


def test_frozen_window_gives_the_file_manager_the_system_libraries(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/app/_internal")
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    system_library_path()
    assert "LD_LIBRARY_PATH" not in __import__("os").environ

    monkeypatch.setenv("LD_LIBRARY_PATH", "/app/_internal:/opt/lib")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/opt/lib")
    system_library_path()
    env = __import__("os").environ
    assert env["LD_LIBRARY_PATH"] == "/opt/lib" and "LD_LIBRARY_PATH_ORIG" not in env


def test_source_run_keeps_the_environment(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib")
    system_library_path()
    assert __import__("os").environ["LD_LIBRARY_PATH"] == "/opt/lib"
