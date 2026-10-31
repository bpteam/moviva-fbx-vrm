"""Real-model smoke test of a BUILT app: an unrigged humanoid goes in, a rigged VRM 1.0 comes out.

usage: python packaging/smoke_convert.py <path to the built executable>
The model is a generated mannequin (no armature, no weights). The check reads the conversion report, the @event
lines and the VRM file itself; a CPU-only machine is enough.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_model(path: Path) -> None:
    import bpy
    from character_vrm.tests.fixtures.mannequin import mannequin_points
    pts = mannequin_points(300)
    n = len(pts) // 3 * 3  # arbitrary triangles over the points: only the vertices matter
    mesh = bpy.data.meshes.new("Mannequin")
    mesh.from_pydata([tuple(p) for p in pts[:n]], [], [(i, i + 1, i + 2) for i in range(0, n, 3)])
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("Mannequin", mesh))
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")


def main(exe: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp, "mannequin.glb"), Path(tmp, "mannequin.vrm")
        make_model(src)
        run = subprocess.run([exe, "convert", str(src), str(out), "--auto-rig", "--events"],
                             capture_output=True, text=True, timeout=600)
        print(run.stdout[-3000:], run.stderr[-1500:])
        assert run.returncode == 0, f"exit code {run.returncode}"
        report = json.loads(Path(str(out) + ".report.json").read_text(encoding="utf-8"))
        rig = report["autoRig"]
        assert report["status"] == "success" and rig["status"] in ("PASS", "CHECK"), rig
        assert any('"step":"auto_rig"' in line and '"status":"' in line for line in run.stdout.splitlines())
        data = out.read_bytes()
        assert data[:4] == b"glTF" and b"VRMC_vrm" in data, "not a VRM 1.0 file"
        print(f"smoke OK: {rig['method']}, deformation {rig['status']}, vrm {len(data)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
