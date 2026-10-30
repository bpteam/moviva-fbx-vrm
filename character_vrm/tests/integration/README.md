# Local integration smoke tests (need real Blender)

Nothing in `tests/unit` or `tests/contract` starts Blender: they cover the converter's logic against a fake
bridge and the VRM Add-on calls against small mock object graphs. **These integration steps are the only thing
that exercises real Blender, the real FBX/glTF importers, real shape keys, the real VRM Add-on and the real
export. They have not been run by the authors of the code; run them once on your machine.**

Setup (Blender 5.2.2 + VRM Add-on 4.7.2): [../../README.md#local-setup-ubuntu](../../README.md#local-setup-ubuntu).
All commands run from `converter/`:

```bash
cd converter
export BLENDER_BIN="$HOME/opt/blender-5.2.2-linux-x64/blender"
```

## 1. Blender

```bash
"$BLENDER_BIN" --version
```

Must print `Blender 5.2.2`.

## 2. Doctor

```bash
python3 -m character_vrm.cli doctor
```

Expected: `expected` equals `detected` for Blender (5.2.2) and the VRM Add-on (4.7.2), `bpy.ops.export_scene.vrm`
present, `Environment OK`, exit code 0. Also check the negative case: with any other Blender, or with the add-on
disabled, it must print `Environment NOT OK` and exit non-zero.

## 3. Inspect a GLB

```bash
python3 -m character_vrm.cli inspect ./fixtures/real-character.glb
```

Check against the model in a Blender UI session: mesh / triangle / material / texture counts, the bone count,
`mapped required: 15/15`, the shape-key count, and that ARKit is `likely` only for models that really carry the 52
ARKit shapes.

## 4. Convert a GLB

```bash
python3 -m character_vrm.cli convert ./fixtures/real-character.glb /tmp/real-character.vrm
```

Expected: exit 0, `/tmp/real-character.vrm` and `/tmp/real-character.vrm.report.json`. Verify in the report:
`status: success`, `humanoid.mappings` (each with confidence and reason), `expressions` (what is `available`,
`generated`, `missing`), `springBones.chains`, `warnings`.

## 5. Convert an FBX

```bash
python3 -m character_vrm.cli convert ./fixtures/real-character.fbx /tmp/real-character-fbx.vrm
```

The importer is `bpy.ops.wm.fbx_import`. If the FBX yields no shape keys after import, the run must print the
`WARNING` block ("no facial shape keys were available after Blender import"), report `expressions.capability:
"none"` and still convert the humanoid. Compare with the GLB of the same character.

## 6. Robustness checks worth doing once

- `character-vrm convert ... --dry-run`: no `.vrm`, no report file.
- `character-vrm map ./fixtures/real-character.glb`: writes `real-character.vrm-map.json`; feed it back with
  `--config` and confirm the same result.
- A model with two armatures: without `--armature` it must stop with `ambiguous-armature` (if both are complete
  humanoids) or convert the primary one and list the other in `warnings`.
- A rig with meaningless bone names (`Bone.001`, ...): if its hierarchy has mirrored limb pairs, the anatomy analysis
  maps it (`humanoid.mappings[].reason` starts with `anatomy analysis`, and the report lists `sidesAssumed: true`
  when no name says left / right: check left and right in Blender). A skeleton with no humanoid structure must still
  stop with `humanoid-incomplete`, and `map` must list the guesses under `_needsReview`.

## 6b. Anatomy analysis and Auto-Author (none of this has been run in real Blender)

Run each step on real characters, for example: a realistic CGTrader / VIVERSE model, the current high-poly model, a
Mixamo-like model, a Character Creator-style model, a model with ARKit morphs, one without facial morphs, one with hair
bones, one without.

```bash
python3 -m character_vrm.cli analyze ./fixtures/real-character.glb --json /tmp/analysis.json
python3 -m character_vrm.cli analyze ./fixtures/real-character.glb --auto-author       # plans only, mutates nothing
python3 -m character_vrm.cli convert ./fixtures/real-character.glb /tmp/auto.vrm --auto-author --geometry-profile video-call
```

Check:

- **Frame and landmarks**: `anatomy.frame` up / forward match the character (rotate the source in Blender and re-run:
  normalised values must not change); eye, nose, mouth and chin landmarks sit on the face when you plot them (add empties
  at `landmarks.*.position`).
- **Existing data wins**: for a model with VRoid / ARKit morphs `authoring.expressions` shows `SOURCE` / `MAPPED` and no
  `Prosopon_*` key exists on the meshes.
- **Generated blink and visemes** (a model without facial morphs, dense face): in Blender's shape-key panel every
  `Prosopon_*` key exists next to the untouched source keys; sliding `Prosopon_BlinkLeft` closes the left lid over the
  eyeball without moving the brow, nose or the other eye; `aa / ih / ou / ee / oh` are visibly different, symmetric and
  do not tear the face; a flat textured mouth or a low-poly face must be *skipped* with a reason, not deformed.
  `face_geometry.eyelids_penetrate_eyeball` is available for an eyelid-into-eyeball check.
- **Hair**: a model with hair bones gets `springBones` chains with a category preset and head / neck / chest / shoulder
  colliders; in the Moviva sandbox the hair lags the head, settles, does not explode, detach or pass through the head.
  Without `--generate-hair-rig` a hair mesh with no bones stays static.
- **Jaw** (a Character Creator FBX): `authoring.shapeKeys` has `Prosopon_JawOpen` on the body, teeth and tongue
  ("jaw rotation 20 deg"), `authoring.augmented` adds it to `aa/oh/ou/ih/ee`, `authoring.result.failed` is empty.
  In the sandbox `aa` 0.4 shows the lower teeth. Checked on Mégane and Amber (CC4, Blender 5.2.2). ActorCore's
  `*_Motion.Fbx` is the light animation file (no teeth mesh, no textures): convert the character FBX instead.
- **Real-time material look** (a Character Creator FBX with its `textures/` folder beside it): `authoring.materials`
  lists cornea / eye with a specular and roughness fix, skin / teeth / hair with `roughness map` and `ao map`, hair with
  `alpha clip 0.35`; `authoring.result.failed` has no `material ...` entry. In the written file the hair materials are
  `alphaMode: MASK`, skin has `metallicRoughnessTexture` and `occlusionTexture`. In the Moviva sandbox the eyes show a
  highlight and the hair no longer shows the head through it. Checked once on Mégane (CC4, Blender 5.2.2).
- **Experimental rigs** (`--generate-finger-rig`, `--generate-hair-rig`, `--generate-face`): only on models you can
  afford to break. Generated bones are `Prosopon_*`, weights of the hand / head bones are only reduced on the planned
  vertices; a mitten hand must be reported *independent finger articulation unavailable*. In the sandbox the hand poses
  (`relaxed`, `open`, `point`, `fist`, `count1..5`, `thumbsUp`, `palmOut`) must stay plausible.
- **Rest pose**: `anatomy.pose` lists the pose type, arm abduction and the head / torso corrections. It is metadata only:
  compare it with how the model stands in the sandbox (nothing consumes it at runtime yet).
- **Output**: still an ordinary VRM 1.0 (`Output.warnings` empty), loadable with no per-model setting.

## 7. Moviva smoke test (the output is an ordinary VRM 1.0)

1. Sandbox: `cd avatar && npm run dev`, top bar → *Load VRM* → `/tmp/real-character.vrm`. It must load through the
   normal loader with no per-model setting; check rest pose (arms down, no twisted limbs), blink and A/I/U/E/O
   lip sync, look-at and hair movement.
2. Extension: popup → *Settings* → *Avatar* → import the same file
   ([extension/README.md](../../../../extension/README.md#use)); the automatic rig calibration
   runs on first load.

Record what differs from expectation in the model's report review. Do not add a converted-model code path to the
runtime: a defect belongs in the converter, in the map file, or in the source asset.
