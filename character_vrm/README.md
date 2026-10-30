# character-vrm — FBX / GLB / glTF → VRM 1.0 through headless Blender

Developer tool of the [converter](../README.md) package. It turns a rigged character (a CGTrader model, say) into a
standards-compliant **VRM 1.0** file that Moviva loads through its ordinary custom-avatar path. The runtime gets
no per-model code.

```text
CGTrader model → FBX / GLB / glTF → Blender 5.2.2 (headless) → Moviva Python auto-mapper
               → VRM Add-on for Blender 4.7.2 → VRM 1.0 → Moviva Sandbox / extension
```

All logic is Python. Blender is the importer and scene backend; the **VRM Add-on serializes the file** (there is no
custom GLB / `VRMC_vrm` writer here). The existing FBX → VRMA animation converter is a different tool and is untouched.

**Status.** The pure logic (mapping, expressions, springs, config, report, CLI) and the exact VRM Add-on calls are
unit- and contract-tested without Blender. Real Blender behaviour (importers, shape keys after FBX import, the
add-on's export, spring physics) has **not** been verified by the authors; run
[the integration smoke tests](tests/integration/README.md) locally before relying on it.

## Toolchain (pinned)

[`toolchain.json`](toolchain.json) is the single lock:

| Component | Version |
|---|---|
| OS | Ubuntu Linux x86_64 |
| Blender | **5.2.2** LTS |
| VRM Add-on for Blender | **4.7.2** |
| Target | VRM 1.0 (`VRMC_vrm` 1.0) |

Other Blender or add-on versions are not silently supported: `doctor` fails on them. An upgrade is an explicit change
of `toolchain.json` plus a review of [`blender/compat_v4_7_2.py`](blender/compat_v4_7_2.py), the only module that touches
the add-on's property tree (its docstring lists every property path, each read from the v4.7.2 sources).

### Local setup (Ubuntu)

Blender: download the official **Blender 5.2.2 LTS, Linux x86_64, tar archive**; no system-wide install is needed.

```bash
mkdir -p ~/opt
tar -xf blender-5.2.2-linux-x64.tar.xz -C ~/opt/
export BLENDER_BIN="$HOME/opt/blender-5.2.2-linux-x64/blender"
"$BLENDER_BIN" --version          # Blender 5.2.2
```

VRM Add-on: install the approved **v4.7.2** package from the [Blender Extensions VRM page](https://extensions.blender.org/add-ons/vrm/)
through *Edit → Preferences → Get Extensions*, or download **v4.7.2** from its [version history](https://extensions.blender.org/add-ons/vrm/versions/)
and use *Edit → Preferences → Extensions → Install from Disk*. Select the extension ZIP as-is; do not install the
PyPI/source archive (`bpy_vrm_format-*.zip`). Enable it.
Then:

```bash
cd converter
python3 -m character_vrm.cli doctor
```

Expected output:

```text
Moviva Character Converter

Blender
  expected: 5.2.2
  detected: 5.2.2
  OK

VRM Add-on
  expected: 4.7.2
  detected: 4.7.2
  OK

VRM exporter
  bpy.ops.export_scene.vrm
  OK

Import / export operators
  ...
  OK

Target
  VRM 1.0

Environment OK
```

The add-on version is read from `bl_info["version"]` for the legacy module (`io_scene_vrm`) and from the adjacent
`blender_manifest.toml` for a Blender Extension module (`bl_ext.<repo>.vrm`). `doctor` exits non-zero on any other
Blender or add-on version, on a
missing or disabled add-on, and when `bpy.ops.export_scene.vrm` is unavailable. It also requires
`bpy.ops.import_scene.gltf`, `bpy.ops.wm.fbx_import` and Blender's glTF exporter (the add-on drives it).

The same check without the launcher: `blender --background --python-exit-code 1 --python
converter/character_vrm/blender_entry.py -- doctor`.

## Usage

Run from `converter/` (or `pip install -e converter` to get a `character-vrm` command; the package has no
dependencies). `cli.py` only parses arguments and spawns `$BLENDER_BIN --background ...`; the conversion runs inside
Blender's Python.

```bash
python3 -m character_vrm.cli doctor
python3 -m character_vrm.cli inspect model.glb
python3 -m character_vrm.cli map model.glb                         # writes model.vrm-map.json, converts nothing
python3 -m character_vrm.cli convert model.glb model.vrm
python3 -m character_vrm.cli convert model.fbx model.vrm
python3 -m character_vrm.cli convert model.fbx model.vrm --config model.vrm-map.json
python3 -m character_vrm.cli convert model.fbx model.vrm --dry-run # analyse and report, write nothing
python3 -m character_vrm.cli analyze model.glb                     # anatomy analysis only; modifies nothing
python3 -m character_vrm.cli analyze model.glb --json analysis.json
python3 -m character_vrm.cli convert model.glb model.vrm --auto-author --geometry-profile video-call
```

Blender is resolved as `--blender PATH` (before or after the subcommand) → `$BLENDER_BIN` → `blender` on `PATH`; an
explicit but invalid choice is an error, never a silent fallback. Other options: `--config FILE`, `--armature NAME`,
`--min-confidence 0..1` (default 0.75), `--no-springs`, `--geometry-profile full|video-call|face` (default
`full`) and the [Auto-Author](#human-anatomy-analysis-and-auto-author) switches `--auto-author`, `--generate-face`,
`--generate-finger-rig`, `--generate-hair-rig`. Exit code 0 on success, 1 on a failed conversion / environment
/ non-convertible `inspect`, 2 on a bad command line.

Geometry profiles are an independent conversion contract: `full` keeps the full body, `video-call` means exactly
**WAIST-UP** (head through the anatomical waist), and `face` is a bust (head, shoulders and upper chest). Both
cropped profiles preserve complete left and right arms, hands, fingers, sleeves and rigid accessories attached to
the arm chains. The planner first builds a skinning-based arm mask (`ARM_PROTECTION_WEIGHT = 0.05` in
[`core/config.py`](core/config.py)), then protects shoulder transitions, head and whole hair before applying any
body cut. A polygon containing a protected vertex survives even when its rest-pose position is below the cut; the
full humanoid/helper skeleton is never optimized. Shared materials and texture atlases are retained. The same
profile can be written to a map file as `"geometry": {"profile": "video-call"}`; unsafe `dropObjects` overrides fail.

Dry-run and reports show the profile's semantic frame, detected or fallback landmark source, safety margin, cut
plane, protected arm vertices and removed/preserved objects.

A successful `convert` writes `model.vrm` and `model.vrm.report.json` ([report](#conversion-report)). A failed one
writes the report with `status: failed` and the error. "The `.vrm` exists" is not the success criterion: the export must
finish, the required humanoid must be mapped with enough confidence, and the **written file is re-read** and checked
(GLB container, `VRMC_vrm` 1.0, required humanoid nodes, skins kept, materials/images present, expression binds and
spring joints referencing valid nodes).

## What happens

| Stage | Where | What |
|---|---|---|
| 1 clean scene | `blender/importer.py` | every object and datablock removed programmatically; no startup cube, selection, active object or UI context |
| 2 import | `blender/importer.py` | GLB/glTF: `bpy.ops.import_scene.gltf`; FBX: `bpy.ops.wm.fbx_import` (no FBX2glTF) |
| 3 inspect | `blender/scene_inspector.py` → `core/model.py` | armatures, bones (world positions), skinned meshes, vertex groups, **shape keys that survived the import**, materials, images, triangles |
| 4 humanoid | `core/humanoid.py` | see below |
| 4b anatomy | `core/anatomy/` | one canonical `HumanModelAnalysis` (frame, measurements, landmarks, skeleton evidence, hands, face, hair, materials); bones the name resolver could not map are filled in from topology when confident. Always runs, mutates nothing |
| 4c author plan | `core/authoring/` | only with `--auto-author` / `--generate-*`: rest-pose calibration, generated blink / visemes, hair springs, colliders, real-time material look, experimental rigs, as inspectable plans that are validated before anything is mutated |
| 4d mutate | `blender/anatomy_adapter.py` | applies the plans (`Prosopon_` shape keys / bones, material edits), then validates the result; failures are recorded and the affected bindings dropped |
| 5 preserve | `core/geometry.py` + `blender/scene_inspector.py` | the source rig is never rebuilt: no new armature, no auto-weighting, vertex groups untouched; the geometry planner protects arms/hands, shoulder transitions, head and hair before cropping static body faces; twist / helper / breast / cloth / hair / corrective bones stay as they are (simply not humanoid) |
| 6-9, 11 configure | `blender/compat_v4_7_2.py` | `spec_version = "1.0"`, human bones, expressions, LookAt, spring bones, meta |
| 11b slim | `blender/slim.py` | removes shape keys no expression binds (Basis and Moviva's VRoid `Fcl_` face morphs stay), scales textures down to 1024 px; a Character Creator model drops from 819 to 170 MB |
| 12 export | `blender/compat_v4_7_2.py` | `bpy.ops.export_scene.vrm(filepath=..., armature_object_name=..., export_try_sparse_sk=True)`: shape keys store only the vertices they move; must return `{'FINISHED'}` |

Blender's own bone-influence policy applies (the add-on's compatible maximum of four influences per vertex).
`--all-bone-influences` is deliberately not implemented until it is verified against three-vrm / Moviva.

### Humanoid mapping

Priority, a lower level never overwrites a higher one, and no source bone is used twice:

1. **explicit config** (`humanoid` in the map file), confidence 1.0
2. **known rig profile**: Mixamo, Character Creator / Reallusion (`CC_Base_*`), Rigify (`DEF-` and metarig names),
   VRoid (`J_Bip_*`), 0.99 ([`profiles/`](profiles/))
3. **normalised names** ([`core/names.py`](core/names.py)): `mixamorig:LeftArm`, `CC_Base_L_Upperarm`, `upper_arm.L`,
   `LeftUpperArm`, `L_UpperArm` all parse to *left upper arm*; twist, roll, helper and end-site bones are excluded
4. **hierarchy / topology**: hips as the common ancestor of head and legs, the spine chain between hips and head
   (first bone `spine`, second `chest`, last `upperChest`; a twist-named neck joint as in CC is accepted), shoulders,
   missing elbow / knee / wrist / foot bones from the chain, finger chains from their first joint. A derived bone is
   never more trusted than the bones it was derived from
5. **anatomical position** (arms and legs from lateral / vertical placement, five fingers by distance from the thumb),
   confidence 0.6 by design
6. **anatomy analysis** ([below](#human-anatomy-analysis-and-auto-author)): topology (mirrored limb pairs, torso chain,
   head branch), position, symmetry and skinning fused per bone; it fills only bones the levels above left unmapped or
   below the gate, and only at confidence >= 0.80. It maps `Bone.001, Bone.002, ...` skeletons without a map file

Every mapping is a `BoneMapping(vrm_bone, source_bone, confidence, reason)`. A **required** bone below the threshold
(default 0.75) fails the conversion; an optional one is dropped with a warning. Position-based guesses therefore never
convert on their own: `map` lists them under `_needsReview`, and you move them into `humanoid` once you have checked
them (or pass `--min-confidence`). Required: hips, spine, head, both arms (upper, lower, hand), both legs (upper, lower,
foot). Also checked: VRM parent rules (nearest mapped ancestor), and left/right against X position (warning).

Multiple armatures are scored (required-bone coverage, skinned triangles, bone count). Two complete, comparable
candidates with skinned meshes fail with `ambiguous-armature` unless `--armature` / `armature` picks one; a clearly
dominant one wins and the others, plus the meshes they deform, are excluded from the export.

### Expressions

Only **existing** shape keys are used here; source shape keys are never modified (generated keys are the
[Auto-Author](#human-anatomy-analysis-and-auto-author)'s, namespaced `Prosopon_`).
Precedence: explicit config > direct name > compound of known keys > recognised motion > missing.

- names are matched namespace-free and side-normalised ([`core/expressions.py`](core/expressions.py) `canon`):
  `Face_Blendshape.Fcl_MTH_A` = `Fcl_MTH_A`, `vrc.v_aa` = `v_aa`, `Mouth_Smile_L` = `mouthSmileLeft`
- direct: VRoid `Fcl_*`, `A/I/U/E/O`, Oculus-style `viseme_*`, VRChat `vrc.v_*`, Daz `eCTRLv*`, MMD `あいうえお` /
  `まばたき`, generic blink / emotion names, CC `Eye_Blink_L/R` ([`profiles/`](profiles/))
- compound: rig recipes first, then ARKit 52. Character Creator 4
  ([`profiles/character_creator.py`](profiles/character_creator.py)): `aa` = `V_Open` + `Jaw_Open`, `ih`/`ee` from
  `V_Wide`, `ou`/`oh` from `V_Tight_O`, each with its share of `Jaw_Open` (CC visemes shape the lips and expect the
  jaw *bone* to open the mouth; a VRM expression cannot turn a bone), emotions and eye direction from the CC keys.
  ARKit ([`profiles/arkit.py`](profiles/arkit.py)): `aa` = `jawOpen`; `oh` = `mouthFunnel` + `jawOpen`; `ou` =
  `mouthPucker` + `jawOpen`; `ih` / `ee` from `mouthSmileLeft/Right`; blink from `eyeBlinkLeft/Right`. A recipe applies
  only when its required keys exist; the weights are heuristics, tunable there or per model in the map file
- recognised motion ([`core/anatomy/morph_signatures.py`](core/anatomy/morph_signatures.py)): only for keys whose names
  say nothing (`Key 17`, `Shape.004`; a key named `Fcl_MTH_Joy` is an emotion even if it opens the mouth), each key's
  deltas are measured on the analysed face. A key that closes one eyelid and leaves the
  rest alone is that side's blink; symmetric mouth-only keys are assigned jointly to `aa/ih/ee/ou/oh` by the nearest
  prototype (open, spread, forward, corner lift; calibrated on VRoid keys), only when some key really opens the mouth.
  Kind `motion`, status `inferred`
- `blink` is composed from `blinkLeft` + `blinkRight` when only the halves exist
- a bind is added on every mesh that carries the key
- if the source has no facial shape keys nothing is generated. For **FBX** this is reported as a `WARNING`
  (`fbx-no-shape-keys`): "no facial shape keys were available after Blender import … Prefer a GLB/glTF source
  containing morph targets when available." What survived is decided by Blender's importer, not assumed
- the report says `available` (direct / config), `generated` (compound) or `inferred` (motion) per expression and an
  overall
  `capability` (`full`, `partial`, `none`)

### LookAt, spring bones, materials, meta

- **LookAt**: `bone` when both eye bones are mapped, `expression` when lookUp/Down/Left/Right are all available,
  otherwise a warning (never fatal). `offsetFromHeadBone` is not derived: the add-on stores it in an axis convention
  that depends on the head bone's axes, so it is only passed through when set in the map file.
- **Spring bones** (best effort, conservative): non-humanoid, skin-weighted bones whose tokenised names contain
  `hair`, `bang(s)`, `fringe`, `ponytail`, `braid` (high confidence) or `strand` (medium) become one spring per chain
  (`HairBack01 → 02 → 03`). `lock` and other ambiguous or branching chains are reported under `skipped`, never
  simulated, unless listed in `springBones.include`. Defaults (`stiffness`, `drag_force`, `gravity_power`,
  `hit_radius`) and all thresholds live in [`core/config.py`](core/config.py) (`DEFAULT_SPRING_PRESET`) and can be
  overridden in the map file; the hit radius scales with skeleton height. With chains present, a head sphere and a
  chest sphere derived from bone lengths are added as collider groups (`springBones.colliders: false` to skip).
  Disable all of it with `--no-springs` or `springBones.enabled: false`.
- **Materials**: not converted to MToon. Blender materials go through the add-on's normal glTF PBR path; the report
  lists preserved PBR channels and warns about procedural nodes and non-Principled shaders that glTF cannot carry.
  With `--auto-author` the [real-time material look](#real-time-material-look) repairs what an FBX import loses.
- **Meta**: only what the map file states plus a name (default: output file stem). Licence fields keep the add-on's
  restrictive defaults; nothing is inferred from the asset.
- Warnings, never limits: very high triangle / material counts, huge textures, missing fingers / eyes / shape keys /
  hair, a skeleton height outside 0.5–3 m (the rig is not rescaled).

## Map file

`model.vrm-map.json`; `map` writes a suggestion, entries starting with `_` are annotations. Explicit entries always
win over heuristics; unknown keys are rejected.

```json
{
  "armature": "Armature",
  "humanoid": { "hips": "CC_Base_Hip", "spine": "CC_Base_Waist", "chest": "CC_Base_Spine01", "head": "CC_Base_Head" },
  "expressions": {
    "aa": { "Mouth_Open": 0.8, "jawOpen": 0.2 },
    "blink": ["Blink_L", "Blink_R"],
    "happy": "Smile",
    "oh": null
  },
  "lookAt": { "type": "bone", "offsetFromHeadBone": [0, 0, 0] },
  "springBones": {
    "enabled": true,
    "include": ["Wisp01"],
    "ignore": ["HairFront01"],
    "preset": { "stiffness": 1.0, "dragForce": 0.5, "gravityPower": 0.1, "hitRadius": 0.01 },
    "colliders": true
  },
  "meta": { "name": "Character", "version": "1.0", "authors": ["..."], "commercialUsage": "personalNonProfit" }
}
```

`autoAuthor` (`enabled`, `generateFace`, `generateFingerRig`, `generateHairRig`, `confidence: {threshold: 0..1}`) is the
file form of the Auto-Author switches; the command-line flags win.

`humanoid` keys are VRM 1.0 bone names (`leftUpperArm`, `rightIndexProximal`, …). An expression is a shape key name, a
list of names (weight 1) or `{name: weight}`; `null` disables it. `lookAt.type` is `bone`, `expression` or `none`.
`meta` accepts the add-on's fields (`avatarPermission`, `commercialUsage`, `creditNotation`, `modification`,
`allowRedistribution`, …) and validates their enum values.

## Human anatomy analysis and Auto-Author

The story is [docs/history/us-anatomy-analyzer.md](../../docs/history/us-anatomy-analyzer.md). The design rule: **the analyzer establishes
facts and probabilities; the Auto-Author decides what to map and what to add.** They never mix.

```text
inspection -> Human Anatomy Analyzer -> HumanModelAnalysis -> Avatar Auto-Author -> plans -> validate -> mutate -> validate
```

`HumanModelAnalysis` ([`core/anatomy/model.py`](core/anatomy/model.py)) is pure data (no `bpy` objects):

| Part | What it holds |
|---|---|
| `frame` | anatomical frame (`up` hips → head, `left`/`right` from shoulder / hip / foot pairs, `forward` = `left` × `up`, cross-checked with the toes); independent of FBX Y-up, Blender Z-up, object rotation, scale and origin. Mirrored rigs and assumed sides lower its confidence and warn. Without names, the sides come from the feet (ankle → toes gives forward), not from the +X convention |
| `measurements` | height, head / shoulder / chest / waist / hip widths, torso, arm, leg, hand and foot lengths from the skeleton and mesh slices; every value has source and confidence; `normalized()` divides by height (no absolute metres) |
| `landmarks` | head centre / top, eyes, nose, mouth, chin, neck, shoulders, chest, waist, hips, elbows, wrists, hand centres, knees, ankles. **Geometry profiles read the waist, shoulder and upper-chest levels from here**, not from their own copy of the anatomy |
| `skeleton` | per bone: `topology`, `position`, `symmetry`, `skinning` and `name` evidence kept un-collapsed, fused with a veto (one strongly contradicting signal caps the result at 0.6); names and topology disagreeing is a reported conflict and names win. Topology without names ([`limbs.py`](core/anatomy/limbs.py)): legs are the ancestor paths to the lowest bones, arms to the widest ones (continued to the finger tips), the hand is the first bone with three finger *chains*, so twist / roll / aim helpers and cloth chains beside the limbs never take their place. Eyes and jaw without names ([`facebones.py`](core/anatomy/facebones.py)): a mirrored pair of compact skin clusters in front of the head; the root bone that carries the lower face down to the chin |
| `left_hand`, `right_hand` | finger chains classified by geometry (the thumb deviates most in direction; the rest are ordered from it), capability state `bones-present` / `finger-bones-missing-geometry-separable` / `fused-fingers`, symmetry between the hands |
| `face` | landmarks from evidence, see [Face analysis](#face-analysis); the canonical template only places what nothing measures (brows without brow geometry, cheeks, jaw corners, nostrils); mouth capability (jaw bone / morphs / geometry) |
| `hair` | per *surface* (a mesh, or one material slot of a multi-material mesh): `HAIR_CONFIDENT` / `HAIR_POSSIBLE` / `NOT_HAIR` from name, material, texture, head proximity, scalp attachment, hair bones, alpha and, without names, the outer shell of the head (reaches the very top, wraps behind the skull, many strands or cut-out cards); hair type (solid mass, clump, ponytail, braid, alpha cards); bone chains with poor names. Runs *before* the face so hair is never read as face skin or head size |
| `materials` | skin, hair, eye, cornea, eyelash, eyebrow, teeth, tongue, mouth interior, clothes, other, each with confidence and evidence; metadata only, shaders are not touched |

Every inferred or generated item carries `confidence` and an `AnalysisSource` (`explicit`, `rig-profile`, `skeleton`,
`geometry`, `material`, `generated`); the report distinguishes **SOURCE / MAPPED / INFERRED / GENERATED / MISSING**.
The gates live in one table, [`core/anatomy/confidence.py`](core/anatomy/confidence.py) (`automatic_mapping` 0.80,
`generated_face` 0.85, `generated_fingers` 0.90, `generated_hair` 0.90, ...), overridable per model with
`autoAuthor.confidence` in the map file. Below a gate the result is *keep / skip / warn*, never a destructive guess.

### Face analysis

[`core/anatomy/face.py`](core/anatomy/face.py) with [`mouth.py`](core/anatomy/mouth.py), [`eyes.py`](core/anatomy/eyes.py),
[`surfaces.py`](core/anatomy/surfaces.py):

- **region**: vertices skinned to the head *and every bone below it* (CC skins the face to `CC_Base_FacialBone` /
  `UpperJaw`, not to the head), with a role per material slot (skin, eye, teeth / mouth interior, lash, brow, hair):
  a VRoid `Face` mesh carries skin, eyes, lashes and the mouth in one mesh. Mouth-interior surfaces up at the eyes are
  the atlas' eyeliner and brows (older VRoid puts them in `FaceMouth`)
- **eyes**: the geometry the eye bones carry, else the eye surfaces; never the eye joints alone (VRoid places
  `J_Adj_*_FaceEye` well inside the head)
- **chin and nose**: the midline front profile
- **lip seam**, every source proposes one and the consensus wins (the largest summed confidence of estimates that
  agree; a lone stronger outlier loses): an existing mouth-opening morph (the highest sharp step in its downward
  motion), the skin's lip opening (a small boundary loop with skin above and below it, after welding split
  vertices), the jaw skinning (the height that best separates jaw-weighted from other skin in a narrow strip on the
  midline, `JAW_MIDLINE_HALF` / `JAW_SEAM_AGREEMENT` in [`mouth.py`](core/anatomy/mouth.py); the corners, where the
  lip line curves and the weights fade, and the inner roll of the lips do not blur it), the mouth interior / teeth,
  the lip concavity on the profile (scored by depth and by its usual place between chin and nose), the template
- **eyelids**: an existing blink morph (what it pulls down), the skin's eye opening (the tightest loop enclosing the
  eye centre; a lash / eyeliner strip above it is the upper lid of stylised eyes), the visible eye surface
- **confidence** is plausibility, not surface coverage: a landmark behind the front half of the head is capped at
  0.2, and chin / mouth / nose / eyes out of order cap them at 0.4. The report lists every mouth candidate, the chosen
  source and the method of each landmark

### What `--auto-author` does

| Feature | Default | Switch |
|---|---|---|
| map existing data, gap-fill weak-named bones (0.80) | on (always) | |
| rest-pose analysis and calibration plan (metadata; T / A / relaxed / non-standard, head tilt, torso lean; relaxed-arm target = the runtime's `calibrateRig` numbers) | on | `--auto-author` |
| existing hair chains → spring bones with per-category presets ([`authoring/springbones.py`](core/authoring/springbones.py)) | on | `--auto-author` |
| anatomical colliders: head, neck, upper chest, shoulders (radii = fractions of measured widths) | on | `--auto-author` |
| generated blink `Prosopon_BlinkLeft/Right` and visemes `Prosopon_aa/ih/ou/ee/oh`, only for missing ones, only at confidence >= 0.85 and with enough eyelid / mouth vertices. The blink region is the measured eye (stylised eyes are far larger than a head-width fraction), capped below the brow. The jaw part of a viseme is a rotation about the jaw joint weighted by each vertex's jaw skin weight when the rig has a jaw carrying the lower lip (lower teeth follow, upper lip and teeth stay); otherwise the face below the measured seam is translated | on | `--auto-author` |
| `Prosopon_JawOpen`: the jaw turned by an anatomical angle (`JAW_OPEN_DEG` = 20 degrees about the jaw joint, a wide spoken /a/; running speech at lip-sync weights 0.3-0.5 lands at 6-12 degrees, as measured in speech), added to *existing* visemes by `JAW_SHARE` (`aa` 1.0, `oh` 0.75, `ou`/`ih` 0.35, `ee` 0.25) when the existing `aa` does not open the lips (under a quarter of that key's lip travel), as Character Creator's lip-only visemes. Needs only a jaw bone whose skinning carries the lower lip, whatever seam the morphs suggest ([`visemes.py`](core/authoring/visemes.py)) | on | `--auto-author` |
| real-time material look ([below](#real-time-material-look)): dead specular, implausible roughness, blended hair cards, roughness / AO maps shipped beside the file | on | `--auto-author` |
| generated smile / sad / surprise / brows / squint (`browUp`, `browDown`, `squint` are authored as shape keys but have no VRM preset to bind to) | off | `--generate-face` |
| generated finger bones for hands whose fingers exist only as geometry (needs five clearly separated regions; a mitten is reported as *independent finger articulation unavailable*) | off, experimental | `--generate-finger-rig` |
| generated hair bones for clear long clumps / ponytails / braids (alpha-card masses stay static) | off, experimental | `--generate-hair-rig` |

### Real-time material look

[`core/authoring/materials.py`](core/authoring/materials.py) plans, [`blender/material_author.py`](blender/material_author.py)
applies. FBX import (Character Creator, ActorCore, most game exports) keeps the textures but not the look a
real-time PBR renderer needs, so the avatar reads as plastic, with dead eyes and see-through hair. Per material
(role from the semantic classification; eye occlusion and tear-line overlays are left alone), and **only where the
imported value is measurably wrong**, so a good GLB passes unchanged:

- specular: an effective specular (`Specular IOR Level` x brightest `Specular Tint` channel) under `DEAD_SPECULAR`
  becomes the role's value with a white tint: the eyes get their highlight back (FBX imports them with a black tint);
- roughness: a constant outside the role's plausible range becomes the role's value (a cornea glossy at 0.05, a scalp
  not a mirror); a roughness map is never replaced;
- maps: `<Material>_roughness` / `<Material>_ao` images under the source file's folder (the Character Creator export
  layout) are connected to an unlinked `Roughness` and to the exporter's `glTF Material Output` occlusion, scaled to
  at most `MAX_MAP_SIZE`. Maps under a folder named after the source file win (several exports side by side); two
  folders of one model holding the same material name are told apart by the mesh name in the path;
  still ambiguous is skipped and reported, never guessed;
- hair cards that blend become alpha clip (glTF `MASK`, `alpha_cutoff` 0.35) through the exporter's
  `1 - (alpha < cutoff)` node pattern: a renderer draws blended surfaces without depth writes, so layered cards of
  one mesh show through each other. The scalp cap stays blended, so the hairline still fades into the skin.

The values per role are `LOOKS` in that file. The report's `authoring.materials` lists every edit with its role and
changes; `authoring.result.materials` what was applied.

Any `--generate-*` flag implies `--auto-author`. Generated shape keys and bones are namespaced `Prosopon_`, never
overwrite a source item, are localised, symmetric where appropriate and scaled by mouth width / lid gap / head height;
each plan is validated (offset size, namespace, protected landmarks such as brows and nose) before Blender is touched,
and the result is validated afterwards (a key that does not match its plan is removed and not bound). The output is an
ordinary VRM 1.0; the runtime needs no per-model code.

`analyze` prints the analysis (and the plan when the switches are given) and can write it as JSON; that JSON is the same
`anatomy` section a conversion report holds (one source of truth, no second file).

## Conversion report

`<output>.report.json`: `status` (`success` / `failed` / `dry-run`), `toolchain`, `source` (path, format, sha256),
`geometry` (counts, armature scores, excluded meshes, selected profile, landmark and protected-arm mask), `humanoid` (profile, every mapping with confidence and reason,
`missingRequired`, `belowThreshold`), `expressions` (capability, ARKit coverage, per-expression status / kind / binds,
`missing`), `lookAt`, `springBones` (chains, colliders, `skipped` with reasons), `materials`, `output` (what was read
back from the written file), `anatomy` (the per-avatar calibration artifact, see below), `authoring` (what the
Auto-Author planned / generated / skipped, material edits included), `warnings` (`code` + message) and `errors`.

## Code layout

```text
character_vrm/
├── cli.py · launcher.py · blender_entry.py · commands.py   argument parsing / spawning / in-Blender entry / command logic
├── toolchain.json · toolchain.py                            the pin and the pure doctor evaluation
├── core/        model, config, geometry, names, humanoid, inspector, expressions, lookat, springbones, materials,
│                validation, report, pipeline (ConverterCore)  — pure Python, never imports bpy
│   ├── gltf_inspect  pure-Python GLB / VRM reader → CharacterInspection (analysis and benchmark without Blender)
│   ├── anatomy/     model (HumanModelAnalysis), frame, measurements, landmarks, skeleton, limbs, facebones, hands,
│   │                surfaces, face, facegeo, mouth, eyes, morph_signatures, regions, hair, materials, confidence,
│   │                report — facts and probabilities only
│   └── authoring/   plan (AutoAuthorPlan), rest_pose, fingers, expressions, face_plan, facefields, blink, visemes,
│                    hair_rig, springbones, colliders, materials, report — plans, never mutations
├── profiles/    generic, mixamo, character_creator, rigify, vroid, arkit
├── benchmark/   ground truth from authored VRMs and the scoring passes (named / blind / strip)
├── blender/     bridge (BlenderBridge protocol + RealBlenderBridge), importer, scene_inspector, vrm_addon,
│                compat_v4_7_2 (the only file that touches vrm_addon_extension), anatomy_adapter, shape_key_author,
│                rig_author, hair_author, face_geometry, material_author (apply the authoring plans)
└── tests/       unit/ · contract/ · fixtures/ · fakes.py · integration/README.md (real Blender, run locally)
```

`ConverterCore → BlenderBridge → RealBlenderBridge → bpy` in production; `ConverterCore → FakeBlenderBridge` in tests.
`bpy` is imported only inside functions of the modules under `blender/`, so `import character_vrm` works without
Blender (a contract test enforces this, and that nothing outside `compat_v4_7_2.py` mentions `vrm_addon_extension`).

## Tests

```bash
cd converter
python3 -m pip install pytest      # once
python3 -m pytest                  # unit + contract, no Blender needed
```

Unit tests cover name normalisation, humanoid profiles, geometry profiles and protected-arm polygon planning, hierarchy mapping, confidence and override priority, required-bone
validation, ARKit / A-I-U-E-O / blink / compound expressions, hair detection, chain grouping, spring overrides,
reports, CLI parsing, Blender resolution and toolchain checks, and the anatomy / authoring layers: frame and
measurement invariance to rotation / scale / origin, topology-only skeleton mapping, finger classification and
segmentation (separated, mitten, no thumb), face template fitting, generated blink / viseme / expression plans
(locality, symmetry, anatomy scaling, namespace, gating, existing-wins), hair detection and spring presets, material
semantics, colliders, rest pose, plan validation and the analyse -> plan -> mutate -> validate flow through the fake
bridge. The authoring adapters have contract tests against tiny hand-written object graphs; none of that runs Blender. Contract tests check that `compat_v4_7_2.py` issues the
expected add-on calls (spec version, humanoid, expressions, springs, meta, export operator) against mock objects.
They do not prove that the real add-on accepts them; that is what
[the integration smoke tests](tests/integration/README.md) are for.

### Ground-truth benchmark

An authored VRM already says what the analyzer has to find, so it is the test set. `character_vrm.benchmark` reads a
VRM with the pure-Python reader ([`core/gltf_inspect.py`](core/gltf_inspect.py): Blender's axis conversion, bind
pose, skin weights, material slots, sparse morph deltas) and runs the real `analyze` pipeline on it:

```bash
cd converter
python3 -m character_vrm.benchmark ../avatar/public/models/mia.vrm other.vrm --json scores.json
```

Truth ([`benchmark/truth.py`](benchmark/truth.py)): the humanoid map, the expression binds, hair by material, the eye
centres (geometry on the eye bones), the upper eyelids (what each half-blink pulls down), the lip seam and corners
(what `aa` opens). A sidecar `<model>.truth.json` names the morphs when the VRM presets are not authored truth, takes
the seam from the jaw skinning (`"mouthTruth": "jaw"`, for rigs whose mouth morphs are correctives on a jaw bone) or
the corners from one-sided smiles (`"cornerKeys"`). Three passes per model: `named` (as authored), `blind` (every bone,
mesh, material and shape-key name replaced by an opaque id) and `strip` (the truth visemes / blinks removed, Auto-Author
on: generated keys compared with the authored ones by moved-region IoU and leakage into hair / the other region).
Landmark errors are in interocular distances. `tests/unit/test_benchmark_regression.py` holds the bundled avatar to
thresholds; `PROSOPON_BENCHMARK_MODELS=a.vrm:b.vrm` adds local models to it.

## Known limitations

- Real-Blender behaviour is unverified (see above). The Blender 5.2 calls used outside the add-on (`bpy.ops.wm.fbx_import`,
  glTF importer, node and modifier access) follow the documented API and are the first thing to check if a run fails.
- No retopology, decimation, LODs, texture generation, cloth or Cycles / Poiyomi → WebGL material conversion. Generated
  face morphs, finger bones and hair bones exist only behind the [Auto-Author](#human-anatomy-analysis-and-auto-author)
  switches, are confidence-gated and have not been run in real Blender. A mitten hand or a flat textured mouth is
  reported as unavailable, never faked.
- The anatomy analyzer works on the imported bind pose and the vertex data of the inspection. It is scored against
  authored VRMs (two VRoid characters and one Character Creator 4 conversion so far; mouth within 0.1 interocular
  distance with names and 0.2 without), not yet against Blender's own FBX import of a purchased model. A head with no
  facial vertices skinned to the head or below it gets no face analysis. Viseme recognition by motion is calibrated on
  VRoid keys; lip-corrective sets (CC) stay unrecognised by design. Stylised blinks generated from geometry overlap the
  authored ones only partly (the authored ones also move the eye surfaces, which generation never does). Rest-pose
  calibration is reported as metadata (`anatomical-frame` quaternions); nothing applies it at
  runtime yet and `RigCalibration.rest` (parent-local) is not written from it.
- Real-Blender geometry mutation is not verified here; the pure plan is tested without Blender. Face deletion keeps
  vertices (and therefore Basis/shape-key topology) aligned, and intentionally does not crop shared texture atlases.
- Compound expressions and hair chains are heuristics; review the report for assets you care about.
- Without left / right names the sides come from the feet (ankle → toes); only a rig whose feet give no direction
  falls back to the +X import convention, with a warning.
- The rig is not rescaled; a centimetre-scale FBX exports as a giant avatar (warned via `scale-suspicious`).
