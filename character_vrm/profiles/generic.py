"""Generic humanoid vocabulary: bone-name tokens and common shape-key names."""

from __future__ import annotations

# --- bone names (consumed by core/names.py) ---------------------------------
#: prefix / namespace tokens that carry no anatomical meaning
IGNORED_TOKENS = frozenset({
    "cc", "base", "j", "bip", "bip01", "bip001", "adj", "sec", "def", "org", "f", "c", "center", "armature",
})
LEFT_TOKENS = frozenset({"l", "left", "lt"})
RIGHT_TOKENS = frozenset({"r", "right", "rt"})
#: adjacent token pairs merged into one anatomical token
MERGES: dict[tuple[str, str], str] = {
    ("upper", "arm"): "upperarm", ("up", "arm"): "upperarm",
    ("lower", "arm"): "lowerarm", ("fore", "arm"): "forearm",
    ("upper", "leg"): "upperleg", ("up", "leg"): "upperleg",
    ("lower", "leg"): "lowerleg",
    ("upper", "chest"): "upperchest",
    ("toe", "base"): "toes", ("eye", "ball"): "eye",
}
#: token -> anatomical part
PARTS: dict[str, str] = {
    "hips": "hips", "hip": "hips", "pelvis": "pelvis",
    "spine": "spine", "abdomen": "spine", "waist": "spine", "torso": "spine",
    "chest": "chest", "ribcage": "chest", "thorax": "chest", "upperchest": "upperchest",
    "neck": "neck", "head": "head", "jaw": "jaw",
    "eye": "eye", "eyeball": "eye",
    "shoulder": "shoulder", "clavicle": "shoulder", "collar": "shoulder",
    "upperarm": "upperarm", "arm": "upperarm", "forearm": "lowerarm", "lowerarm": "lowerarm",
    "hand": "hand",
    "thigh": "upperleg", "upperleg": "upperleg", "lowerleg": "lowerleg",
    "calf": "lowerleg", "shin": "lowerleg", "leg": "leg",
    "foot": "foot", "ankle": "foot", "toe": "toes", "toes": "toes", "toebase": "toes", "ball": "toes",
    "thumb": "thumb", "index": "index", "middle": "middle", "mid": "middle",
    "ring": "ring", "pinky": "little", "pinkie": "little", "pink": "little", "little": "little",
}
FINGER_PARTS = frozenset({"thumb", "index", "middle", "ring", "little"})
TWIST_TOKENS = frozenset({"twist", "roll"})
HELPER_TOKENS = frozenset({
    "helper", "corrective", "correct", "fix", "share", "pole", "ik", "ctrl", "control", "mch", "target",
    "tweak", "jiggle", "cloth", "skirt", "cape", "breast", "boob", "bust", "pectoral",
})
END_TOKENS = frozenset({"end", "nub", "tip", "top", "site"})

# --- shape keys (consumed by core/expressions.py) ---------------------------
# normalised name (lowercase alphanumerics only) -> confidence
EXPRESSION_ALIASES: dict[str, dict[str, float]] = {
    "blink": {"blink": 0.9, "blinkboth": 0.9, "eyeblink": 0.9, "eyesclosed": 0.9, "eyeclose": 0.9,
              "eyeclosed": 0.9, "closeeyes": 0.85},
    "blinkLeft": {"blinkleft": 0.9, "blinkl": 0.9, "eyeblinkleft": 0.9, "eyeblinkl": 0.9, "eyecloseleft": 0.9,
                  "eyeclosel": 0.9, "eyeclosedleft": 0.9, "lefteyeclose": 0.9, "leftblink": 0.9},
    "blinkRight": {"blinkright": 0.9, "blinkr": 0.9, "eyeblinkright": 0.9, "eyeblinkr": 0.9,
                   "eyecloseright": 0.9, "eyecloser": 0.9, "eyeclosedright": 0.9, "righteyeclose": 0.9,
                   "rightblink": 0.9},
    # A/I/U/E/O, Oculus / ReadyPlayerMe, VRChat (``vrc.v_aa`` after the namespace), Daz Genesis (``eCTRLvAA``),
    # Mixamo Fuse and MMD (あいうえお) viseme names
    "aa": {"aa": 0.9, "a": 0.9, "visemeaa": 0.9, "vaa": 0.9, "moutha": 0.9, "ectrlvaa": 0.9, "mouthopen": 0.8,
           "あ": 0.9},
    "ih": {"ih": 0.9, "i": 0.9, "visemeih": 0.9, "visemei": 0.9, "vih": 0.9, "mouthi": 0.9, "ectrlvih": 0.9, "い": 0.9},
    "ou": {"ou": 0.9, "u": 0.9, "visemeou": 0.9, "visemeu": 0.9, "vou": 0.9, "mouthu": 0.9, "ectrlvuw": 0.9, "う": 0.9},
    "ee": {"ee": 0.9, "e": 0.9, "visemee": 0.9, "visemeee": 0.9, "vee": 0.9, "ve": 0.9, "mouthe": 0.9, "ectrlvee": 0.9,
           "え": 0.9},
    "oh": {"oh": 0.9, "o": 0.9, "visemeoh": 0.9, "visemeo": 0.9, "voh": 0.9, "moutho": 0.9, "ectrlvow": 0.9, "お": 0.9},
    "happy": {"happy": 0.9, "joy": 0.9, "smile": 0.8, "smiling": 0.8, "にっこり": 0.75},
    "angry": {"angry": 0.9, "anger": 0.9, "怒り": 0.8},
    "sad": {"sad": 0.9, "sorrow": 0.9, "困る": 0.75},
    "relaxed": {"relaxed": 0.9, "calm": 0.8, "なごみ": 0.7},
    "surprised": {"surprised": 0.9, "surprise": 0.9, "びっくり": 0.8},
}
# MMD and Daz blink names (merged into the tables above so each expression has one alias table).
EXPRESSION_ALIASES["blink"].update({"まばたき": 0.9, "ectrleyesclosed": 0.9})
EXPRESSION_ALIASES["blinkLeft"].update({"ectrleyesclosedl": 0.9, "ウィンク": 0.75, "eyeclosedleft": 0.9})
EXPRESSION_ALIASES["blinkRight"].update({"ectrleyesclosedr": 0.9, "ウィンク右": 0.8, "eyeclosedright": 0.9})
