"""Human anatomy analysis: pure, ``bpy``-free semantic facts about a humanoid character.

Everything here consumes a ``CharacterInspection`` and a ``HumanoidResult`` and produces
``HumanModelAnalysis``. Authoring (``core/authoring``) consumes the analysis; nothing here mutates anything.
"""
