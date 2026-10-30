"""Avatar Auto-Author: turns a ``HumanModelAnalysis`` into inspectable *plans*.

ANALYZE -> PLAN -> VALIDATE PLAN -> MUTATE (``blender/*_author.py``) -> VALIDATE RESULT.
Nothing in this package touches a scene; every plan is plain data with a ``validate()`` and a skip reason.
"""
