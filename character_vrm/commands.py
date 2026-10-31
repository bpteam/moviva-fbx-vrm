"""Command implementations shared by the outer CLI (argument parsing only) and the Blender entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable, Sequence, TextIO

from .blender.bridge import BlenderBridge
from .core.config import ConfigError, ConversionConfig, GEOMETRY_PROFILES, load_config
from .core.errors import ConversionError
from .core.events import encode as encode_event
from .core.preflight import run_preflight
from .core.steps import plan as plan_steps
from .core.pipeline import ConverterCore, source_format
from .core.anatomy.report import format_analysis
from .core.authoring.report import authoring_to_dict, format_authoring
from .core.report import anatomy_section, format_inspection, suggest_map
from .toolchain import evaluate_environment

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def build_parser(prog: str = "character-vrm", with_blender: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="Convert FBX / GLB / glTF characters to VRM 1.0 with headless Blender.")
    if with_blender:
        parser.add_argument("--blender", metavar="PATH",
                            help="Blender executable (default: $BLENDER_BIN, then blender on PATH)")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        if with_blender:  # also accepted after the subcommand; SUPPRESS keeps the value given before it
            p.add_argument("--blender", metavar="PATH", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--config", metavar="FILE", help="*.vrm-map.json; explicit entries beat every heuristic")
        p.add_argument("--armature", metavar="NAME", help="primary armature object (overrides the map file)")
        p.add_argument("--min-confidence", type=float, metavar="0..1",
                       help="humanoid confidence gate (default 0.75)")
        p.add_argument("--geometry-profile", choices=sorted(GEOMETRY_PROFILES),
                       help="geometry profile (default: full)")
        p.add_argument("--auto-rig", action="store_true",
                       help="source without a skeleton: generate a humanoid skeleton and skin weights from the "
                            "geometry (validated by pose tests; never replaces an existing armature)")
        p.add_argument("--auto-author", action="store_true",
                       help="rest-pose calibration, hair -> spring bones, anatomical colliders and generated "
                            "blink / visemes when confidence is high (conservative)")
        p.add_argument("--generate-face", action="store_true",
                       help="also generate basic expressions (smile, sad, surprise, brows, squint); implies --auto-author")
        p.add_argument("--generate-finger-rig", action="store_true",
                       help="experimental: bones for fingers that exist only as geometry; implies --auto-author")
        p.add_argument("--generate-hair-rig", action="store_true",
                       help="experimental: bones for a hair mesh that has none; implies --auto-author")

    doctor = sub.add_parser("doctor", help="verify Blender, the VRM Add-on and the pinned toolchain")
    if with_blender:
        doctor.add_argument("--blender", metavar="PATH", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p = sub.add_parser("inspect", help="import and describe a character")
    p.add_argument("source")
    common(p)
    p = sub.add_parser("preflight", help="inspect a model (with or without a skeleton) and print the processing "
                                         "plan as JSON; converts nothing")
    p.add_argument("source")
    common(p)
    p = sub.add_parser("analyze", help="describe the anatomy of a character (frame, skeleton, hands, face, hair, "
                                       "materials) without modifying it")
    p.add_argument("source")
    p.add_argument("--json", metavar="FILE", help="write the analysis as JSON (the same section a conversion report holds)")
    common(p)
    p = sub.add_parser("map", help="write a suggested <model>.vrm-map.json without converting")
    p.add_argument("source")
    p.add_argument("--output", metavar="FILE", help="default: <source without extension>.vrm-map.json")
    common(p)
    p = sub.add_parser("convert", help="convert to VRM 1.0 (anatomy analysis always runs; authoring needs --auto-author)")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--dry-run", action="store_true", help="analyse and report; write no VRM")
    p.add_argument("--events", action="store_true", help="print structured progress as '@event {json}' lines")
    p.add_argument("--no-springs", action="store_true", help="never generate spring bones")
    common(p)
    return parser


def _load_config(args: argparse.Namespace) -> ConversionConfig:
    config = load_config(args.config) if getattr(args, "config", None) else ConversionConfig()
    if getattr(args, "armature", None):
        config.armature = args.armature
    if getattr(args, "min_confidence", None) is not None:
        if not 0.0 <= args.min_confidence <= 1.0:
            raise ConfigError("--min-confidence must be within 0..1")
        config.min_confidence = args.min_confidence
    if getattr(args, "no_springs", False):
        config.spring.enabled = False
    if getattr(args, "geometry_profile", None):
        config.geometry.profile = args.geometry_profile
    if getattr(args, "auto_rig", False):
        config.auto_rig = True
    author = config.auto_author
    if getattr(args, "auto_author", False):
        author.enabled = True
    for flag, attr in (("generate_face", "generate_face"), ("generate_finger_rig", "generate_finger_rig"),
                       ("generate_hair_rig", "generate_hair_rig")):
        if getattr(args, flag, False):
            setattr(author, attr, True)
    return config


def format_warning(code: str, message: str) -> str:
    if code == "fbx-no-shape-keys":
        return "WARNING\n\n" + "\n".join(f"  {line}" for line in (
            "Source FBX imported successfully, but no facial shape keys",
            "were available after Blender import.", "",
            "Humanoid conversion can continue.",
            "Blink / visemes / facial expressions cannot be generated",
            "from nonexistent morph targets.", "",
            "Prefer a GLB/glTF source containing morph targets when available."))
    return f"WARNING [{code}]: {message}"


def run(
    argv: Sequence[str],
    bridge_factory: Callable[[], BlenderBridge],
    out: TextIO | None = None,
) -> int:
    import sys
    out = out or sys.stdout
    emit = lambda text="": print(text, file=out)  # noqa: E731
    args = build_parser(with_blender=False).parse_args(list(argv))
    bridge = bridge_factory()

    try:
        if args.command == "doctor":
            report = evaluate_environment(bridge.environment())
            emit(report.format())
            return EXIT_OK if report.ok else EXIT_FAILED

        config = _load_config(args)
        events = (lambda e: emit(encode_event(e))) if getattr(args, "events", False) else None
        core = ConverterCore(bridge, events=events)
        source = Path(args.source).resolve()

        if args.command == "preflight":
            info, facts = run_preflight(core.load(source), source, config)
            document = {"model": info, "facts": asdict(facts), "steps": [s.to_dict() for s in plan_steps(facts)]}
            emit("@preflight " + json.dumps(document))  # one line: the window finds it among Blender's log output
            return EXIT_OK

        if args.command == "inspect":
            analysis = core.inspect(source, config)
            emit(format_inspection(analysis))
            return EXIT_OK if analysis.convertible[0] else EXIT_FAILED

        if args.command == "analyze":
            analysis = core.inspect(source, config)
            emit(format_analysis(analysis.anatomy))
            if analysis.authoring is not None:
                emit()
                emit(format_authoring(analysis.authoring))
            if args.json:
                target = Path(args.json)
                target.parent.mkdir(parents=True, exist_ok=True)
                document = anatomy_section(analysis)
                if analysis.authoring is not None:
                    document = {**document, "authoring": authoring_to_dict(analysis.authoring, None)}
                target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
                emit()
                emit(f"Analysis written: {target}")
            return EXIT_OK if analysis.anatomy is not None else EXIT_FAILED

        if args.command == "map":
            analysis = core.inspect(source, config)
            target = Path(args.output) if args.output else source.with_suffix("").with_name(source.stem + ".vrm-map.json")
            target.write_text(json.dumps(suggest_map(analysis, config.min_confidence), indent=2) + "\n", encoding="utf-8")
            emit(format_inspection(analysis))
            emit()
            emit(f"Suggested map written: {target}")
            return EXIT_OK

        # convert
        output = Path(args.output).resolve()
        report_path = output.with_name(output.name + ".report.json")
        try:
            outcome = core.convert(source, output, config, dry_run=args.dry_run)
        except ConversionError as exc:
            if getattr(exc, "report", None) is not None and not args.dry_run:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(exc.report, indent=2) + "\n", encoding="utf-8")
                emit(f"Report: {report_path}")
            raise
        for warning in outcome.report["warnings"]:
            emit(format_warning(warning["code"], warning["message"]))
        if args.dry_run:
            emit(format_inspection(outcome.analysis))
            emit()
            emit("Dry run: nothing was exported.")
            return EXIT_OK
        report_path.write_text(json.dumps(outcome.report, indent=2) + "\n", encoding="utf-8")
        emit(f"Converted: {output}")
        emit(f"Report:    {report_path}")
        return EXIT_OK
    except (ConversionError, ConfigError) as exc:
        emit(f"ERROR {exc}")
        return EXIT_FAILED
