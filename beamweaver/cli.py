"""Interactive menu and command-line workflows with operation-specific inputs."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

from .constants import (
    BW_VERSION, BW4_EMIN_MEV, BW4_EMAX_MEV, BW4_M_DEFAULT,
    DEFAULT_TRANSPORT_ECUT_MEV, DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_INFERENCE_BATCH_SIZE, DEFAULT_FACTOR_VALIDATION_SAMPLES,
    DEFAULT_REFERENCE_VALIDATION_SAMPLES, DEFAULT_RUN_ENERGY_MEV,
    DEFAULT_RUN_HISTORIES, DEFAULT_COMPARISON_ENERGIES,
    DEFAULT_COMPARISON_HISTORIES, PHOTON_CROSS_SECTION_TABLE,
    RAYLEIGH_CROSS_SECTION_TABLE, PHOTO_SHELL_TABLE,
    COHERENT_FORM_FACTOR_TABLE, INCOHERENT_SCATTERING_TABLE,
    ELECTRON_STOPPING_POWER_TABLE,
)


###############################################################################
#                           HEADER
###############################################################################
HEADER = r"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ____                        __        __                        ║
║  | __ )  ___  __ _ _ __ ___   \ \      / /__  __ ___   _____ _ __ ║
║  |  _ \ / _ \/ _` | '_ ` _ \   \ \ /\ / / _ \/ _` \ \ / / _ \ '__|║
║  | |_) |  __/ (_| | | | | | |   \ V  V /  __/ (_| |\ V /  __/ |   ║
║  |____/ \___|\__,_|_| |_| |_|    \_/\_/ \___|\__,_| \_/ \___|_|   ║
║                                                                   ║
║                        v {version:<41}                            ║
║                                                                   ║
║  Learned-kernel Monte Carlo photon transport                      ║
║  Supervised stochastic transport kernel                           ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Copyright (c) 2025-2026 Pedro Teles                              ║
║  Department of Physics and Astronomy                              ║
║  Faculty of Sciences, University of Porto, Portugal               ║
║                                                                   ║
║  Licensed under the Apache License, Version 2.0                   ║
║  You may obtain a copy of the License at:                         ║
║  http://www.apache.org/licenses/LICENSE-2.0                       ║
║                                                                   ║
║  Unless required by applicable law or agreed to in writing,       ║
║  software distributed under the License is distributed on an      ║
║  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.     ║
║                                                                   ║
║  GitHub: https://github.com/PedroTelesFCUP/Beam_weaver            ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""".format(version=" . ".join(BW_VERSION.split(".")))


_MATERIAL_TABLES = (PHOTON_CROSS_SECTION_TABLE, RAYLEIGH_CROSS_SECTION_TABLE,
                    PHOTO_SHELL_TABLE, COHERENT_FORM_FACTOR_TABLE)


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _energy(value):
    number = float(value)
    if not math.isfinite(number) or not BW4_EMIN_MEV <= number <= BW4_EMAX_MEV:
        raise argparse.ArgumentTypeError(
            f"energy must be in [{BW4_EMIN_MEV}, {BW4_EMAX_MEV}] MeV")
    return number


def _energies(value):
    try:
        return [_energy(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated energies in MeV") from exc


def _parser():
    parser = argparse.ArgumentParser(
        prog="beamweaver",
        description="Beam Weaver: generate data, train, run and validate transport.",
        epilog="Without a command, opens the six-option interactive menu.")
    commands = parser.add_subparsers(dest="command")

    def output_options(subparser):
        subparser.add_argument("--output", metavar="RUN_DIR",
                               help="new output directory (default: runs/<command>-<UTC time>)")

    def data_options(subparser):
        subparser.add_argument("--data-dir", default=".", metavar="DIRECTORY",
                               help="directory containing the supplied water CSV tables")

    def device_options(subparser):
        subparser.add_argument("--device", default="auto",
                               help="torch device, e.g. cpu, cuda:0 (default: auto)")

    generate = commands.add_parser("generate", help="generate schema-v4 MC data")
    data_options(generate)
    output_options(generate)
    generate.add_argument("--smoke", action="store_true",
                          help="reduced grid and event counts; retains the data sanity check")
    generate.add_argument("--events", type=_positive_int, metavar="N",
                          help="events per photon energy for each sampling group; photoelectric events are counted separately for each shell; default preserves group-specific counts")

    generate.add_argument("--categorical-events", type=_positive_int, metavar="N",
                          help=f"events per photon energy for each of process selection and shell selection (default: {BW4_M_DEFAULT['process']})")
    generate.add_argument("--continuous-events", type=_positive_int, metavar="N",
                          help=f"events per photon energy for each of Rayleigh and Compton; photoelectric events per photon energy and shell (default: {BW4_M_DEFAULT['ray']})")
    generate.add_argument("--pair-events", type=_positive_int, metavar="N",
                          help=f"pair events per photon energy above threshold (default: {BW4_M_DEFAULT['pair']}); group counts override --events")

    train = commands.add_parser("train", help="train all heads or one selected head")
    train.add_argument("dataset", help="schema-v4 .npz dataset")
    device_options(train)
    group = train.add_mutually_exclusive_group()
    group.add_argument("--output", metavar="RUN_DIR", help="new training output directory")
    group.add_argument("--resume", metavar="RUN_DIR",
                       help="explicitly resume head checkpoints from this run directory")
    train.add_argument("--factor", help="train/retrain one selected head (e.g. process, comp_u, pair_frac)")
    train.add_argument("--epochs", type=_positive_int,
                       help="override each head's existing epoch limit")
    train.add_argument("--batch", type=_positive_int, default=DEFAULT_TRAIN_BATCH_SIZE,
                       help=f"training batch size (default: {DEFAULT_TRAIN_BATCH_SIZE})")

    run = commands.add_parser("run", help="run an audited single-energy shower")
    run.add_argument("checkpoint", help=f"complete v{BW_VERSION} policy checkpoint")
    data_options(run)
    output_options(run)
    device_options(run)
    run.add_argument("--energy", type=_energy, default=DEFAULT_RUN_ENERGY_MEV, metavar="MEV")
    run.add_argument("--histories", type=_positive_int, default=DEFAULT_RUN_HISTORIES)

    compare = commands.add_parser("compare", help="compare MC1, MC2 and BeamWeaver")
    compare.add_argument("checkpoint", help=f"complete v{BW_VERSION} policy checkpoint")
    data_options(compare)
    output_options(compare)
    device_options(compare)
    compare.add_argument("--energies", type=_energies,
                         default=list(DEFAULT_COMPARISON_ENERGIES), metavar="MEV,...")
    compare.add_argument("--histories", type=_positive_int, default=DEFAULT_COMPARISON_HISTORIES,
                         help=f"histories per arm and energy (default: {DEFAULT_COMPARISON_HISTORIES})")
    compare.add_argument("--batch", type=_positive_int, default=DEFAULT_INFERENCE_BATCH_SIZE,
                         help=f"BeamWeaver inference batch size (default: {DEFAULT_INFERENCE_BATCH_SIZE})")

    validate = commands.add_parser("validate", help="validate learned factors or reference samplers")
    validate.add_argument("checkpoint", nargs="?", help="policy checkpoint (required unless --reference)")
    validate.add_argument("--reference", action="store_true",
                          help="validate reference samplers against analytic angular targets; no policy needed")
    data_options(validate)
    output_options(validate)
    device_options(validate)
    validate.add_argument("--energies", type=_energies, metavar="MEV,...")
    validate.add_argument("--samples", type=_positive_int,
                          help=f"MC samples per photon energy (photoelectric: per photon energy and shell); "
                               f"factor validation uses two independent MC replicas with this count each "
                               f"(default: {DEFAULT_FACTOR_VALIDATION_SAMPLES} "
                               f"for factors, {DEFAULT_REFERENCE_VALIDATION_SAMPLES} for --reference)")
    validate.add_argument("--no-figures", action="store_true")

    report = commands.add_parser("report", help="rebuild figures from a saved comparison")
    report.add_argument("run_dir", help="directory containing comparison.json and dose arrays")
    report.add_argument("--output", metavar="DIRECTORY",
                        help="figure directory (default: <run_dir>/figures)")
    return parser


def _required_file(value, description):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _data_directory(value, command):
    data_dir = Path(value).expanduser().resolve()
    tables = list(_MATERIAL_TABLES)
    if command in ("generate", "compare", "validate"):
        tables.append(INCOHERENT_SCATTERING_TABLE)
    if command in ("run", "compare"):
        tables.append(ELECTRON_STOPPING_POWER_TABLE)
    missing = [name for name in tables if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing supplied physics tables in {data_dir}: {', '.join(missing)}. "
            "Use --data-dir to select the directory containing those files.")
    return data_dir


@contextmanager
def _physics_directory(data_dir):
    """Contain existing relative CSV readers inside a CLI-only data context.

    Every user path is resolved before entry. Training and reporting never
    enter this context; the original working directory is restored on error.
    """
    original = Path.cwd()
    try:
        os.chdir(data_dir)
        yield
    finally:
        os.chdir(original)


def _load_data():
    from .materials import WaterPhotonData
    return WaterPhotonData(PHOTON_CROSS_SECTION_TABLE,
                                RAYLEIGH_CROSS_SECTION_TABLE)


def _device(value):
    if value != "auto":
        return value
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _new_run_dir(value, command):
    if value is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        value = Path("runs") / f"{command}-{stamp}"
    path = Path(value).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {path}; choose a new --output")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_command(run_dir, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = run_dir / f"command_{stamp}.json"
    path.write_text(json.dumps({"working_directory": str(Path.cwd()),
                                "arguments": vars(args)}, indent=2, default=str))


def _execute(args):
    """Run one operation and return its output directory."""
    command = args.command
    if command == "report":
        run_dir = Path(args.run_dir).expanduser().resolve()
        _required_file(run_dir / "comparison.json", "Comparison report")
        output = Path(args.output).expanduser().resolve() if args.output else None
        from .reporting import regenerate_reports
        regenerate_reports(run_dir, save_dir=output)
        return output or run_dir / "figures"

    if command == "train":
        dataset = _required_file(args.dataset, "Dataset")
        from .constants import BW4_FACTOR_HEADS
        if args.factor and args.factor not in BW4_FACTOR_HEADS:
            raise ValueError(f"Unknown head {args.factor!r}; choose from: "
                             f"{', '.join(BW4_FACTOR_HEADS)}")
        if args.resume:
            run_dir = Path(args.resume).expanduser().resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Training run directory not found: {run_dir}")
            if not any(run_dir.glob("v040_head_*.pt")):
                raise FileNotFoundError(f"No head checkpoints found in {run_dir}")
            if args.factor:
                checkpoint = _required_file(run_dir / "v040_policy.pt", "Policy for head retraining")
        else:
            run_dir = _new_run_dir(args.output, command)
        from .training import create_policy, train_policy, train_factor
        device = _device(args.device)
        if args.factor and args.resume:
            from .policy import GenerativeTransportPolicy
            policy = GenerativeTransportPolicy.load_policy(str(checkpoint), device=device)
        else:
            policy = create_policy(str(dataset))
        kwargs = {"device": device, "batch": args.batch}
        if args.epochs is not None:
            kwargs["epochs"] = args.epochs
        _record_command(run_dir, args)
        if args.factor:
            train_factor(policy, args.factor, str(dataset),
                             run_dir=run_dir, **kwargs)
            policy.save_policy(run_dir / "v040_policy.pt")
            print(f"  [train] policy saved -> {run_dir / 'v040_policy.pt'}")
        else:
            train_policy(policy, str(dataset), run_dir,
                               resume=bool(args.resume), **kwargs)
        return run_dir

    # File preflight is specific to the selected operation. No tables are
    # required to open the menu, show help, train heads, or regenerate plots.
    reference = command == "validate" and args.reference
    if command == "validate":
        if reference and args.checkpoint:
            raise ValueError("--reference does not use a policy checkpoint; omit the checkpoint argument")
        if reference and args.energies is not None:
            raise ValueError("--reference uses the fixed sampler-validation energy suite; omit --energies")
        if not reference and not args.checkpoint:
            raise ValueError("validate requires a policy checkpoint or --reference")
    checkpoint = (_required_file(args.checkpoint, "Policy checkpoint")
                  if command in ("run", "compare", "validate") and not reference else None)
    data_dir = _data_directory(args.data_dir, command)
    run_dir = _new_run_dir(args.output, command)
    _record_command(run_dir, args)
    with _physics_directory(data_dir):
        data = _load_data()
        if command == "generate":
            from .dataset import generate_dataset, write_generator_spec
            from .coordinates import transform_selftest
            transform_selftest()
            counts = dict(BW4_M_DEFAULT)
            kwargs = {}
            if args.smoke:
                kwargs["grid_stride"] = 8
                counts.update(process=2048, shell=2048, ray=512,
                              comp=512, photo=512, pair=512)
                print("  [generate] reduced grid (stride 8); "
                      "the representation sanity check remains active.")
            if args.events is not None:
                counts = {factor: args.events for factor in counts}
            for value, factors in (
                    (args.categorical_events, ("process", "shell")),
                    (args.continuous_events, ("ray", "comp", "photo")),
                    (args.pair_events, ("pair",))):
                if value is not None:
                    counts.update({factor: value for factor in factors})
            kwargs["M"] = counts
            print(f"  [generate] events per photon energy (photo: per photon energy and shell): {counts}")
            generate_dataset(data, out=str(run_dir / "schema_v4_data.npz"), **kwargs)
            write_generator_spec(str(run_dir / "schema_v4_generator_spec.json"))
            return run_dir

        if reference:
            from .validation import validate_reference_samplers
            report = validate_reference_samplers(
                data, n_samples=args.samples or DEFAULT_REFERENCE_VALIDATION_SAMPLES)
            (run_dir / "reference_validation.json").write_text(json.dumps(report, indent=2))
            return run_dir

        from .policy import GenerativeTransportPolicy
        device = _device(args.device)
        policy = GenerativeTransportPolicy.load_policy(str(checkpoint), device=device)
        if command == "validate":
            from .validation import validate_policy
            validate_policy(policy, data, energies=args.energies,
                                 M=args.samples or DEFAULT_FACTOR_VALIDATION_SAMPLES,
                                 out_json="validation.json",
                                 output_dir=str(run_dir), figures=not args.no_figures)
            return run_dir

        from .geometry import WaterPhantom
        env = WaterPhantom(data, fixed_energy=getattr(args, "energy", DEFAULT_RUN_ENERGY_MEV),
                           ecut=DEFAULT_TRANSPORT_ECUT_MEV,
                           energy_range=(BW4_EMIN_MEV, BW4_EMAX_MEV))
        if command == "run":
            from .evaluation import evaluate_transport
            evaluate_transport(policy, data, env, args.energy,
                                   args.histories, device=device, save_dir=run_dir)
        elif command == "compare":
            from .evaluation import compare_transport
            compare_transport(policy, data, env, args.energies,
                                     args.histories, device=device,
                                     batch=args.batch, save_dir=run_dir)
    return run_dir


def _ask(label, default=None):
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"  {label}{suffix}: ").strip()
    return answer or (str(default) if default is not None else "")


def interactive_menu():
    """Six operations; every transport action selects its checkpoint directly."""
    parser = _parser()
    dataset = "schema_v4_data.npz"
    checkpoint = "v040_policy.pt"
    data_dir = "."
    while True:
        print(f"\nBeam Weaver {BW_VERSION}\n"
              "  1  Generate training/validation/test data\n"
              "  2  Train all heads or one selected head\n"
              "  3  Run an audited shower\n"
              "  4  Compare MC1, MC2 and BeamWeaver\n"
              "  5  Validate factors or reference samplers\n"
              "  6  Exit")
        try:
            choice = _ask("Select option", "6")
            if choice == "6":
                return 0
            if choice not in {"1", "2", "3", "4", "5"}:
                print("  Choose an option from 1 to 6.")
                continue
            command = {"1": "generate", "2": "train", "3": "run",
                       "4": "compare", "5": "validate"}[choice]
            argv = [command]
            if command == "train":
                dataset = _ask("Dataset", dataset)
                argv.append(dataset)
                factor = _ask("Head to train (blank = all heads)")
                if factor:
                    argv.extend(["--factor", factor])
                resume = _ask("Resume directory (blank = new training run)")
                if resume:
                    argv.extend(["--resume", resume])
            else:
                reference = (command == "validate" and
                             _ask("Validate reference samplers? y/N", "N").lower() == "y")
                if reference:
                    argv.append("--reference")
                elif command != "generate":
                    checkpoint = _ask("Policy checkpoint", checkpoint)
                    argv.append(checkpoint)
                data_dir = _ask("Water CSV directory", data_dir)
                argv.extend(["--data-dir", data_dir])
            if command == "generate":
                if _ask("Reduced smoke dataset? y/N", "N").lower() == "y":
                    argv.append("--smoke")
                else:
                    argv.extend(["--categorical-events", _ask("Events per photon energy for each of process selection and shell selection", BW4_M_DEFAULT["process"]),
                                 "--continuous-events", _ask("Events per photon energy for each of Rayleigh/Compton; photoelectric per energy and shell", BW4_M_DEFAULT["ray"]),
                                 "--pair-events", _ask("Pair events per photon energy above threshold", BW4_M_DEFAULT["pair"])])
            elif command == "run":
                argv.extend(["--energy", _ask("Photon energy (MeV)", DEFAULT_RUN_ENERGY_MEV),
                             "--histories", _ask("Histories", DEFAULT_RUN_HISTORIES)])
            elif command == "compare":
                argv.extend(["--energies", _ask("Energies (MeV, comma separated)",
                                              ",".join(map(str, DEFAULT_COMPARISON_ENERGIES))),
                             "--histories", _ask("Histories per arm", DEFAULT_COMPARISON_HISTORIES)])
            elif command == "validate":
                argv.extend(["--samples", _ask("MC samples per photon energy (photoelectric: per energy and shell)",
                                              DEFAULT_REFERENCE_VALIDATION_SAMPLES
                                              if "--reference" in argv
                                              else DEFAULT_FACTOR_VALIDATION_SAMPLES)])
            if "--resume" not in argv:
                output = _ask("Output run directory (blank = automatic)")
                if output:
                    argv.extend(["--output", output])
            args = parser.parse_args(argv)
            run_dir = _execute(args)
            if command == "generate":
                dataset = str(run_dir / "schema_v4_data.npz")
            elif command == "train":
                checkpoint = str(run_dir / "v040_policy.pt")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            return 0
        except SystemExit:
            # argparse already explains invalid input; keep the menu usable.
            continue
        except Exception as exc:
            print(f"  {type(exc).__name__}: {exc}")


def main(argv=None):
    print(HEADER, flush=True)
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return interactive_menu()
    try:
        _execute(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"beamweaver: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0
