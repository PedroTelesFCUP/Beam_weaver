"""Interactive menu and command-line workflows with operation-specific inputs."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import count
import json
import math
import os
from pathlib import Path
import sys
import textwrap

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
║                            v {version:<10}                           ║
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
                               help="new output directory (default: runs/<command>-YYYYMMDD-vN, UTC date)")

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
                          help="MC draws per reference sampling set (one outcome at one incident energy; photoelectric also fixes a shell); default preserves quantity-specific counts")

    generate.add_argument("--categorical-events", type=_positive_int, metavar="N",
                          help=f"MC draws at each photon energy for interaction choice and shell choice separately (default: {BW4_M_DEFAULT['process']})")
    generate.add_argument("--continuous-events", type=_positive_int, metavar="N",
                          help=f"MC draws at each energy for Rayleigh and Compton separately, and at each energy and shell for photoelectric (default: {BW4_M_DEFAULT['ray']})")
    generate.add_argument("--pair-events", type=_positive_int, metavar="N",
                          help=f"pair MC draws per photon energy above threshold (default: {BW4_M_DEFAULT['pair']}); quantity-specific counts override --events")

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
                         help=f"primary photon histories per selected energy per method (MC1, MC2, BeamWeaver; default: {DEFAULT_COMPARISON_HISTORIES})")
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
        root = Path("runs").expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        for version in count(1):
            path = root / f"{command}-{day}-v{version}"
            try:
                path.mkdir()  # Atomic reservation even if two runs start together.
                return path
            except FileExistsError:
                continue
    path = Path(value).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {path}; choose a new --output")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_command(run_dir, args):
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    for version in count(1):
        path = run_dir / f"command-{day}-v{version}.json"
        try:
            with path.open("x") as fh:
                json.dump({"working_directory": str(Path.cwd()),
                           "arguments": vars(args)}, fh, indent=2, default=str)
            return
        except FileExistsError:
            continue


def _explain_operation(command):
    """Explain the selected operation before the interactive questions/CLI run."""
    explanations = {
        "generate": (
            "Beam Spinner will sample individual interaction outcomes at fixed photon "
            "energies before each collision. The training, validation and test energies "
            "and sample counts will be shown before sampling starts. The training "
            "samples also set output-bin boundaries; validation samples check that "
            "representation. No neural head is trained by this operation."),
        "train": (
            "Train 13 separate stochastic heads in 10 stages from the generated "
            "reference data. By default, nine heads learn sampled outcome "
            "distributions for up to 400 epochs each; four pair-direction heads "
            "learn from individual events for up to 20 epochs each. Each epoch "
            "is evaluated on validation cross-entropy. Training may stop early, "
            "and the weights with the lowest validation cross-entropy are saved "
            "for each head. This selection helps limit overfitting; it does not "
            "establish physical accuracy."),
        "run": (
            "Transport primary photons at one selected source energy using the "
            "learned interaction heads. The audit checks execution provenance "
            "and energy accounting; free paths and charged-particle transport "
            "still use the existing physics routines. Results include an "
            "execution record, summary and deposited-energy tally."),
        "compare": (
            "Run two independently seeded Beam Spinner references (MC1 and MC2) "
            "and Beam Weaver separately at each selected source energy. The "
            "requested history count applies to EACH method at EACH energy. "
            "MC1 versus MC2 illustrates sampling variability; a separate "
            "report command makes figures from the saved arrays."),
        "validate": (
            "Choose learned-head diagnostics or reference-sampler checks. "
            "Learned-head diagnostics compare all 13 heads with fresh Beam Spinner "
            "samples and save discrepancy metrics and optional figures; they do "
            "not issue a global physics pass. Reference mode checks the built-in "
            "angular samplers against their specified analytical targets and "
            "requires no learned policy."),
        "report": (
            "Regenerate figures from the saved comparison summary and dose "
            "arrays. The simulation data are read from the selected run; "
            "this command does not repeat any photon histories."),
    }
    print("\n" + textwrap.fill(
        explanations[command], width=79,
        initial_indent=f"  [{command}] ", subsequent_indent="    ")
        + "\n", flush=True)


def _execute(args):
    """Run one operation and return its output directory."""
    command = args.command
    if command == "report":
        run_dir = Path(args.run_dir).expanduser().resolve()
        _required_file(run_dir / "comparison.json", "Comparison report")
        output = Path(args.output).expanduser().resolve() if args.output else None
        from .reporting import regenerate_reports
        print(f"  [report] rebuilding figures from {run_dir}; "
              f"saving to {output or run_dir / 'figures'}.", flush=True)
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
        limits = (f"up to {args.epochs} epochs for each selected head"
                  if args.epochs is not None else
                  "up to 400 epochs for nine grouped-target heads; "
                  "up to 20 for four pair-direction heads")
        print(f"  [train] dataset: {dataset}\n"
              f"  [train] epoch limits: {limits}\n"
              "  [train] each epoch is checked on validation samples; "
              "the lowest validation cross-entropy checkpoint is retained.",
              flush=True)
        if args.factor:
            if not args.resume:
                print("  [train] a new single-head run saves an incomplete policy; "
                      "train all heads before running a shower.", flush=True)
            else:
                print("  [train] retraining this head in the resumed policy; "
                      "use the same dataset and binning as the original run.",
                      flush=True)
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
    print(f"  [{command}] results directory: {run_dir}", flush=True)
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
            print("  [generate] MC draws per reference sampling set "
                  "(one quantity at one photon energy; photoelectric also fixes a shell):\n"
                  f"             process={counts['process']:,}, shell={counts['shell']:,}, "
                  f"Rayleigh={counts['ray']:,}, Compton={counts['comp']:,},\n"
                  f"             photoelectric={counts['photo']:,}, "
                  f"pair={counts['pair']:,}", flush=True)
            generate_dataset(data, out=str(run_dir / "schema_v4_data.npz"), **kwargs)
            write_generator_spec(str(run_dir / "schema_v4_generator_spec.json"))
            return run_dir

        if reference:
            from .validation import validate_reference_samplers
            print("  [validate] reference-sampler mode: checking sampled angles "
                  "against the built-in analytical targets; no neural policy is used.\n"
                  f"  [validate] {args.samples or DEFAULT_REFERENCE_VALIDATION_SAMPLES:,} "
                  "draws per tested energy and channel.", flush=True)
            report = validate_reference_samplers(
                data, n_samples=args.samples or DEFAULT_REFERENCE_VALIDATION_SAMPLES)
            (run_dir / "reference_validation.json").write_text(json.dumps(report, indent=2))
            return run_dir

        from .policy import GenerativeTransportPolicy
        device = _device(args.device)
        policy = GenerativeTransportPolicy.load_policy(str(checkpoint), device=device)
        if command == "validate":
            from .validation import validate_policy
            selected = args.energies or [0.02, 0.05, 0.1, 0.5, 1.05, 2.0, 5.0, 10.0]
            print("  [validate] learned-head mode: 13 heads compared with two "
                  "fresh MC replicas; discrepancy metrics are diagnostic, "
                  "without a global pass/fail threshold.\n"
                  f"  [validate] photon energies (MeV): {', '.join(f'{e:g}' for e in selected)}; "
                  f"{args.samples or DEFAULT_FACTOR_VALIDATION_SAMPLES:,} "
                  "samples per energy and shell when applicable, for EACH MC replica.",
                  flush=True)
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
            print(f"  [run] {args.histories:,} primary photon histories at "
                  f"{args.energy:g} MeV.", flush=True)
            evaluate_transport(policy, data, env, args.energy,
                                   args.histories, device=device, save_dir=run_dir)
        elif command == "compare":
            from .evaluation import compare_transport
            print(f"  [compare] source energies (MeV): "
                  f"{', '.join(f'{e:g}' for e in args.energies)}\n"
                  f"  [compare] {args.histories:,} primary photon histories "
                  "at EACH energy for EACH method (MC1, MC2, BeamWeaver).",
                  flush=True)
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
            _explain_operation(command)
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
                    argv.extend(["--categorical-events", _ask("MC draws at each energy for interaction choice AND shell choice separately", BW4_M_DEFAULT["process"]),
                                 "--continuous-events", _ask("MC draws at each energy for Rayleigh/Compton; photoelectric at each energy AND shell", BW4_M_DEFAULT["ray"]),
                                 "--pair-events", _ask("Pair MC draws at each energy above threshold", BW4_M_DEFAULT["pair"])])
            elif command == "run":
                argv.extend(["--energy", _ask("Photon energy (MeV)", DEFAULT_RUN_ENERGY_MEV),
                             "--histories", _ask("Primary photon histories", DEFAULT_RUN_HISTORIES)])
            elif command == "compare":
                argv.extend(["--energies", _ask("Energies (MeV, comma separated)",
                                              ",".join(map(str, DEFAULT_COMPARISON_ENERGIES))),
                             "--histories", _ask("Primary photon histories per energy per method (MC1, MC2, BeamWeaver)", DEFAULT_COMPARISON_HISTORIES)])
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
        _explain_operation(args.command)
        _execute(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"beamweaver: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0
