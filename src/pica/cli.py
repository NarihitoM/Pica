import argparse
import sys

from pica.utils.paths import annotations_dir, config_path, home, model_path

COMMAND = "narihito-pica"


def gesture_names() -> list[str]:
    """Every gesture the config references, in a sensible order to record them."""
    from pica.utils.config import load_config

    config = load_config()
    names = []
    cursor = config.get("cursor", {}).get("gesture")
    if cursor:
        names.append(cursor)
    names.extend(config.get("gestures", {}))
    scroll = config.get("scroll", {})
    names.extend(n for n in (scroll.get("up_gesture"), scroll.get("down_gesture")) if n)
    # two_hand_* is composed at runtime from two hands doing the same pose, so there's
    # nothing to record for it
    return [n for n in dict.fromkeys(names) if not n.startswith("two_hand_")]


def cmd_run(args):
    from pica.pipeline.run_app import run

    if not model_path().exists():
        sys.exit(f"no trained model at {model_path()}\nrecord gestures with '{COMMAND} collect', then run '{COMMAND} train'")
    run()


def cmd_collect(args):
    from pica.pipeline.collect import collect

    labels = [args.gesture] if args.gesture else gesture_names()
    if not labels:
        sys.exit(f"no gestures listed in {config_path()}")

    for index, label in enumerate(labels, start=1):
        if not args.gesture:
            prompt = f"[{index}/{len(labels)}] get ready for '{label}' -- Enter to record, 's' to skip, 'q' to stop: "
            answer = input(prompt).strip().lower()
            if answer == "q":
                break
            if answer == "s":
                continue
        collect(
            label,
            num_samples=args.samples,
            camera_id=args.camera,
            append=not args.replace,
            position=None if args.gesture else (index, len(labels)),
            next_label=None if args.gesture or index >= len(labels) else labels[index],
        )


def cmd_train(args):
    from pica.pipeline.train import train

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


def cmd_where(args):
    print(f"home:        {home()}")
    print(f"config:      {config_path()}")
    print(f"recordings:  {annotations_dir()}")
    print(f"model:       {model_path()}{'' if model_path().exists() else '  (not trained yet)'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=COMMAND, description="Control your PC with hand gestures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="start gesture control")
    run_parser.set_defaults(func=cmd_run)

    collect_parser = subparsers.add_parser("collect", help="record gesture samples (all configured gestures if none named)")
    collect_parser.add_argument("gesture", nargs="?", help="record just this one gesture")
    collect_parser.add_argument("--samples", type=int, default=200)
    collect_parser.add_argument("--camera", type=int, default=0)
    collect_parser.add_argument("--replace", action="store_true", help="overwrite instead of appending to existing samples")
    collect_parser.set_defaults(func=cmd_collect)

    train_parser = subparsers.add_parser("train", help="train the classifier on your recordings")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.set_defaults(func=cmd_train)

    where_parser = subparsers.add_parser("where", help="show where your config, recordings and model live")
    where_parser.set_defaults(func=cmd_where)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def demo():
    parser = build_parser()

    args = parser.parse_args(["collect", "open_palm", "--samples", "50", "--replace"])
    assert args.gesture == "open_palm" and args.samples == 50 and args.replace is True

    args = parser.parse_args(["collect"])
    assert args.gesture is None and args.replace is False, "bare collect must loop and append"

    assert parser.parse_args(["train", "--epochs", "5"]).epochs == 5
    assert parser.parse_args(["run"]).func is cmd_run

    try:
        parser.parse_args([])
        raise AssertionError("expected a parser error when no subcommand is given")
    except SystemExit:
        pass

    names = gesture_names()
    assert "open_palm" in names and len(names) == len(set(names)), names
    print("cli demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
