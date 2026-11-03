"""Command-line entry point.

python main.py train   --data data/ddd
python main.py detect  --model models/ddd_cnn_lstm.keras
python main.py predict --model models/ddd_cnn_lstm.keras path/to/face.png
"""

import argparse

from drowsiness.config import DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR, IMG_SIZE, TIMESTEPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Driver drowsiness detection with a CNN-LSTM (train / detect / predict)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train the CNN-LSTM on the DDD dataset")
    t.add_argument("--data", required=True, help="Dataset folder (contains 'Drowsy' and 'Non Drowsy')")
    t.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Where to save the model")
    t.add_argument("--reports", default=str(DEFAULT_REPORT_DIR), help="Where to write plots and metrics")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Frames per sequence")
    t.add_argument("--img-size", type=int, default=IMG_SIZE)
    t.add_argument("--stride", type=int, default=2, help="Step between sliding windows")
    t.add_argument(
        "--split",
        choices=["subject", "random"],
        default="subject",
        help="'subject' = unseen drivers in validation (recommended)",
    )
    t.add_argument("--val-fraction", type=float, default=0.2)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=42)

    d = sub.add_parser("detect", help="Real-time detection on a webcam or video file")
    d.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help=".keras or legacy .h5 model")
    d.add_argument("--source", default="0", help="Webcam index or path to a video file")
    d.add_argument("--threshold", type=float, default=0.6, help="Drowsiness probability threshold")
    d.add_argument("--alarm-after", type=float, default=1.5, help="Seconds of drowsiness before the alarm")
    d.add_argument("--smoothing", type=int, default=8, help="Predictions in the moving average")
    d.add_argument("--no-sound", action="store_true", help="Visual alert only")
    d.add_argument("--no-display", action="store_true", help="Run without a window")
    d.add_argument("--output", help="Save the annotated video (e.g. demo.mp4)")

    p = sub.add_parser("predict", help="Classify face images or folders of frames")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("inputs", nargs="+", help="Image files and/or folders of consecutive frames")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        from drowsiness.train import train

        train(
            data_dir=args.data,
            model_path=args.model,
            report_dir=args.reports,
            epochs=args.epochs,
            batch_size=args.batch_size,
            timesteps=args.timesteps,
            img_size=args.img_size,
            stride=args.stride,
            split=args.split,
            val_fraction=args.val_fraction,
            learning_rate=args.lr,
            seed=args.seed,
        )
    elif args.command == "detect":
        from drowsiness.detector import run

        run(
            model_path=args.model,
            source=args.source,
            threshold=args.threshold,
            alarm_after=args.alarm_after,
            smoothing=args.smoothing,
            sound=not args.no_sound,
            display=not args.no_display,
            output=args.output,
        )
    elif args.command == "predict":
        from drowsiness.predict import predict_paths

        for name, label, prob in predict_paths(args.model, args.inputs):
            print(f"{name:<40} {label:<7} drowsiness={prob:.1%}")


if __name__ == "__main__":
    main()
