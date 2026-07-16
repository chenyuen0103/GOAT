from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from goat.core.artifacts import portraits_data_file, portraits_raw_dir


IMAGE_OPTIONS = {
    "batch_size": 100,
    "class_mode": "binary",
    "color_mode": "grayscale",
}


def save_data(
    data_dir: str | Path,
    save_file: str | Path,
    *,
    stats_file: str | Path,
    target_size: tuple[int, int] = (32, 32),
) -> None:
    data_dir = Path(data_dir).expanduser()
    save_file = Path(save_file).expanduser()
    stats_file = Path(stats_file).expanduser()
    save_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    xs, ys = [], []
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    data_generator = datagen.flow_from_directory(
        str(data_dir),
        shuffle=False,
        target_size=target_size,
        **IMAGE_OPTIONS,
    )
    while True:
        next_x, next_y = next(data_generator)
        xs.append(next_x)
        ys.append(next_y)
        if data_generator.batch_index == 0:
            break

    xs_array = np.concatenate(xs)
    ys_array = np.concatenate(ys)
    filenames = [filename[2:] for filename in data_generator.filenames]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Portrait filenames must be unique after removing class prefixes")

    indices = [index for _, index in sorted(zip(filenames, range(len(filenames))))]
    genders = np.asarray([filename[:1] for filename in data_generator.filenames])[indices]
    binary_genders = genders == "F"
    with stats_file.open("wb") as handle:
        pickle.dump(binary_genders, handle)
    print(f"Saved gender statistics to {stats_file}")

    scipy.io.savemat(
        str(save_file),
        mdict={"Xs": xs_array[indices], "Ys": ys_array[indices]},
    )
    print(f"Saved processed portraits to {save_file}")


def resize_directory(path: str | Path, *, size: int = 32) -> None:
    path = Path(path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Portrait class directory not found: {path}")
    for item in path.iterdir():
        if not item.is_file():
            continue
        with Image.open(item) as image:
            resized = image.resize((size, size), Image.LANCZOS)
            resized.save(item.with_suffix(".png"), "PNG")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_output = portraits_data_file()
    parser = argparse.ArgumentParser(description="Prepare the aligned Portraits dataset.")
    parser.add_argument(
        "--input-dir",
        default=str(portraits_raw_dir()),
        help="Directory containing the extracted M/ and F/ folders.",
    )
    parser.add_argument(
        "--output-file",
        default=str(default_output),
        help="Destination .mat file consumed by GOAT.",
    )
    parser.add_argument(
        "--stats-file",
        default=str(default_output.parent / "portraits_gender_stats"),
    )
    parser.add_argument("--size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).expanduser()
    for class_name in ("M", "F"):
        resize_directory(input_dir / class_name, size=args.size)
    save_data(
        input_dir,
        args.output_file,
        stats_file=args.stats_file,
        target_size=(args.size, args.size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
