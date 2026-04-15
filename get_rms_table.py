from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCES = [
    # BASE_DIR / "experiments" / "harvester_80mm_v2",
    BASE_DIR / "experiments" / "harvester_50mm_ID1",
    BASE_DIR / "experiments_reversed" / "harvester_50mm_ID1",
]
DEFAULT_OUTPUT = BASE_DIR / "rms_table.csv"

EMF_COLUMN_INDEX = 1
CSV_SEPARATOR = ";"
CSV_DECIMAL = ","


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    source: Path


def parse_frequency_from_name(file_path: Path) -> float | None:
    raw = file_path.stem.replace("_", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def infer_dataset_label(source: Path) -> str:
    match = re.search(r"(\d+)\s*mm", source.name, flags=re.IGNORECASE)
    if match:
        label = f"{match.group(1)}мм"
    else:
        label = source.name

    if any("reversed" in part.lower() for part in source.parts):
        label = f"{label} reversed"

    return label


def exp_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name.lower()
    return sys.maxsize, path.name.lower()


def discover_data_directories(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Directory does not exist: {source}")

    if source.is_file():
        raise NotADirectoryError(f"Expected a directory, got file: {source}")

    if source.name.lower() == "data":
        return [source]

    data_dirs: list[Path] = []

    exp_dirs = sorted(
        [path for path in source.iterdir() if path.is_dir() and path.name.lower().startswith("exp_")],
        key=exp_sort_key,
    )
    for exp_dir in exp_dirs:
        data_dir = exp_dir / "data"
        if data_dir.is_dir():
            data_dirs.append(data_dir)

    if data_dirs:
        return data_dirs

    direct_data_dir = source / "data"
    if direct_data_dir.is_dir():
        return [direct_data_dir]

    raise FileNotFoundError(
        f"No data folders found under {source}. Expected exp_*\\data or data."
    )


def infer_experiment_number(data_dir: Path, fallback_index: int) -> int:
    parent_name = data_dir.parent.name.lower()
    if parent_name.startswith("exp_"):
        match = re.search(r"(\d+)", data_dir.parent.name)
        if match:
            return int(match.group(1))
    return fallback_index


def load_signal_mv(file_path: Path, column_index: int = EMF_COLUMN_INDEX) -> np.ndarray:
    raw = pd.read_csv(file_path, sep=CSV_SEPARATOR, header=None, dtype=str)
    if raw.shape[1] <= column_index:
        raise ValueError(
            f"Expected at least {column_index + 1} columns, got {raw.shape[1]}"
        )

    values = pd.to_numeric(
        raw.iloc[:, column_index]
        .astype(str)
        .str.strip()
        .str.replace(CSV_DECIMAL, ".", regex=False),
        errors="coerce",
    ).dropna()

    if values.empty:
        raise ValueError("No numeric values found in the EMF column")

    return values.to_numpy(dtype=float)


def compute_rms(values: np.ndarray, center: bool = False) -> float:
    arr = np.asarray(values, dtype=float)
    if center:
        arr = arr - np.mean(arr)
    return float(np.sqrt(np.mean(arr**2)))


def collect_rms_by_frequency(
    source: Path,
    dataset_label: str,
    center: bool = False,
) -> tuple[list[pd.Series], list[str], list[Path]]:
    experiment_series: list[pd.Series] = []
    errors: list[str] = []
    data_dirs = discover_data_directories(source)

    for experiment_index, data_dir in enumerate(data_dirs, start=1):
        rms_by_frequency: dict[float, float] = {}
        column_label = (
            f"{dataset_label} ({infer_experiment_number(data_dir, experiment_index)})"
        )

        for file_path in sorted(data_dir.glob("*.csv"), key=exp_sort_key):
            freq_hz = parse_frequency_from_name(file_path)
            if freq_hz is None:
                continue

            try:
                signal_mv = load_signal_mv(file_path)
                rms_by_frequency[freq_hz] = compute_rms(signal_mv, center=center)
            except Exception as exc:
                relative_path = file_path.relative_to(source)
                errors.append(f"{relative_path}: {exc}")

        experiment_series.append(pd.Series(rms_by_frequency, name=column_label, dtype=float))

    return experiment_series, errors, data_dirs


def build_rms_table(
    datasets: list[DatasetSpec],
    center: bool = False,
    only_common: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    series_list: list[pd.Series] = []
    errors: list[str] = []
    info_lines: list[str] = []

    for dataset in datasets:
        experiment_series, dataset_errors, data_dirs = collect_rms_by_frequency(
            dataset.source,
            dataset.label,
            center=center,
        )
        series_list.extend(experiment_series)

        experiments = ", ".join(data_dir.parent.name for data_dir in data_dirs)
        info_lines.append(
            f"{dataset.label}: found {len(data_dirs)} experiments ({experiments}) in {dataset.source}"
        )
        errors.extend(f"{dataset.label}: {message}" for message in dataset_errors)

    if not series_list:
        return pd.DataFrame(columns=["freq_Hz"]), errors, info_lines

    table = pd.concat(series_list, axis=1).sort_index()
    table.index.name = "freq_Hz"

    if only_common:
        table = table.dropna()

    table = table.reset_index()
    return table, errors, info_lines


def format_frequency(freq_hz: float) -> str:
    if math.isclose(freq_hz, round(freq_hz), abs_tol=1e-9):
        return f"{int(round(freq_hz))} Гц"
    return f"{freq_hz:g} Гц"


def format_mv(value: float, precision: int) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{precision}f}".replace(".", ",") + " мВ"


def format_csv_column_name(label: str) -> str:
    match = re.fullmatch(r"(.+?) \((\d+)\)", label)
    if match:
        return f"{match.group(1)}_мВ ({match.group(2)})"
    return f"{label}_мВ"


def render_plain_table(
    table: pd.DataFrame,
    dataset_labels: list[str],
    precision: int,
) -> str:
    headers = ["Частота", *dataset_labels]
    rows: list[list[str]] = []

    for _, row in table.iterrows():
        rendered_row = [format_frequency(float(row["freq_Hz"]))]
        rendered_row.extend(format_mv(row[label], precision) for label in dataset_labels)
        rows.append(rendered_row)

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def save_csv(
    table: pd.DataFrame,
    output_path: Path,
    dataset_labels: list[str],
    precision: int,
) -> None:
    export = table.rename(columns={"freq_Hz": "Частота_Гц"})
    export = export.rename(columns={label: format_csv_column_name(label) for label in dataset_labels})
    export.to_csv(
        output_path,
        index=False,
        sep=";",
        decimal=",",
        float_format=f"%.{precision}f",
        encoding="utf-8-sig",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an RMS table from experiment directories. "
            "Each source may contain exp_*\\data folders or a direct data folder, "
            "and every experiment becomes a separate output column."
        )
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help=(
            "Dataset roots, for example "
            "experiments\\harvester_80mm_v2 experiments\\harvester_50mm"
        ),
    )
    parser.add_argument(
        "--dir-80mm",
        "--source-80mm",
        dest="dir_80mm",
        type=Path,
        help="Compatibility option for the 80mm dataset root or data folder.",
    )
    parser.add_argument(
        "--dir-50mm",
        "--source-50mm",
        dest="dir_50mm",
        type=Path,
        help="Compatibility option for the 50mm dataset root or data folder.",
    )
    parser.add_argument(
        "--dir-50mm-reversed",
        "--source-50mm-reversed",
        dest="dir_50mm_reversed",
        type=Path,
        help="Compatibility option for the reversed 50mm dataset root or data folder.",
    )
    parser.add_argument(
        "--only-common",
        action="store_true",
        help="Keep only frequencies that exist in all experiment columns.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Subtract the mean value before RMS calculation.",
    )
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save CSV, print only to console.",
    )
    return parser


def resolve_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.sources:
        sources = args.sources
    else:
        sources = [
            # args.dir_80mm or DEFAULT_SOURCES[0],
            args.dir_50mm or DEFAULT_SOURCES[0],
            args.dir_50mm_reversed or DEFAULT_SOURCES[1],
        ]

    return [DatasetSpec(label=infer_dataset_label(source), source=source) for source in sources]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    datasets = resolve_datasets(args)

    table, errors, info_lines = build_rms_table(
        datasets=datasets,
        center=args.center,
        only_common=args.only_common,
    )

    if table.empty:
        print("No RMS data found.", file=sys.stderr)
        return 1

    experiment_labels = [column for column in table.columns if column != "freq_Hz"]
    print(render_plain_table(table, experiment_labels, args.precision))

    if not args.no_save:
        output_path = args.output.resolve()
        save_csv(table, output_path, experiment_labels, args.precision)
        print(f"\nCSV saved to: {output_path}", file=sys.stderr)

    if info_lines:
        print("\nSources:", file=sys.stderr)
        for line in info_lines:
            print(f"- {line}", file=sys.stderr)

    if errors:
        print("\nWarnings:", file=sys.stderr)
        for message in errors:
            print(f"- {message}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
