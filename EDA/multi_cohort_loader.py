from pathlib import Path
import re

import pandas as pd

BASE_FILES_DIR = Path("base_files")
STAGE_FILE_PATTERN = re.compile(r"^responses(\d+)\.csv$", re.IGNORECASE)


def extract_stage_from_filename(file_path: Path):
    match = STAGE_FILE_PATTERN.match(file_path.name)
    if match:
        return int(match.group(1))
    return pd.NA


def load_survey_data(base_files_dir: Path = BASE_FILES_DIR):
    if not base_files_dir.exists():
        raise FileNotFoundError(
            "The base_files directory was not found. Expected cohort data in "
            "'base_files/<cohort>/responses<stage>.csv'."
        )

    source_records = []
    frames = []

    cohort_dirs = sorted(
        [path for path in base_files_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )

    for cohort_dir in cohort_dirs:
        cohort = cohort_dir.name
        csv_files = sorted(
            [
                path
                for path in cohort_dir.iterdir()
                if path.is_file() and STAGE_FILE_PATTERN.match(path.name)
            ],
            key=lambda path: (extract_stage_from_filename(path), path.name),
        )

        for csv_path in csv_files:
            stage = extract_stage_from_filename(csv_path)
            frame = pd.read_csv(csv_path)
            frame["cohort"] = cohort
            frame["stage"] = stage
            frames.append(frame)

            source_records.append(
                {
                    "cohort": cohort,
                    "stage": stage,
                    "file_name": csv_path.name,
                    "relative_path": str(csv_path.as_posix()),
                    "n_rows": len(frame),
                    "n_columns": frame.shape[1],
                }
            )

    if not frames:
        raise FileNotFoundError(
            "No files matching 'responses<stage>.csv' were found under base_files/<cohort>."
        )

    raw = pd.concat(frames, ignore_index=True, sort=False)
    source_files = pd.DataFrame(source_records).sort_values(
        ["cohort", "stage", "file_name"]
    ).reset_index(drop=True)

    return raw, source_files
