from pathlib import Path

from datasets import Dataset

SUPPORTED_TYPES = {
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
}


def save_dataset_to_file(dataset: Dataset, output_path: str | Path) -> None:
    """
    Saves a HuggingFace Dataset to file in a supported format.

    :param dataset: Dataset to save.
    :param output_path: Output file path (.json, .jsonl, .csv, .parquet).
    :raises ValueError: If the file extension is not supported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        dataset.to_csv(output_path)
    elif suffix in {".json", ".jsonl"}:
        dataset.to_json(output_path)
    elif suffix == ".parquet":
        dataset.to_parquet(output_path)
    else:
        raise ValueError(
            f"Unsupported file suffix '{suffix}' in output_path'{output_path}'."
            f" Only {SUPPORTED_TYPES} are supported."
        )
