"""
Unit tests for guidellm.data.deserializers.file module.

### WRITTEN BY AI ###
"""

import csv
import io
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, DatasetDict
from pyarrow import ipc

from guidellm.data.deserializers.deserializer import DataNotSupportedError
from guidellm.data.deserializers.file import (
    ArrowFileDatasetDeserializer,
    CSVFileDatasetDeserializer,
    DBFileDatasetDeserializer,
    FileDataArgs,
    HDF5FileDatasetDeserializer,
    JSONFileDatasetDeserializer,
    ParquetFileDatasetDeserializer,
    TarFileDatasetDeserializer,
    TextFileDatasetDeserializer,
)


def processor_factory():
    return None


###################
# Tests text file deserializer
###################


@pytest.mark.sanity
def test_text_file_deserializer_success(tmp_path):
    """TextFileDatasetDeserializer reads .txt file into Dataset.

    ### WRITTEN BY AI ###
    """
    file_path = tmp_path / "sample.txt"
    file_content = ["hello\n", "world\n"]
    file_path.write_text("".join(file_content))

    deserializer = TextFileDatasetDeserializer()
    config = FileDataArgs(kind="text_file", path=file_path)

    dataset = deserializer(
        config=config, processor_factory=processor_factory(), random_seed=123
    )

    assert isinstance(dataset, Dataset)
    assert dataset["text"] == file_content
    assert len(dataset) == 2


@pytest.mark.sanity
def test_text_file_deserializer_file_not_exists(tmp_path):
    """TextFileDatasetDeserializer raises DataNotSupportedError for missing file.

    ### WRITTEN BY AI ###
    """
    deserializer = TextFileDatasetDeserializer()
    config = FileDataArgs(kind="text_file", path=tmp_path / "missing.txt")

    with pytest.raises(DataNotSupportedError):
        deserializer(
            config=config, processor_factory=processor_factory(), random_seed=0
        )


@pytest.mark.sanity
def test_text_file_deserializer_not_a_file(tmp_path):
    """TextFileDatasetDeserializer raises DataNotSupportedError for directory.

    ### WRITTEN BY AI ###
    """
    directory = tmp_path / "folder"
    directory.mkdir()
    deserializer = TextFileDatasetDeserializer()
    config = FileDataArgs(kind="text_file", path=directory)

    with pytest.raises(DataNotSupportedError):
        deserializer(
            config=config, processor_factory=processor_factory(), random_seed=0
        )


@pytest.mark.sanity
def test_text_file_deserializer_invalid_file_extension(tmp_path):
    """TextFileDatasetDeserializer raises DataNotSupportedError for wrong extension.

    ### WRITTEN BY AI ###
    """
    file_path = tmp_path / "data.ttl"
    file_path.write_text("hello")
    deserializer = TextFileDatasetDeserializer()
    config = FileDataArgs(kind="text_file", path=file_path)

    with pytest.raises(DataNotSupportedError):
        deserializer(
            config=config, processor_factory=processor_factory(), random_seed=0
        )


###################
# Tests parquet file deserializer
###################


def create_parquet_file(path: Path):
    table = pa.Table.from_pydict({"text": ["hello", "world"]})
    pq.write_table(table, path)


@pytest.mark.sanity
def test_parquet_file_deserializer_success(tmp_path):
    """ParquetFileDatasetDeserializer reads .parquet file into Dataset.

    ### WRITTEN BY AI ###
    """
    file_path = tmp_path / "sample.parquet"
    create_parquet_file(file_path)

    deserializer = ParquetFileDatasetDeserializer()
    config = FileDataArgs(kind="parquet_file", path=file_path)

    dataset = deserializer(
        config=config, processor_factory=processor_factory(), random_seed=42
    )

    assert isinstance(dataset, DatasetDict)
    assert dataset["train"].column_names == ["text"]
    assert dataset["train"]["text"] == ["hello", "world"]
    assert len(dataset["train"]["text"]) == 2


@pytest.mark.sanity
def test_parquet_file_deserializer_file_not_exists(tmp_path):
    """ParquetFileDatasetDeserializer raises DataNotSupportedError for missing file.

    ### WRITTEN BY AI ###
    """
    deserializer = ParquetFileDatasetDeserializer()
    config = FileDataArgs(kind="parquet_file", path=tmp_path / "missing.parquet")

    with pytest.raises(DataNotSupportedError):
        deserializer(
            config=config, processor_factory=processor_factory(), random_seed=3
        )


###################
# Tests csv file deserializer
###################


def create_csv_file(path: Path):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text"])
    writer.writerow(["hello world"])
    with path.open("w") as f:
        f.write(output.getvalue())


@pytest.mark.sanity
def test_csv_file_deserializer_success(tmp_path):
    """CSVFileDatasetDeserializer reads .csv file into Dataset.

    ### WRITTEN BY AI ###
    """
    file_path = tmp_path / "sample.csv"
    create_csv_file(file_path)

    deserializer = CSVFileDatasetDeserializer()
    config = FileDataArgs(kind="csv_file", path=file_path)

    dataset = deserializer(
        config=config, processor_factory=processor_factory(), random_seed=43
    )

    assert isinstance(dataset, DatasetDict)
    assert dataset["train"]["text"] == ["hello world"]
    assert len(["train"]) == 1


###################
# Tests json file deserializer
###################


@pytest.mark.sanity
def test_json_file_deserializer_success(tmp_path):
    """JSONFileDatasetDeserializer reads .json file into Dataset.

    ### WRITTEN BY AI ###
    """
    file_path = tmp_path / "sample.json"
    file_content = '{"text": "hello world"}\n'
    file_path.write_text("".join(file_content))

    deserializer = JSONFileDatasetDeserializer()
    config = FileDataArgs(kind="json_file", path=file_path)

    dataset = deserializer(
        config=config, processor_factory=processor_factory(), random_seed=123
    )

    assert isinstance(dataset, DatasetDict)
    assert dataset["train"]["text"] == ["hello world"]
    assert len(dataset) == 1


###################
# Tests arrow file deserializer
###################


@pytest.mark.sanity
def test_arrow_file_deserializer_success(monkeypatch, tmp_path):
    """ArrowFileDatasetDeserializer reads .arrow file into Dataset.

    ### WRITTEN BY AI ###
    """
    table = pa.Table.from_pydict({"text": ["hello", "world"]})
    file_path = tmp_path / "sample.arrow"

    with (
        pa.OSFile(str(file_path), "wb") as sink,
        ipc.RecordBatchFileWriter(sink, table.schema) as writer,
    ):
        writer.write_table(table)

    deserializer = ArrowFileDatasetDeserializer()
    config = FileDataArgs(kind="arrow_file", path=file_path)

    dataset = deserializer(
        config=config, processor_factory=processor_factory(), random_seed=42
    )

    assert isinstance(dataset, DatasetDict)
    assert "train" in dataset
    assert isinstance(dataset["train"], Dataset)
    assert dataset["train"].num_rows == 2


###################
# Tests HDF5 file deserializer
###################


@pytest.mark.skip(
    reason="add pyproject extras group in the future \
                to install hdf5 dependency such as pytables & h5py"
)
def test_hdf5_file_deserializer_success(tmp_path):
    """HDF5FileDatasetDeserializer reads .h5 file into Dataset.

    ### WRITTEN BY AI ###
    """
    df_sample = pd.DataFrame({"text": ["hello", "world"]})
    file_path = tmp_path / "sample.h5"
    df_sample.to_hdf(str(file_path), key="data", mode="w", format="fixed")

    deserializer = HDF5FileDatasetDeserializer()
    config = FileDataArgs(kind="hdf5_file", path=file_path)

    dataset = deserializer(
        config=config, processor_factory=processor_factory(), random_seed=1
    )

    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 2
    assert dataset["text"] == ["hello", "world"]


##################
# Tests DB file deserializer
###################


@pytest.mark.sanity
def test_db_file_deserializer_success(monkeypatch, tmp_path):
    """DBFileDatasetDeserializer reads .db file into Dataset.

    ### WRITTEN BY AI ###
    """
    import sqlite3

    def create_sqlite_db(path: Path):
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE samples (text TEXT)")
        cur.execute("INSERT INTO samples (text) VALUES ('hello')")
        cur.execute("INSERT INTO samples (text) VALUES ('world')")
        conn.commit()
        conn.close()

    db_path = tmp_path / "sample.db"
    create_sqlite_db(db_path)

    mocked_ds = Dataset.from_dict({"text": ["hello", "world"]})

    def mock_from_sql(sql, con, **kwargs):
        assert sql == "SELECT * FROM samples"
        assert con == (str(db_path))
        return mocked_ds

    monkeypatch.setattr("datasets.Dataset.from_sql", mock_from_sql)

    deserializer = DBFileDatasetDeserializer()
    config = FileDataArgs(
        kind="db_file",
        path=db_path,
        load_kwargs={"sql": "SELECT * FROM samples"},
    )

    dataset = deserializer(
        config=config, processor_factory=processor_factory(), random_seed=1
    )

    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 2
    assert dataset["text"] == ["hello", "world"]


##################
# Tests Tar file deserializer
###################


def create_simple_tar(tar_path: str):
    import tarfile

    content = b"hello world\nthis is a tar file\n"
    with tarfile.open(tar_path, "w") as tar:
        data_stream = io.BytesIO(content)
        info = tarfile.TarInfo(name="sample.txt")
        info.size = len(content)
        tar.addfile(info, data_stream)


@pytest.mark.sanity
def test_tar_file_deserializer_success(tmp_path):
    """TarFileDatasetDeserializer reads .tar file into Dataset.

    ### WRITTEN BY AI ###
    """
    file_path = tmp_path / "sample.tar"
    create_simple_tar(str(file_path))

    deserializer = TarFileDatasetDeserializer()
    config = FileDataArgs(kind="tar_file", path=file_path)

    dataset = deserializer(
        config=config, processor_factory=processor_factory(), random_seed=43
    )

    assert isinstance(dataset, DatasetDict)
    assert "train" in dataset
    assert isinstance(dataset["train"], Dataset)
    assert dataset["train"].num_rows == 1
