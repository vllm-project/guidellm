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
    # Arrange: create a temp text file
    file_path = tmp_path / "sample.txt"
    file_content = ["hello\n", "world\n"]
    file_path.write_text("".join(file_content))

    deserializer = TextFileDatasetDeserializer()

    dataset = deserializer(
        data=file_path,
        processor_factory=processor_factory(),
        random_seed=123,
    )

    # Assert
    assert isinstance(dataset, Dataset)
    assert dataset["text"] == file_content
    assert len(dataset) == 2


@pytest.mark.parametrize(
    "invalid_data",
    [
        123,  # Not a path
        None,  # Not a path
        {"file": "abc.txt"},  # Wrong type
    ],
)
@pytest.mark.sanity
def test_text_file_deserializer_invalid_type(invalid_data):
    deserializer = TextFileDatasetDeserializer()

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data=invalid_data,
            processor_factory=processor_factory(),
            random_seed=0,
        )


@pytest.mark.sanity
def test_text_file_deserializer_file_not_exists(tmp_path):
    deserializer = TextFileDatasetDeserializer()
    non_existent_file = tmp_path / "missing.txt"

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data=non_existent_file,
            processor_factory=processor_factory(),
            random_seed=0,
        )


@pytest.mark.sanity
def test_text_file_deserializer_not_a_file(tmp_path):
    deserializer = TextFileDatasetDeserializer()
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data=directory,
            processor_factory=processor_factory(),
            random_seed=0,
        )


@pytest.mark.sanity
def test_text_file_deserializer_invalid_file_extension(tmp_path):
    deserializer = TextFileDatasetDeserializer()

    file_path = tmp_path / "data.ttl"
    file_path.write_text("hello")

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data=file_path,
            processor_factory=processor_factory(),
            random_seed=0,
        )


###################
# Tests parquet file deserializer
###################


def create_parquet_file(path: Path):
    # Arrange: to create a minimal parquet file
    table = pa.Table.from_pydict({"text": ["hello", "world"]})
    pq.write_table(table, path)


@pytest.mark.sanity
def test_parquet_file_deserializer_success(tmp_path):
    file_path = tmp_path / "sample.parquet"
    create_parquet_file(file_path)

    deserializer = ParquetFileDatasetDeserializer()

    dataset = deserializer(
        data=file_path,
        processor_factory=processor_factory(),
        random_seed=42,
    )

    # Assert
    assert isinstance(dataset, DatasetDict)
    assert dataset["train"].column_names == ["text"]
    assert dataset["train"]["text"] == ["hello", "world"]
    assert len(dataset["train"]["text"]) == 2


@pytest.mark.sanity
def test_parquet_file_deserializer_file_not_exists(tmp_path):
    deserializer = ParquetFileDatasetDeserializer()
    missing_file = tmp_path / "missing.parquet"

    with pytest.raises(DataNotSupportedError):
        deserializer(
            data=missing_file,
            processor_factory=processor_factory(),
            random_seed=3,
        )


###################
# Tests csv file deserializer
###################


def create_csv_file(path: Path):
    """Helper to create a minimal csv file."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text"])
    writer.writerow(["hello world"])
    with path.open("w") as f:
        f.write(output.getvalue())


@pytest.mark.sanity
def test_csv_file_deserializer_success(tmp_path):
    # Arrange: create a temp csv file
    file_path = tmp_path / "sample.csv"
    create_csv_file(file_path)

    deserializer = CSVFileDatasetDeserializer()

    dataset = deserializer(
        data=file_path,
        processor_factory=processor_factory(),
        random_seed=43,
    )

    # Assert
    assert isinstance(dataset, DatasetDict)
    assert dataset["train"]["text"] == ["hello world"]
    assert len(["train"]) == 1


###################
# Tests json file deserializer
###################


@pytest.mark.sanity
def test_json_file_deserializer_success(tmp_path):
    # Arrange: create a temp json file
    file_path = tmp_path / "sample.json"
    file_content = '{"text": "hello world"}\n'
    file_path.write_text("".join(file_content))

    deserializer = JSONFileDatasetDeserializer()

    dataset = deserializer(
        data=file_path,
        processor_factory=processor_factory(),
        random_seed=123,
    )

    # Assert
    assert isinstance(dataset, DatasetDict)
    assert dataset["train"]["text"] == ["hello world"]
    assert len(dataset) == 1


###################
# Tests arrow file deserializer
###################


@pytest.mark.sanity
def test_arrow_file_deserializer_success(monkeypatch, tmp_path):
    # Arrange: create a temp arrow file
    table = pa.Table.from_pydict({"text": ["hello", "world"]})
    file_path = tmp_path / "sample.arrow"

    with (
        pa.OSFile(str(file_path), "wb") as sink,
        ipc.RecordBatchFileWriter(sink, table.schema) as writer,
    ):
        writer.write_table(table)

    deserializer = ArrowFileDatasetDeserializer()

    dataset = deserializer(
        data=file_path,
        processor_factory=processor_factory(),
        random_seed=42,
    )

    # assert
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
    df_sample = pd.DataFrame({"text": ["hello", "world"]})
    file_path = tmp_path / "sample.h5"
    df_sample.to_hdf(str(file_path), key="data", mode="w", format="fixed")

    deserializer = HDF5FileDatasetDeserializer()

    dataset = deserializer(
        data=file_path,
        processor_factory=processor_factory(),
        random_seed=1,
    )

    # assert
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 2
    assert dataset["text"] == ["hello", "world"]


##################
# Tests DB file deserializer
###################


@pytest.mark.skip(reason="issue: #492")
def test_db_file_deserializer_success(monkeypatch, tmp_path):
    import sqlite3

    def create_sqlite_db(path: Path):
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE samples (text TEXT)")
        cur.execute("INSERT INTO samples (text) VALUES ('hello')")
        cur.execute("INSERT INTO samples (text) VALUES ('world')")
        conn.commit()
        conn.close()

    # Arrange: create a valid .db file
    db_path = tmp_path / "sample.db"
    create_sqlite_db(db_path)

    # arrange: mock Dataset.from_sql return one dataset
    mocked_ds = Dataset.from_dict({"text": ["hello", "world"]})

    def mock_from_sql(sql, con, **kwargs):
        assert sql == "SELECT * FROM samples"
        assert con == (str(db_path))
        return mocked_ds

    monkeypatch.setattr("datasets.Dataset.from_sql", mock_from_sql)

    deserializer = DBFileDatasetDeserializer()

    dataset = deserializer(
        data=db_path,
        processor_factory=processor_factory(),
        random_seed=1,
    )

    # Assert: result is of type Dataset
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 2
    assert dataset["text"] == ["hello", "world"]


##################
# Tests Tar file deserializer
###################


def create_simple_tar(tar_path: str):
    import tarfile

    # create tar 文件 in write mode
    with tarfile.open(tar_path, "w") as tar:
        # write content to be added to the tar file
        content = b"hello world\nthis is a tar file\n"

        # using BytesIO
        data_stream = io.BytesIO(content)

        # tarinfo: file description info
        info = tarfile.TarInfo(name="sample.txt")
        info.size = len(content)

        # write file to tar archive
        tar.addfile(info, data_stream)


@pytest.mark.sanity
def test_tar_file_deserializer_success(tmp_path):
    file_path = tmp_path / "sample.tar"
    create_simple_tar(file_path)

    deserializer = TarFileDatasetDeserializer()

    dataset = deserializer(
        data=file_path,
        processor_factory=processor_factory(),
        random_seed=43,
    )

    assert isinstance(dataset, DatasetDict)
    assert "train" in dataset
    assert isinstance(dataset["train"], Dataset)
    assert dataset["train"].num_rows == 1
