from __future__ import annotations

import csv
import io
import pickle
import random
import sys
import time
from typing import Any

import click
import numpy as np
from pydantic import BaseModel

from guidellm.utils import EncodingTypesAlias, MessageEncoding, SerializationTypesAlias

from .utils import create_all_test_objects


def calculate_size(obj: Any) -> int:
    if isinstance(obj, BaseModel):
        return sys.getsizeof(obj.__dict__)

    if isinstance(obj, (tuple, list)) and any(
        isinstance(item, BaseModel) for item in obj
    ):
        return sum(
            sys.getsizeof(item.__dict__)
            if isinstance(item, BaseModel)
            else sys.getsizeof(item)
            for item in obj
        )
    elif isinstance(obj, dict) and any(
        isinstance(value, BaseModel) for value in obj.values()
    ):
        return sum(
            sys.getsizeof(value.__dict__)
            if isinstance(value, BaseModel)
            else sys.getsizeof(value)
            for value in obj.values()
            if isinstance(value, BaseModel)
        )

    return sys.getsizeof(obj)


def time_encode_decode(
    objects: list[Any],
    serialization: SerializationTypesAlias,
    encoding: EncodingTypesAlias,
    pydantic_models: list[type[BaseModel]] | None,
    num_iterations: int,
) -> tuple[float, float, float, float]:
    message_encoding = MessageEncoding(serialization=serialization, encoding=encoding)
    if pydantic_models:
        for model in pydantic_models:
            message_encoding.register_pydantic(model)
    msg_sizes = []
    decoded = []
    encode_time = 0.0
    decode_time = 0.0

    for _ in range(num_iterations):
        for obj in objects:
            start = time.perf_counter_ns()
            message = message_encoding.encode(obj)
            pickled_msg = pickle.dumps(message)
            end = time.perf_counter_ns()
            encode_time += end - start

            msg_sizes.append(calculate_size(pickled_msg))

            start = time.perf_counter_ns()
            message = pickle.loads(pickled_msg)
            decoded.append(message_encoding.decode(message=message))
            end = time.perf_counter_ns()
            decode_time += end - start

    correct = 0
    for obj, dec in zip(objects, decoded):
        if (
            obj == dec
            or type(obj) is type(dec)
            and (
                (
                    hasattr(obj, "model_dump")
                    and hasattr(dec, "model_dump")
                    and obj.model_dump() == dec.model_dump()
                )
                or str(obj) == str(dec)
            )
        ):
            correct += 1

    percent_differences = 100.0 * correct / len(objects)
    avg_msg_size = np.mean(msg_sizes)

    return (
        encode_time / len(objects),
        decode_time / len(objects),
        avg_msg_size,
        percent_differences,
    )


def run_benchmarks(objects_size: int, num_objects: int, num_iterations: int):
    results = {}

    for obj_type, objects, pydantic_models in create_all_test_objects(
        objects_size=objects_size,
        num_objects=num_objects,
    ):
        for serialization in ("dict", "sequence", None):
            for encoding in ("msgpack", "msgspec", None):
                try:
                    encode_time, decode_time, avg_msg_size, percent_differences = (
                        time_encode_decode(
                            objects=objects,
                            serialization=serialization,
                            encoding=encoding,
                            pydantic_models=pydantic_models,
                            num_iterations=num_iterations,
                        )
                    )
                    error = None
                except Exception as err:
                    print(
                        f"Error occurred while benchmarking {obj_type} for "
                        f"serialization={serialization} and encoding={encoding}: {err}"
                    )
                    error = err
                    encode_time = None
                    decode_time = None
                    avg_msg_size = None
                    percent_differences = None

                results[f"{obj_type}_{serialization}_{encoding}"] = {
                    "obj_type": obj_type,
                    "serialization": serialization,
                    "encoding": encoding,
                    "encode_time": encode_time,
                    "decode_time": decode_time,
                    "total_time": (
                        encode_time + decode_time
                        if encode_time is not None and decode_time is not None
                        else None
                    ),
                    "avg_msg_size": avg_msg_size,
                    "percent_differences": percent_differences,
                    "err": error,
                }

    # Print results as a CSV table

    # Create CSV output
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(
        [
            "Object Type",
            "Serialization",
            "Encoding",
            "Encode Time (ns)",
            "Decode Time (ns)",
            "Total Time (ns)",
            "Avg Message Size (bytes)",
            "Accuracy (%)",
            "Error",
        ]
    )

    # Write data rows
    for result in results.values():
        writer.writerow(
            [
                result["obj_type"],
                result["serialization"],
                result["encoding"],
                result["encode_time"],
                result["decode_time"],
                result["total_time"],
                result["avg_msg_size"],
                result["percent_differences"],
                result["err"],
            ]
        )

    # Print the CSV table
    print(output.getvalue())


@click.command()
@click.option("--size", default=1024, type=int, help="Size of each object in bytes")
@click.option(
    "--objects", default=1000, type=int, help="Number of objects to benchmark"
)
@click.option("--iterations", default=5, type=int, help="Number of iterations to run")
def main(size, objects, iterations):
    random.seed(42)
    run_benchmarks(objects_size=size, num_objects=objects, num_iterations=iterations)


if __name__ == "__main__":
    run_benchmarks(objects_size=1024, num_objects=10, num_iterations=5)
