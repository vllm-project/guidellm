from __future__ import annotations

import random
import string
import sys
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from guidellm.backend import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.scheduler import RequestSchedulerTimings, ScheduledRequestInfo

__all__ = [
    "TestModel",
    "calculate_size",
    "create_all_test_objects",
    "create_test_objects",
    "generate_str",
    "generate_strs_dict",
    "generate_strs_list",
]


class TestModel(BaseModel):
    test_str: str = Field(default="")
    test_int: int = Field(default=0)
    test_float: float = Field(default=0.0)
    test_bool: bool = Field(default=True)


def generate_str(target_bytes: int) -> str:
    chars = string.ascii_letters + string.digits + " "
    return "".join(random.choice(chars) for _ in range(target_bytes))


def generate_strs_list(target_bytes: int, num_strs: int) -> list[str]:
    bytes_per_str = target_bytes // num_strs

    return [
        generate_str(
            bytes_per_str + 1 if ind < target_bytes % num_strs else bytes_per_str
        )
        for ind in range(num_strs)
    ]


def generate_strs_dict(target_bytes: int, num_strs: int) -> dict[str, str]:
    bytes_per_element = target_bytes // num_strs
    bytes_per_key = bytes_per_element // 4
    bytes_per_value = bytes_per_element - bytes_per_key

    return {
        generate_str(bytes_per_key): generate_str(
            bytes_per_value + 1 if ind < num_strs - 1 else bytes_per_value
        )
        for ind in range(num_strs)
    }


def create_test_objects(
    type_: Literal[
        "bytes",
        "str",
        "list",
        "dict",
        "pydantic",
        "tuple(pydantic)",
        "dict[pydantic]",
        "tuple[GenerativeUpdate]",
        "tuple[GenerationResponse]",
    ],
    objects_size: int,
    num_objects: int,
) -> tuple[list[Any], list[type[BaseModel]] | None]:
    if type_ == "bytes":
        return [random.randbytes(objects_size) for _ in range(num_objects)], None

    if type_ == "str":
        return [generate_str(objects_size) for _ in range(num_objects)], None

    if type_ == "list":
        return [generate_strs_list(objects_size, 10) for _ in range(num_objects)], None

    if type_ == "dict":
        return [generate_strs_dict(objects_size, 10) for _ in range(num_objects)], None

    if type_ == "pydantic":
        return (
            [
                TestModel(
                    test_str=generate_str(objects_size),
                    test_int=random.randint(1, 100),
                    test_float=random.random(),
                    test_bool=random.choice([True, False]),
                )
                for _ in range(num_objects)
            ],
            [TestModel],
        )

    if type_ == "tuple(pydantic)":
        return [
            (
                TestModel(
                    test_str=generate_str(objects_size // 8),
                    test_int=random.randint(1, 100),
                    test_float=random.random(),
                    test_bool=random.choice([True, False]),
                ),
                TestModel(
                    test_str=generate_str(objects_size // 2),
                    test_int=random.randint(1, 100),
                    test_float=random.random(),
                    test_bool=random.choice([True, False]),
                ),
                TestModel(
                    test_str=generate_str(objects_size // 4 + objects_size // 8),
                    test_int=random.randint(1, 100),
                    test_float=random.random(),
                    test_bool=random.choice([True, False]),
                ),
            )
        ], [TestModel]

    if type_ == "dict[pydantic]":
        return [
            {
                generate_str(8): TestModel(
                    test_str=generate_str(objects_size // 4),
                    test_int=random.randint(1, 100),
                    test_float=random.random(),
                    test_bool=random.choice([True, False]),
                ),
                generate_str(8): TestModel(
                    test_str=generate_str(objects_size // 2 + objects_size // 4),
                    test_int=random.randint(1, 100),
                    test_float=random.random(),
                    test_bool=random.choice([True, False]),
                ),
            }
            for _ in range(num_objects)
        ], [TestModel]

    if type_ == "tuple[GenerativeUpdate]":
        return [
            (
                None,
                GenerationRequest(
                    content=generate_str(objects_size),
                ),
                ScheduledRequestInfo(
                    scheduler_timings=RequestSchedulerTimings(
                        targeted_start=time.time(),
                        queued=time.time(),
                        dequeued=time.time(),
                        scheduled_at=time.time(),
                        resolve_start=time.time(),
                        resolve_end=time.time(),
                        finalized=time.time(),
                    ),
                    request_timings=GenerationRequestTimings(
                        request_start=time.time(),
                        request_end=time.time(),
                        first_iteration=time.time(),
                        last_iteration=time.time(),
                    ),
                ),
            )
            for _ in range(num_objects)
        ], [GenerationRequest, ScheduledRequestInfo]

    if type_ == "tuple[GenerationResponse]":
        return [
            (
                GenerationResponse(
                    request_id=str(uuid.uuid4()),
                    request_args={},
                    value=generate_str(objects_size // 2),
                ),
                GenerationRequest(
                    content=generate_str(objects_size // 2),
                ),
                ScheduledRequestInfo(
                    scheduler_timings=RequestSchedulerTimings(
                        targeted_start=time.time(),
                        queued=time.time(),
                        dequeued=time.time(),
                        scheduled_at=time.time(),
                        resolve_start=time.time(),
                        resolve_end=time.time(),
                        finalized=time.time(),
                    ),
                    request_timings=GenerationRequestTimings(
                        request_start=time.time(),
                        request_end=time.time(),
                        first_iteration=time.time(),
                        last_iteration=time.time(),
                    ),
                ),
            )
            for _ in range(num_objects)
        ], [
            GenerationResponse,
            GenerationRequest,
            ScheduledRequestInfo,
        ]

    raise ValueError(f"Unknown type_: {type_}")


def create_all_test_objects(
    objects_size: int, num_objects: int
) -> list[tuple[str, list[Any], dict[str, type[BaseModel]] | None]]:
    tests = []

    for object_type in (
        "bytes",
        "str",
        "list",
        "dict",
        "pydantic",
        "tuple(pydantic)",
        "dict[pydantic]",
        "tuple[GenerativeUpdate]",
        "tuple[GenerationResponse]",
    ):
        tests.append(
            (object_type, *create_test_objects(object_type, objects_size, num_objects))
        )

    return tests


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
