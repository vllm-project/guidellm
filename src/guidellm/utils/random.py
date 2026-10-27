import random
from collections.abc import Iterator

import numpy as np

__all__ = ["FloatRangeSampler", "IntegerRangeSampler", "derive_seed"]


def derive_seed(random_seed: int, *keys: int) -> int:
    """
    Derive an independent seed for one random stream from a base seed.

    Seeds made by adding small offsets to a base seed collide across base seeds
    and passes: a stream seeded with ``seed + 1`` shares its seed with the
    stream seeded with ``seed`` when the base seed is one higher. Mixing the
    base seed and the keys through ``numpy.random.SeedSequence`` gives each
    distinct ``(random_seed, *keys)`` its own seed.

    :param random_seed: The base seed.
    :param keys: Non-negative integers naming the stream, such as the pass
        index and a fixed stream id.
    :return: A 64-bit seed for ``random.Random`` or ``Faker.seed_instance``.
    """
    sequence = np.random.SeedSequence(random_seed % 2**64, spawn_key=keys)
    return int(sequence.generate_state(1, np.uint64)[0])


class IntegerRangeSampler:
    def __init__(
        self,
        average: int,
        variance: int | None,
        min_value: int | None,
        max_value: int | None,
        random_seed: int,
    ):
        self.average = average
        self.variance = variance
        self.min_value = min_value
        self.max_value = max_value
        self.seed = random_seed
        self.rng = random.Random(random_seed)  # noqa: S311

    def __iter__(self) -> Iterator[int]:
        calc_min = self.min_value
        if calc_min is None:
            calc_min = max(
                1, self.average - 5 * self.variance if self.variance else self.average
            )
        calc_max = self.max_value
        if calc_max is None:
            calc_max = (
                self.average + 5 * self.variance if self.variance else self.average
            )

        if calc_min == calc_max:
            while True:
                yield calc_min
        elif not self.variance:
            while True:
                yield self.rng.randint(calc_min, calc_max)
        else:
            while True:
                rand = self.rng.gauss(self.average, self.variance)
                yield round(max(calc_min, min(calc_max, rand)))


class FloatRangeSampler:
    def __init__(
        self,
        average: float,
        variance: float | None,
        min_value: float | None,
        max_value: float | None,
        random_seed: int,
    ):
        self.average = average
        self.variance = variance
        self.min_value = min_value
        self.max_value = max_value
        self.seed = random_seed
        self.rng = random.Random(random_seed)  # noqa: S311

    def __iter__(self) -> Iterator[float]:
        calc_min = self.min_value
        if calc_min is None:
            calc_min = max(
                0.0, self.average - 5 * self.variance if self.variance else self.average
            )
        calc_max = self.max_value
        if calc_max is None:
            calc_max = (
                self.average + 5 * self.variance if self.variance else self.average
            )

        if calc_min == calc_max:
            while True:
                yield calc_min
        elif not self.variance:
            while True:
                yield self.rng.uniform(calc_min, calc_max)
        else:
            while True:
                rand = self.rng.gauss(self.average, self.variance)
                yield max(calc_min, min(calc_max, rand))
