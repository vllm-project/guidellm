import random
from collections.abc import Iterator

__all__ = ["IntegerRangeSampler"]


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

        while True:
            if calc_min == calc_max:
                yield calc_min
            elif not self.variance:
                yield self.rng.randint(calc_min, calc_max)
            else:
                rand = self.rng.gauss(self.average, self.variance)
                yield round(max(calc_min, min(calc_max, rand)))
