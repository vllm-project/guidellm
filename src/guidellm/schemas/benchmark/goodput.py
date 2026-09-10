"""
Service level objective definitions for goodput measurement.

Goodput is the rate of requests that complete within a set of per-request latency
objectives, as opposed to throughput which counts every completed request. A
request is conforming only when it satisfies every configured objective.

Objective names follow the convention used by vLLM's ``benchmark_serving``
(``ttft``, ``tpot``, ``e2el``, all in milliseconds) so that objectives can be
carried between the two tools unchanged.
"""

from __future__ import annotations

from pydantic import Field, PositiveFloat, model_validator

from guidellm.schemas import StandardBaseModel

__all__ = ["GoodputSLO"]


class GoodputSLO(StandardBaseModel):
    """
    Per-request latency objectives defining which requests count as conforming.

    Every objective left unset is ignored. A request conforms when it satisfies
    all objectives that are set; a benchmark with no objectives set has no
    meaningful goodput and reports it as None.

    Note the mapping between objective names and GuideLLM metrics. ``tpot`` is
    compared against :attr:`GenerativeRequestStats.inter_token_latency_ms`,
    which excludes the first token, and not against GuideLLM's
    ``time_per_output_token_ms``, which includes it. This is the closest
    GuideLLM metric to vLLM's ``tpot`` but is not identical: vLLM divides by
    the interval ending at the request's completion, while inter-token latency
    ends at the last token received.

    Example:
    ::
        slo = GoodputSLO(ttft_ms=2000, tpot_ms=100)
        conforming = slo.is_conforming(ttft_ms=150.0, tpot_ms=12.0, e2el_ms=None)
    """

    ttft_ms: PositiveFloat | None = Field(
        default=None,
        description=(
            "Maximum time to first token in milliseconds. Compared against "
            "each request's time_to_first_token_ms"
        ),
        examples=[2000.0],
    )
    tpot_ms: PositiveFloat | None = Field(
        default=None,
        description=(
            "Maximum time per output token in milliseconds, excluding the "
            "first token. Compared against each request's "
            "inter_token_latency_ms. Requests producing one token or fewer have "
            "no inter-token latency and are left undetermined"
        ),
        examples=[100.0],
    )
    e2el_ms: PositiveFloat | None = Field(
        default=None,
        description=(
            "Maximum end-to-end request latency in milliseconds. Compared "
            "against each request's request_latency, converted from seconds"
        ),
        examples=[30000.0],
    )

    @model_validator(mode="after")
    def _require_an_objective(self) -> GoodputSLO:
        """
        Validate that at least one objective is set.

        :return: The validated instance
        :raises ValueError: If no objective is set
        """
        if all(value is None for value in (self.ttft_ms, self.tpot_ms, self.e2el_ms)):
            raise ValueError(
                "GoodputSLO requires at least one of ttft_ms, tpot_ms, or e2el_ms"
            )

        return self

    def is_conforming(
        self,
        ttft_ms: float | None,
        tpot_ms: float | None,
        e2el_ms: float | None,
    ) -> bool | None:
        """
        Determine whether one request's measured latencies satisfy the objectives.

        A request is undetermined as soon as any configured objective has no
        corresponding measurement, even if another objective is already
        breached. Deciding such a request on its measurable objectives alone
        would bias the population it is averaged over: on a workload where an
        objective is never measurable, only the requests that happen to breach
        a different objective would remain, driving attainment to zero.

        :param ttft_ms: Measured time to first token in milliseconds
        :param tpot_ms: Measured inter-token latency in milliseconds
        :param e2el_ms: Measured end-to-end latency in milliseconds
        :return: True if conforming, False if violating, None if undetermined
        """
        measured = (ttft_ms, tpot_ms, e2el_ms)
        objectives = (self.ttft_ms, self.tpot_ms, self.e2el_ms)

        conforming = True
        for value, objective in zip(measured, objectives, strict=True):
            if objective is None:
                continue
            if value is None:
                return None
            if value > objective:
                conforming = False

        return conforming
