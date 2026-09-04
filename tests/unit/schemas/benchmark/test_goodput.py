"""Unit tests for goodput service level objective semantics."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.schemas.benchmark import GoodputSLO


class TestGoodputSLOValidation:
    @pytest.mark.smoke
    def test_requires_at_least_one_objective(self):
        """
        Reject an objective set with nothing configured.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match="at least one of"):
            GoodputSLO()

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"ttft_ms": 2000}, (2000.0, None, None)),
            ({"tpot_ms": 100}, (None, 100.0, None)),
            ({"e2el_ms": 30000}, (None, None, 30000.0)),
            (
                {"ttft_ms": 2000, "tpot_ms": 100, "e2el_ms": 30000},
                (2000.0, 100.0, 30000.0),
            ),
        ],
    )
    def test_accepts_any_single_objective(self, payload, expected):
        """
        Accept an objective set with any one threshold configured.

        ## WRITTEN BY AI ##
        """
        # model_validate raises or returns an instance, so assert on the
        # parsed thresholds rather than on the object existing.
        parsed = GoodputSLO.model_validate(payload)

        assert (parsed.ttft_ms, parsed.tpot_ms, parsed.e2el_ms) == expected

    @pytest.mark.sanity
    @pytest.mark.parametrize("field", ["ttft_ms", "tpot_ms", "e2el_ms"])
    def test_rejects_non_positive_thresholds(self, field):
        """
        Reject a threshold of zero or below, which no request can satisfy.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match=field):
            GoodputSLO.model_validate({field: 0})


class TestGoodputSLOConformance:
    @pytest.mark.smoke
    def test_all_objectives_met_conforms(self):
        """
        Report conformance when every configured objective is satisfied.

        ## WRITTEN BY AI ##
        """
        slo = GoodputSLO(ttft_ms=2000, tpot_ms=100, e2el_ms=30000)
        assert slo.is_conforming(ttft_ms=150.0, tpot_ms=12.0, e2el_ms=5000.0) is True

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("ttft", "tpot", "e2el"),
        [
            (2500.0, 12.0, 5000.0),
            (150.0, 150.0, 5000.0),
            (150.0, 12.0, 45000.0),
        ],
    )
    def test_any_objective_breached_violates(self, ttft, tpot, e2el):
        """
        Report violation when any single configured objective is breached.

        ## WRITTEN BY AI ##
        """
        slo = GoodputSLO(ttft_ms=2000, tpot_ms=100, e2el_ms=30000)
        assert slo.is_conforming(ttft_ms=ttft, tpot_ms=tpot, e2el_ms=e2el) is False

    @pytest.mark.sanity
    def test_threshold_boundary_is_inclusive(self):
        """
        Treat a measurement exactly at the threshold as conforming.

        ## WRITTEN BY AI ##
        """
        slo = GoodputSLO(ttft_ms=2000)
        assert slo.is_conforming(ttft_ms=2000.0, tpot_ms=None, e2el_ms=None) is True
        assert slo.is_conforming(ttft_ms=2000.1, tpot_ms=None, e2el_ms=None) is False

    @pytest.mark.sanity
    def test_unconfigured_objective_ignores_its_measurement(self):
        """
        Ignore a measurement whose objective is not configured.

        ## WRITTEN BY AI ##
        """
        slo = GoodputSLO(e2el_ms=30000)
        assert (
            slo.is_conforming(ttft_ms=999999.0, tpot_ms=999999.0, e2el_ms=5000.0)
            is True
        )

    @pytest.mark.sanity
    def test_missing_measurement_is_undetermined(self):
        """
        Report undetermined when a configured objective has no measurement.

        ## WRITTEN BY AI ##
        """
        slo = GoodputSLO(ttft_ms=2000)
        assert slo.is_conforming(ttft_ms=None, tpot_ms=None, e2el_ms=None) is None

    @pytest.mark.regression
    def test_missing_measurement_dominates_a_breach(self):
        """
        Report undetermined rather than violating when one objective is
        unmeasurable and another is breached.

        Deciding such a request on its measurable objectives alone would bias
        the population attainment averages over: on a non-streaming workload,
        where time to first token is never measured, only requests breaching
        the end-to-end objective would remain determined and attainment would
        collapse to zero regardless of how the server actually performed.

        ## WRITTEN BY AI ##
        """
        slo = GoodputSLO(ttft_ms=2000, e2el_ms=15000)
        assert slo.is_conforming(ttft_ms=None, tpot_ms=None, e2el_ms=20000.0) is None
        assert slo.is_conforming(ttft_ms=None, tpot_ms=None, e2el_ms=5000.0) is None
        # The same holds when the breach is measured before the gap, so the
        # verdict cannot depend on the order objectives are checked in.
        assert slo.is_conforming(ttft_ms=2500.0, tpot_ms=None, e2el_ms=None) is None
