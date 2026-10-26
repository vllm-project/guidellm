"""
Tests for random sampling utilities.
"""

import itertools

import pytest

from guidellm.utils.random import derive_seed


class TestDeriveSeed:
    """Test cases for derive_seed."""

    @pytest.mark.smoke
    def test_same_inputs_give_same_seed(self):
        """The derived seed depends only on the base seed and the keys."""
        assert derive_seed(42, 1, 3) == derive_seed(42, 1, 3)

    @pytest.mark.sanity
    def test_no_collisions_across_seeds_passes_and_streams(self):
        """Neighboring base seeds, passes and streams never share a seed."""
        inputs = list(itertools.product(range(40, 50), range(5), range(10)))
        seeds = {
            derive_seed(seed, iteration, stream) for seed, iteration, stream in inputs
        }
        assert len(seeds) == len(inputs)

    @pytest.mark.sanity
    def test_offset_seeds_do_not_line_up(self):
        """Stream 1 of seed 42 differs from stream 0 of seed 43 and of pass 1."""
        assert derive_seed(42, 0, 1) != derive_seed(43, 0, 0)
        assert derive_seed(42, 0, 1) != derive_seed(42, 1, 0)

    @pytest.mark.sanity
    def test_extra_key_changes_seed(self):
        """A branch index gives each branch its own seed."""
        assert derive_seed(42, 0, 8, 0) != derive_seed(42, 0, 8, 1)
        assert derive_seed(42, 0, 8) != derive_seed(42, 0, 8, 0)

    @pytest.mark.sanity
    def test_negative_seed_is_accepted(self):
        """A negative base seed maps to a valid 64-bit seed."""
        seed = derive_seed(-1, 0, 0)
        assert 0 <= seed < 2**64
