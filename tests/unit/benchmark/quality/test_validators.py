from __future__ import annotations

import numpy as np
import pytest

from guidellm.benchmark.quality.validators import compute_cosine_similarity

# Check for sentence-transformers availability for quality validator tests
try:
    import sentence_transformers  # noqa: F401

    EMBEDDINGS_VALIDATOR_AVAILABLE = True
except ImportError:
    EMBEDDINGS_VALIDATOR_AVAILABLE = False

if EMBEDDINGS_VALIDATOR_AVAILABLE:
    from guidellm.benchmark.quality.validators import EmbeddingsQualityValidator


class TestComputeCosineSimilarity:
    """Tests for cosine similarity computation function."""

    @pytest.mark.smoke
    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        similarity = compute_cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.smoke
    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.smoke
    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1.0."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, abs=1e-6)

    @pytest.mark.sanity
    def test_similar_vectors(self):
        """Test cosine similarity of similar vectors is close to 1.0."""
        vec1 = np.array([1.0, 2.0, 3.0, 4.0])
        vec2 = np.array([1.1, 2.1, 2.9, 4.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity > 0.99
        assert similarity <= 1.0

    @pytest.mark.sanity
    def test_dissimilar_vectors(self):
        """Test cosine similarity of dissimilar vectors is low."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.1, 1.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity < 0.2
        assert similarity >= 0.0

    @pytest.mark.sanity
    def test_normalized_vectors(self):
        """Test with pre-normalized vectors (unit length)."""
        # Pre-normalized to unit length
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.707107, 0.707107, 0.0])  # 45 degrees
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.707107, abs=1e-5)

    @pytest.mark.regression
    def test_high_dimensional_vectors(self):
        """Test with high-dimensional vectors (typical embedding size)."""
        rng = np.random.default_rng(42)
        vec1 = rng.random(384)  # Common embedding dimension
        vec2 = rng.random(384)

        similarity = compute_cosine_similarity(vec1, vec2)
        assert -1.0 <= similarity <= 1.0

    @pytest.mark.regression
    def test_zero_vector_handling(self):
        """Test behavior with zero vectors (edge case)."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        # Zero vector should return 0.0 (implementation handles gracefully)
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    @pytest.mark.regression
    def test_single_dimension_vectors(self):
        """Test with single-dimension vectors."""
        vec1 = np.array([5.0])
        vec2 = np.array([3.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0, abs=1e-6)

        vec3 = np.array([-5.0])
        similarity_neg = compute_cosine_similarity(vec1, vec3)
        assert similarity_neg == pytest.approx(-1.0, abs=1e-6)

    @pytest.mark.sanity
    def test_return_type(self):
        """Test that return type is Python float."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert isinstance(similarity, float)


@pytest.mark.skipif(
    not EMBEDDINGS_VALIDATOR_AVAILABLE,
    reason="EmbeddingsQualityValidator requires sentence-transformers",
)
class TestEmbeddingsQualityValidator:
    """Tests for EmbeddingsQualityValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a validator with a test model."""
        # Use a small, fast model for testing
        return EmbeddingsQualityValidator(
            baseline_model="sentence-transformers/all-MiniLM-L6-v2"
        )

    @pytest.mark.smoke
    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert validator.baseline_model is not None

    @pytest.mark.sanity
    def test_validate_against_baseline_same_model(self, validator):
        """Test validation against baseline with same model."""
        text = "This is a test sentence for embeddings."

        # Get baseline embedding
        baseline_embedding = validator.baseline_model.encode(text)

        # Validate against itself (should be very high similarity)
        similarity = validator.validate_against_baseline(text, baseline_embedding)

        assert similarity == pytest.approx(1.0, abs=1e-6)
        assert isinstance(similarity, float)

    @pytest.mark.sanity
    def test_validate_against_baseline_different_embedding(self, validator):
        """Test validation with a different (random) embedding."""
        text = "This is a test sentence."

        # Create a random embedding (different from baseline)
        rng = np.random.default_rng(42)
        random_embedding = rng.random(384)  # MiniLM dimension
        # Normalize to unit length
        random_embedding = random_embedding / np.linalg.norm(random_embedding)

        similarity = validator.validate_against_baseline(text, random_embedding)

        # Random embedding should have low similarity
        assert similarity < 0.5
        assert similarity >= -1.0

    @pytest.mark.regression
    def test_validate_multiple_texts(self, validator):
        """Test validation with multiple different texts."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny and warm.",
            "Python is a popular programming language.",
        ]

        for text in texts:
            baseline_embedding = validator.baseline_model.encode(text)
            similarity = validator.validate_against_baseline(text, baseline_embedding)
            # Same model should have perfect similarity
            assert similarity == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.sanity
    def test_check_self_consistency_identical_embeddings(self, validator):
        """Test self-consistency with identical embeddings."""
        text = "Test sentence for consistency check."

        # Generate same embedding twice
        emb1 = validator.baseline_model.encode(text)
        emb2 = validator.baseline_model.encode(text)

        consistency = validator.check_self_consistency(text, [emb1, emb2])

        # Should be perfectly consistent
        assert consistency == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.sanity
    def test_check_self_consistency_single_embedding(self, validator):
        """Test self-consistency with only one embedding."""
        text = "Single embedding test."
        emb = validator.baseline_model.encode(text)

        consistency = validator.check_self_consistency(text, [emb])

        # Single embedding should return 1.0 (perfectly consistent)
        assert consistency == 1.0

    @pytest.mark.sanity
    def test_check_self_consistency_empty_list(self, validator):
        """Test self-consistency with empty embedding list."""
        text = "Empty list test."

        consistency = validator.check_self_consistency(text, [])

        # Empty list should return 1.0 (no inconsistency)
        assert consistency == 1.0

    @pytest.mark.regression
    def test_check_self_consistency_multiple_embeddings(self, validator):
        """Test self-consistency with multiple embeddings."""
        text = "Test sentence for multiple embeddings."

        # Generate same embedding multiple times
        embeddings = [validator.baseline_model.encode(text) for _ in range(5)]

        consistency = validator.check_self_consistency(text, embeddings)

        # Should be highly consistent (model is deterministic)
        assert consistency == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.regression
    def test_check_self_consistency_different_embeddings(self, validator):
        """Test self-consistency with intentionally different embeddings."""
        text = "Consistency test."
        rng = np.random.default_rng(42)

        # First embedding from model
        emb1 = validator.baseline_model.encode(text)

        # Second embedding is random
        emb2 = rng.random(384)
        emb2 = emb2 / np.linalg.norm(emb2)

        consistency = validator.check_self_consistency(text, [emb1, emb2])

        # Should have low consistency
        assert consistency < 0.5

    @pytest.mark.sanity
    def test_embedding_dimensions(self, validator):
        """Test that baseline model produces expected dimensions."""
        text = "Dimension test."
        embedding = validator.baseline_model.encode(text)

        # MiniLM-L6-v2 produces 384-dimensional embeddings
        assert embedding.shape == (384,)

    @pytest.mark.regression
    def test_baseline_model_deterministic(self, validator):
        """Test that baseline model produces deterministic results."""
        text = "Deterministic test."

        # Encode same text multiple times
        emb1 = validator.baseline_model.encode(text)
        emb2 = validator.baseline_model.encode(text)
        emb3 = validator.baseline_model.encode(text)

        # All embeddings should be identical
        assert np.allclose(emb1, emb2, atol=1e-6)
        assert np.allclose(emb2, emb3, atol=1e-6)

    @pytest.mark.sanity
    def test_similarity_range(self, validator):
        """Test that similarity values are within valid range."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Completely different topic about weather.",
        ]

        for text in texts:
            baseline_emb = validator.baseline_model.encode(text)
            similarity = validator.validate_against_baseline(text, baseline_emb)

            # Similarity should always be in [-1, 1]
            assert -1.0 <= similarity <= 1.0

    @pytest.mark.regression
    def test_vllm_tolerance_standard(self, validator):
        """Test that similarity meets vLLM standard tolerance (1e-2)."""
        text = "vLLM tolerance test."

        baseline_emb = validator.baseline_model.encode(text)
        similarity = validator.validate_against_baseline(text, baseline_emb)

        # Same model should easily meet 1e-2 tolerance
        assert abs(1.0 - similarity) < 1e-2

    @pytest.mark.regression
    def test_vllm_tolerance_mteb(self, validator):
        """Test that similarity meets vLLM MTEB tolerance (5e-4)."""
        text = "vLLM MTEB tolerance test."

        baseline_emb = validator.baseline_model.encode(text)
        similarity = validator.validate_against_baseline(text, baseline_emb)

        # Same model should easily meet 5e-4 tolerance
        assert abs(1.0 - similarity) < 5e-4
