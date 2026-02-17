"""
Quality validation for embeddings benchmarks.

Provides tools for validating embedding quality through cosine similarity
comparison against baseline models. Supports HuggingFace SentenceTransformers
models as baselines and implements tolerance-based validation following vLLM
patterns (1e-2 standard, 5e-4 MTEB).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "EmbeddingsQualityValidator",
    "compute_cosine_similarity",
]


def compute_cosine_similarity(
    emb1: NDArray[np.float32] | list[float],
    emb2: NDArray[np.float32] | list[float],
) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical direction). For normalized
    embeddings, this is equivalent to the dot product.

    Formula: cos_sim = dot(emb1, emb2) / (||emb1|| * ||emb2||)

    :param emb1: First embedding vector (numpy array or list)
    :param emb2: Second embedding vector (numpy array or list)
    :return: Cosine similarity score between -1.0 and 1.0
    :raises ValueError: If embeddings have different dimensions or are empty

    Example:
    ::
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])
        similarity = compute_cosine_similarity(emb1, emb2)  # Returns 1.0

        emb3 = np.array([0.0, 1.0, 0.0])
        similarity = compute_cosine_similarity(emb1, emb3)  # Returns 0.0
    """
    # Convert to numpy arrays if needed
    vec1 = np.array(emb1, dtype=np.float32)
    vec2 = np.array(emb2, dtype=np.float32)

    # Validate dimensions
    if vec1.shape != vec2.shape:
        raise ValueError(
            f"Embedding dimensions must match: {vec1.shape} vs {vec2.shape}"
        )

    if vec1.size == 0:
        raise ValueError("Embeddings cannot be empty")

    # Compute norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle zero vectors
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    cosine_sim = dot_product / (norm1 * norm2)

    return float(cosine_sim)


class EmbeddingsQualityValidator:
    """
    Validates embedding quality against baseline models.

    Loads a HuggingFace SentenceTransformers model as a baseline and compares
    target embeddings against baseline outputs using cosine similarity. Supports
    configurable tolerance thresholds following vLLM patterns.

    Example:
    ::
        validator = EmbeddingsQualityValidator(
            baseline_model="sentence-transformers/all-MiniLM-L6-v2",
            tolerance=1e-2
        )

        text = "This is a test sentence"
        target_embedding = [0.1, 0.2, 0.3, ...]  # From target model

        similarity = validator.validate_against_baseline(text, target_embedding)
        is_valid = validator.check_tolerance(similarity)
    """

    def __init__(
        self,
        baseline_model: str,
        tolerance: float = 1e-2,
        device: str | None = None,
    ):
        """
        Initialize quality validator with baseline model.

        :param baseline_model: HuggingFace model name or path
            (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        :param tolerance: Cosine similarity tolerance threshold
            (1e-2 for standard, 5e-4 for MTEB-level validation)
        :param device: Device for model inference ("cpu", "cuda", "mps", or
            None for auto)
        :raises ImportError: If sentence-transformers is not installed
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for quality validation. "
                "Install with: pip install sentence-transformers"
            ) from e

        self.baseline_model_name = baseline_model
        self.tolerance = tolerance
        self.device = device

        # Load baseline model
        self.baseline_model = SentenceTransformer(baseline_model, device=device)

    def encode_baseline(
        self,
        texts: str | list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """
        Generate embeddings using the baseline model.

        :param texts: Single text or list of texts to encode
        :param normalize: Whether to normalize embeddings to unit length
        :param batch_size: Batch size for encoding
        :return: Embeddings as numpy array (shape: [n_texts, embedding_dim])
        """
        embeddings = self.baseline_model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        # Ensure return type is correct
        if isinstance(texts, str):
            return np.array(embeddings, dtype=np.float32)
        return np.array(embeddings, dtype=np.float32)

    def validate_against_baseline(
        self,
        text: str,
        target_embedding: NDArray[np.float32] | list[float],
        normalize: bool = True,
    ) -> float:
        """
        Compare target embedding against baseline model output.

        :param text: Input text that was embedded
        :param target_embedding: Embedding from target model to validate
        :param normalize: Whether to normalize embeddings before comparison
        :return: Cosine similarity score (0.0 to 1.0)

        Example:
        ::
            text = "Example sentence"
            target_emb = model.encode(text)  # From target model
            similarity = validator.validate_against_baseline(text, target_emb)
            # High similarity (>0.95) indicates good quality
        """
        # Generate baseline embedding
        baseline_embedding = self.encode_baseline(text, normalize=normalize)

        # Convert target to numpy if needed
        target_array = np.array(target_embedding, dtype=np.float32)

        # Normalize target if requested
        if normalize:
            norm = np.linalg.norm(target_array)
            if norm > 0:
                target_array = target_array / norm

        # Compute similarity
        return compute_cosine_similarity(baseline_embedding, target_array)

    def validate_batch(
        self,
        texts: list[str],
        target_embeddings: NDArray[np.float32] | list[list[float]],
        normalize: bool = True,
    ) -> list[float]:
        """
        Validate multiple embeddings against baseline model.

        :param texts: List of input texts
        :param target_embeddings: Embeddings from target model (shape: [n, dim])
        :param normalize: Whether to normalize embeddings before comparison
        :return: List of cosine similarity scores

        Example:
        ::
            texts = ["Text 1", "Text 2", "Text 3"]
            target_embs = model.encode(texts)
            similarities = validator.validate_batch(texts, target_embs)
            mean_similarity = np.mean(similarities)
        """
        # Generate baseline embeddings for all texts
        baseline_embeddings = self.encode_baseline(texts, normalize=normalize)

        # Convert target to numpy if needed
        target_array = np.array(target_embeddings, dtype=np.float32)

        # Normalize targets if requested
        if normalize:
            norms = np.linalg.norm(target_array, axis=1, keepdims=True)
            target_array = np.where(norms > 0, target_array / norms, target_array)

        # Compute similarities
        similarities = []
        for baseline_emb, target_emb in zip(
            baseline_embeddings, target_array, strict=False
        ):
            sim = compute_cosine_similarity(baseline_emb, target_emb)
            similarities.append(sim)

        return similarities

    def check_tolerance(self, similarity: float) -> bool:
        """
        Check if similarity meets tolerance threshold.

        :param similarity: Cosine similarity score to validate
        :return: True if similarity is within tolerance (similarity >= 1.0 - tolerance)

        Example:
        ::
            # With tolerance=1e-2 (0.01)
            validator.check_tolerance(0.99)   # True (within 1% of perfect)
            validator.check_tolerance(0.985)  # False (outside tolerance)
        """
        return similarity >= (1.0 - self.tolerance)

    def check_self_consistency(
        self,
        _text: str,
        embeddings: list[NDArray[np.float32] | list[float]],
        tolerance: float | None = None,
    ) -> tuple[float, bool]:
        """
        Verify that same input produces consistent embeddings.

        Self-consistency check ensures the model produces identical (or nearly
        identical) embeddings for the same input text across multiple inferences.

        :param text: Input text (same for all embeddings)
        :param embeddings: List of embeddings from repeated encodings of the same text
        :param tolerance: Optional tolerance override (uses instance tolerance if None)
        :return: Tuple of (mean_similarity, is_consistent)

        Example:
        ::
            text = "Consistency test"
            embeddings = [model.encode(text) for _ in range(5)]
            mean_sim, is_consistent = validator.check_self_consistency(text, embeddings)
            # Should be near 1.0 for deterministic models
        """
        if len(embeddings) < 2:  # noqa: PLR2004
            # Need at least 2 embeddings to compare
            return 1.0, True

        tolerance_threshold = tolerance if tolerance is not None else self.tolerance

        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        # Compute mean similarity
        mean_similarity = float(np.mean(similarities))

        # Check if all comparisons meet tolerance
        is_consistent = mean_similarity >= (1.0 - tolerance_threshold)

        return mean_similarity, is_consistent

    def get_embedding_stats(
        self, embeddings: NDArray[np.float32] | list[list[float]]
    ) -> dict[str, float]:
        """
        Compute statistical properties of embeddings.

        :param embeddings: Embeddings array (shape: [n, dim])
        :return: Dictionary with statistics (mean_norm, std_norm, mean_value, std_value)

        Example:
        ::
            embeddings = model.encode(texts)
            stats = validator.get_embedding_stats(embeddings)
            print(f"Mean norm: {stats['mean_norm']:.4f}")
        """
        emb_array = np.array(embeddings, dtype=np.float32)

        # Compute norms
        norms = np.linalg.norm(emb_array, axis=1)

        # Compute value statistics
        mean_value = float(np.mean(emb_array))
        std_value = float(np.std(emb_array))

        return {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "mean_value": mean_value,
            "std_value": std_value,
            "min_value": float(np.min(emb_array)),
            "max_value": float(np.max(emb_array)),
        }
