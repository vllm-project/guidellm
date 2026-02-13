"""
MTEB (Massive Text Embedding Benchmark) integration for embeddings quality evaluation.

Provides standardized benchmark evaluation using MTEB tasks like STS (Semantic Textual
Similarity) to measure embedding quality across multiple standardized datasets. Follows
vLLM patterns for MTEB evaluation with configurable task selection and lightweight
defaults suitable for CI/CD environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "MTEBValidator",
    "DEFAULT_MTEB_TASKS",
]

DEFAULT_MTEB_TASKS = ["STS12", "STS13", "STSBenchmark"]
"""Default MTEB tasks for lightweight evaluation (Semantic Textual Similarity)."""


class MTEBValidator:
    """
    MTEB benchmark integration for standardized quality evaluation.

    Runs MTEB evaluation tasks on embedding models to produce standardized quality
    scores. Supports configurable task selection with defaults focused on lightweight
    STS (Semantic Textual Similarity) tasks suitable for regular benchmarking.

    Example:
    ::
        validator = MTEBValidator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task_names=["STS12", "STS13"]
        )

        results = validator.run_evaluation()
        print(f"MTEB Main Score: {results['mteb_main_score']:.4f}")
        for task, score in results['mteb_task_scores'].items():
            print(f"{task}: {score:.4f}")
    """

    def __init__(
        self,
        model_name: str,
        task_names: list[str] | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize MTEB validator with model and task configuration.

        :param model_name: HuggingFace model name or path for evaluation
        :param task_names: List of MTEB tasks to evaluate (uses DEFAULT_MTEB_TASKS if None)
        :param device: Device for model inference ("cpu", "cuda", "mps", or None for auto)
        :param batch_size: Batch size for encoding during evaluation
        :raises ImportError: If mteb or sentence-transformers is not installed
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for MTEB evaluation. "
                "Install with: pip install sentence-transformers"
            ) from e

        try:
            import mteb
        except ImportError as e:
            raise ImportError(
                "mteb is required for MTEB evaluation. "
                "Install with: pip install mteb"
            ) from e

        self.model_name = model_name
        self.task_names = task_names if task_names is not None else DEFAULT_MTEB_TASKS
        self.device = device
        self.batch_size = batch_size

        # Load model
        self.model = SentenceTransformer(model_name, device=device)

        # Store mteb module reference
        self.mteb = mteb

    def run_evaluation(
        self,
        output_folder: str | None = None,
        verbosity: int = 1,
    ) -> dict[str, Any]:
        """
        Run MTEB evaluation on configured tasks.

        Executes MTEB benchmark tasks and computes standardized quality scores.
        Returns both individual task scores and an aggregated main score.

        :param output_folder: Optional folder to save detailed results
        :param verbosity: Verbosity level (0=silent, 1=progress, 2=detailed)
        :return: Dictionary with 'mteb_main_score' and 'mteb_task_scores'

        Example:
        ::
            results = validator.run_evaluation()

            # Access main score (average across tasks)
            main_score = results['mteb_main_score']

            # Access individual task scores
            for task, score in results['mteb_task_scores'].items():
                print(f"{task}: {score:.4f}")
        """
        # Get MTEB task objects
        tasks = self.mteb.get_tasks(tasks=self.task_names)

        # Create MTEB evaluation object
        evaluation = self.mteb.MTEB(tasks=tasks)

        # Run evaluation
        results = evaluation.run(
            self.model,
            output_folder=output_folder,
            verbosity=verbosity,
            encode_kwargs={"batch_size": self.batch_size},
        )

        # Extract scores from results
        task_scores = {}
        for task_name in self.task_names:
            if task_name in results:
                # MTEB results structure varies by task type
                # Try to extract main_score or test score
                task_result = results[task_name]

                if isinstance(task_result, dict):
                    # Look for main_score in various possible locations
                    if "main_score" in task_result:
                        task_scores[task_name] = float(task_result["main_score"])
                    elif "test" in task_result and isinstance(task_result["test"], dict):
                        # Some tasks have test split with scores
                        test_result = task_result["test"]
                        if "main_score" in test_result:
                            task_scores[task_name] = float(test_result["main_score"])
                        elif "cosine_spearman" in test_result:
                            # STS tasks use cosine_spearman as primary metric
                            task_scores[task_name] = float(test_result["cosine_spearman"])
                    elif "scores" in task_result:
                        # Fallback to scores field
                        scores = task_result["scores"]
                        if isinstance(scores, list) and scores:
                            task_scores[task_name] = float(np.mean(scores))
                        elif isinstance(scores, (int, float)):
                            task_scores[task_name] = float(scores)

        # Compute main score as average across tasks
        if task_scores:
            main_score = float(np.mean(list(task_scores.values())))
        else:
            main_score = 0.0

        return {
            "mteb_main_score": main_score,
            "mteb_task_scores": task_scores,
        }

    def get_available_tasks(self) -> list[str]:
        """
        Get list of all available MTEB tasks.

        :return: List of available task names

        Example:
        ::
            validator = MTEBValidator(model_name="...")
            tasks = validator.get_available_tasks()
            print(f"Available tasks: {tasks}")
        """
        all_tasks = self.mteb.get_tasks()
        return [task.metadata.name for task in all_tasks]

    def get_task_info(self, task_name: str) -> dict[str, Any]:
        """
        Get metadata information about a specific MTEB task.

        :param task_name: Name of the MTEB task
        :return: Dictionary with task metadata
        :raises ValueError: If task is not found

        Example:
        ::
            info = validator.get_task_info("STS12")
            print(f"Task: {info['name']}")
            print(f"Description: {info['description']}")
        """
        tasks = self.mteb.get_tasks(tasks=[task_name])

        if not tasks:
            raise ValueError(f"MTEB task '{task_name}' not found")

        task = tasks[0]
        metadata = task.metadata

        return {
            "name": metadata.name,
            "description": getattr(metadata, "description", ""),
            "type": getattr(metadata, "type", ""),
            "category": getattr(metadata, "category", ""),
            "eval_splits": getattr(metadata, "eval_splits", []),
            "main_score": getattr(metadata, "main_score", ""),
        }

    @staticmethod
    def get_recommended_tasks(category: str = "sts") -> list[str]:
        """
        Get recommended MTEB tasks for specific evaluation categories.

        :param category: Evaluation category ("sts", "classification", "retrieval", etc.)
        :return: List of recommended task names

        Example:
        ::
            sts_tasks = MTEBValidator.get_recommended_tasks("sts")
            # Returns: ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark"]
        """
        recommendations = {
            "sts": [
                "STS12",
                "STS13",
                "STS14",
                "STS15",
                "STS16",
                "STSBenchmark",
                "SICKRelatedness",
            ],
            "classification": [
                "AmazonCounterfactualClassification",
                "AmazonPolarityClassification",
                "AmazonReviewsClassification",
                "Banking77Classification",
                "EmotionClassification",
            ],
            "clustering": [
                "ArxivClusteringP2P",
                "ArxivClusteringS2S",
                "BiorxivClusteringP2P",
                "BiorxivClusteringS2S",
                "MedrxivClusteringP2P",
            ],
            "retrieval": [
                "ArguAna",
                "ClimateFEVER",
                "CQADupstackRetrieval",
                "DBPedia",
                "FEVER",
            ],
            "lightweight": DEFAULT_MTEB_TASKS,  # Fastest tasks for CI/CD
        }

        return recommendations.get(category.lower(), DEFAULT_MTEB_TASKS)
