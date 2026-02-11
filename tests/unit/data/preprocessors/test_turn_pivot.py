"""
Unit tests for the TurnPivot preprocessor.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.data.preprocessors import TurnPivot
from guidellm.data.preprocessors.preprocessor import PreprocessorRegistry


class TestTurnPivot:
    """Test suite for TurnPivot preprocessor."""

    @pytest.mark.smoke
    def test_class_registration(self):
        """
        Test that TurnPivot is properly registered in the PreprocessorRegistry.

        ### WRITTEN BY AI ###
        """
        assert "turn_pivot" in PreprocessorRegistry.registry
        assert PreprocessorRegistry.registry["turn_pivot"] is TurnPivot

    @pytest.mark.smoke
    def test_initialization(self):
        """
        Test TurnPivot initialization with no parameters.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()
        assert isinstance(preprocessor, TurnPivot)

    @pytest.mark.smoke
    def test_basic_transpose(self):
        """
        Test basic transpose operation with 2 turns and 2 batches.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        # Input: 2 turns, each with 2 batches
        turns = [
            {"prompt": ["P1a", "P1b"], "output_tokens_count": [11, 12]},  # Turn 1
            {"prompt": ["P2a", "P2b"], "output_tokens_count": [21, 22]},  # Turn 2
        ]

        result = preprocessor(turns)

        # Output: 2 batches, each with 2 turns
        assert len(result) == 2
        assert result[0] == {"prompt": ["P1a", "P2a"], "output_tokens_count": [11, 21]}
        assert result[1] == {"prompt": ["P1b", "P2b"], "output_tokens_count": [12, 22]}

    @pytest.mark.sanity
    def test_single_turn_multiple_batches(self):
        """
        Test transpose with a single turn containing multiple batches.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"text": ["batch1", "batch2", "batch3"], "count": [1, 2, 3]},
        ]

        result = preprocessor(turns)

        assert len(result) == 3
        assert result[0] == {"text": ["batch1"], "count": [1]}
        assert result[1] == {"text": ["batch2"], "count": [2]}
        assert result[2] == {"text": ["batch3"], "count": [3]}

    @pytest.mark.sanity
    def test_multiple_turns_single_batch(self):
        """
        Test transpose with multiple turns each containing a single batch.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"question": ["Q1"], "answer": ["A1"]},
            {"question": ["Q2"], "answer": ["A2"]},
            {"question": ["Q3"], "answer": ["A3"]},
        ]

        result = preprocessor(turns)

        assert len(result) == 1
        assert result[0] == {
            "question": ["Q1", "Q2", "Q3"],
            "answer": ["A1", "A2", "A3"],
        }

    @pytest.mark.sanity
    def test_varying_batch_sizes_across_turns(self):
        """
        Test transpose with different batch sizes across turns.

        This tests the dynamic batch creation feature where later turns
        can have more batches than earlier turns.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"prompt": ["P1a", "P1b"], "tokens": [10, 20]},  # 2 batches
            {"prompt": ["P2a", "P2b", "P2c"], "tokens": [30, 40, 50]},  # 3 batches
        ]

        result = preprocessor(turns)

        assert len(result) == 3
        assert result[0] == {"prompt": ["P1a", "P2a"], "tokens": [10, 30]}
        assert result[1] == {"prompt": ["P1b", "P2b"], "tokens": [20, 40]}
        assert result[2] == {"prompt": ["P2c"], "tokens": [50]}

    @pytest.mark.sanity
    def test_columns_not_present_in_all_turns(self):
        """
        Test transpose when some columns are missing in certain turns.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"prompt": ["P1a", "P1b"], "system": ["S1a", "S1b"]},  # Has system column
            {"prompt": ["P2a", "P2b"]},  # No system column
            {"prompt": ["P3a", "P3b"], "image": ["I3a", "I3b"]},  # Has image column
        ]

        result = preprocessor(turns)

        assert len(result) == 2
        assert result[0] == {
            "prompt": ["P1a", "P2a", "P3a"],
            "system": ["S1a"],
            "image": ["I3a"],
        }
        assert result[1] == {
            "prompt": ["P1b", "P2b", "P3b"],
            "system": ["S1b"],
            "image": ["I3b"],
        }

    @pytest.mark.sanity
    def test_empty_input(self):
        """
        Test transpose with empty input list.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()
        result = preprocessor([])
        assert result == []

    @pytest.mark.sanity
    def test_single_turn_single_batch(self):
        """
        Test transpose with a single turn containing a single batch.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"question": ["What is 2+2?"], "answer": ["4"]},
        ]

        result = preprocessor(turns)

        assert len(result) == 1
        assert result[0] == {"question": ["What is 2+2?"], "answer": ["4"]}

    @pytest.mark.sanity
    def test_empty_columns_in_turn(self):
        """
        Test transpose when a turn has empty column lists.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"prompt": [], "tokens": []},  # Empty turn
            {"prompt": ["P2"], "tokens": [20]},  # Non-empty turn
        ]

        result = preprocessor(turns)

        # Should only create a batch for the second turn
        assert len(result) == 1
        assert result[0] == {"prompt": ["P2"], "tokens": [20]}

    @pytest.mark.sanity
    def test_different_column_names_per_turn(self):
        """
        Test transpose with completely different columns in each turn.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"column_a": ["A1", "A2"]},
            {"column_b": ["B1", "B2"]},
            {"column_c": ["C1", "C2"]},
        ]

        result = preprocessor(turns)

        assert len(result) == 2
        assert result[0] == {"column_a": ["A1"], "column_b": ["B1"], "column_c": ["C1"]}
        assert result[1] == {"column_a": ["A2"], "column_b": ["B2"], "column_c": ["C2"]}

    @pytest.mark.regression
    def test_large_dataset(self):
        """
        Test transpose with a large number of turns and batches.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        num_turns = 100
        num_batches = 50

        turns = [
            {
                "turn_id": [f"turn_{t}_batch_{b}" for b in range(num_batches)],
                "value": [t * num_batches + b for b in range(num_batches)],
            }
            for t in range(num_turns)
        ]

        result = preprocessor(turns)

        assert len(result) == num_batches
        for b in range(num_batches):
            assert len(result[b]["turn_id"]) == num_turns
            assert len(result[b]["value"]) == num_turns
            assert result[b]["turn_id"][0] == f"turn_0_batch_{b}"
            assert result[b]["value"][0] == b

    @pytest.mark.sanity
    def test_mixed_data_types(self):
        """
        Test transpose with various data types in column values.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {
                "strings": ["hello", "world"],
                "ints": [1, 2],
                "floats": [1.5, 2.5],
                "lists": [[1, 2], [3, 4]],
                "dicts": [{"a": 1}, {"b": 2}],
                "none_values": [None, None],
            },
            {
                "strings": ["foo", "bar"],
                "ints": [3, 4],
                "floats": [3.5, 4.5],
                "lists": [[5, 6], [7, 8]],
                "dicts": [{"c": 3}, {"d": 4}],
                "none_values": [None, None],
            },
        ]

        result = preprocessor(turns)

        assert len(result) == 2
        assert result[0]["strings"] == ["hello", "foo"]
        assert result[0]["ints"] == [1, 3]
        assert result[0]["floats"] == [1.5, 3.5]
        assert result[0]["lists"] == [[1, 2], [5, 6]]
        assert result[0]["dicts"] == [{"a": 1}, {"c": 3}]
        assert result[0]["none_values"] == [None, None]

    @pytest.mark.sanity
    def test_preserves_order(self):
        """
        Test that transpose preserves the order of turns and batches.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"sequence": ["T1B1", "T1B2", "T1B3"]},
            {"sequence": ["T2B1", "T2B2", "T2B3"]},
            {"sequence": ["T3B1", "T3B2", "T3B3"]},
        ]

        result = preprocessor(turns)

        assert len(result) == 3
        assert result[0]["sequence"] == ["T1B1", "T2B1", "T3B1"]
        assert result[1]["sequence"] == ["T1B2", "T2B2", "T3B2"]
        assert result[2]["sequence"] == ["T1B3", "T2B3", "T3B3"]

    @pytest.mark.sanity
    def test_result_is_regular_dict(self):
        """
        Test that the result contains regular dicts, not defaultdicts.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        turns = [
            {"col1": ["val1"], "col2": ["val2"]},
        ]

        result = preprocessor(turns)

        assert len(result) == 1
        assert type(result[0]) is dict
        assert not hasattr(result[0], "default_factory")

    @pytest.mark.regression
    def test_multiturn_conversation_example(self):
        """
        Test realistic multi-turn conversation scenario.

        This simulates a conversational AI dataset where each turn
        represents a user message and assistant response.

        ### WRITTEN BY AI ###
        """
        preprocessor = TurnPivot()

        # 3 conversations (batches), each with 2 turns
        turns = [
            {
                "role": ["user", "user", "user"],
                "content": [
                    "What is Python?",
                    "Explain quantum computing",
                    "How does ML work?",
                ],
                "timestamp": [1000, 2000, 3000],
            },
            {
                "role": ["assistant", "assistant", "assistant"],
                "content": [
                    "Python is a programming language",
                    "Quantum computing uses qubits",
                    "ML learns from data",
                ],
                "timestamp": [1001, 2001, 3001],
            },
        ]

        result = preprocessor(turns)

        # Should have 3 conversations, each with 2 turns
        assert len(result) == 3

        # First conversation
        assert result[0]["role"] == ["user", "assistant"]
        assert result[0]["content"] == [
            "What is Python?",
            "Python is a programming language",
        ]
        assert result[0]["timestamp"] == [1000, 1001]

        # Second conversation
        assert result[1]["role"] == ["user", "assistant"]
        assert result[1]["content"] == [
            "Explain quantum computing",
            "Quantum computing uses qubits",
        ]
        assert result[1]["timestamp"] == [2000, 2001]

        # Third conversation
        assert result[2]["role"] == ["user", "assistant"]
        assert result[2]["content"] == ["How does ML work?", "ML learns from data"]
        assert result[2]["timestamp"] == [3000, 3001]
