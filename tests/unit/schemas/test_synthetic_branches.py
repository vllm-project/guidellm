"""
Unit tests for synthetic sub-agent branch graph building.
"""

from __future__ import annotations

import pytest

from guidellm.scheduler.dag import DAGExecutionState
from guidellm.scheduler.schemas import HistoryContext
from guidellm.schemas import GenerationRequest
from guidellm.schemas.conversation_graph import GenerativeConversationGraph


def _make_request(label: str) -> GenerationRequest:
    return GenerationRequest(columns={"text_column": [label]})


def _branch_factory(b_idx: int, t_idx: int) -> GenerationRequest:
    return _make_request(f"branch_{b_idx}_turn_{t_idx}")


class TestFromLinearChainWithBranches:
    """Test ConversationGraph construction with sub-agent branches.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_single_branch(self):
        """
        A single branch spawns and merges back.

        ## WRITTEN BY AI ##
        """
        main = [_make_request(f"main_{i}") for i in range(4)]
        graph = GenerativeConversationGraph.from_linear_chain_with_branches(
            main_requests=main,
            branches=[{"at_turn": 1, "turns": 2}],
            branch_request_factory=_branch_factory,
        )

        assert len(graph.nodes) == 6  # 4 main + 2 branch
        assert "branch_0_0" in graph.nodes
        assert "branch_0_1" in graph.nodes

        # Branch nodes have "worker" agent_id by default
        assert graph.nodes["branch_0_0"].agent_id == "worker"

        # Find edges from/to branch
        edge_map = {
            (e.source_node_id, e.target_node_id): e.history_context
            for e in graph.edges
        }
        # Spawn: main_1 -> branch_0_0 (new)
        assert edge_map[("main_1", "branch_0_0")] == HistoryContext.NEW.value
        # Branch chain: branch_0_0 -> branch_0_1 (full)
        assert edge_map[("branch_0_0", "branch_0_1")] == HistoryContext.FULL.value
        # Merge: branch_0_1 -> main_2 (last)
        assert edge_map[("branch_0_1", "main_2")] == HistoryContext.LAST.value

    @pytest.mark.sanity
    def test_multiple_branches_same_turn(self):
        """
        Multiple branches at the same turn with different lengths.

        ## WRITTEN BY AI ##
        """
        main = [_make_request(f"main_{i}") for i in range(5)]
        graph = GenerativeConversationGraph.from_linear_chain_with_branches(
            main_requests=main,
            branches=[
                {"at_turn": 2, "turns": 3, "agent_id": "researcher"},
                {"at_turn": 2, "turns": 1, "agent_id": "reviewer"},
            ],
            branch_request_factory=_branch_factory,
        )

        # 5 main + 3 branch_0 + 1 branch_1 = 9 nodes
        assert len(graph.nodes) == 9

        # Agent IDs
        assert graph.nodes["branch_0_0"].agent_id == "researcher"
        assert graph.nodes["branch_1_0"].agent_id == "reviewer"

        # Both merge at main_3
        edge_targets = {
            (e.source_node_id, e.target_node_id)
            for e in graph.edges
            if e.history_context == HistoryContext.LAST.value
        }
        assert ("branch_0_2", "main_3") in edge_targets
        assert ("branch_1_0", "main_3") in edge_targets

    @pytest.mark.sanity
    def test_branch_history_assembly(self):
        """
        Verify walk-back produces correct history at merge point:
        full main chain + last from each branch.

        ## WRITTEN BY AI ##
        """
        main = [_make_request(f"main_{i}") for i in range(4)]
        graph = GenerativeConversationGraph.from_linear_chain_with_branches(
            main_requests=main,
            branches=[
                {"at_turn": 1, "turns": 2},
                {"at_turn": 1, "turns": 1},
            ],
            branch_request_factory=_branch_factory,
        )

        state = DAGExecutionState(graph)

        # Execute all nodes before main_2 (the merge point)
        state.mark_completed("main_0", "r_m0", "resp_m0")
        state.mark_completed("main_1", "r_m1", "resp_m1")
        state.mark_completed("branch_0_0", "r_b0_0", "resp_b0_0")
        state.mark_completed("branch_0_1", "r_b0_1", "resp_b0_1")
        state.mark_completed("branch_1_0", "r_b1_0", "resp_b1_0")

        hist = state.assemble_history("main_2")
        req_ids = [h[0] for h in hist]

        # Full chain: main_0, main_1
        assert "r_m0" in req_ids
        assert "r_m1" in req_ids
        # Last from branches (final turn only)
        assert "r_b0_1" in req_ids
        assert "r_b1_0" in req_ids
        # Branch internals NOT in history
        assert "r_b0_0" not in req_ids

    @pytest.mark.smoke
    def test_branch_nodes_have_no_history(self):
        """
        Branch nodes (spawned via new edge) should have no history.

        ## WRITTEN BY AI ##
        """
        main = [_make_request(f"main_{i}") for i in range(3)]
        graph = GenerativeConversationGraph.from_linear_chain_with_branches(
            main_requests=main,
            branches=[{"at_turn": 0, "turns": 1}],
            branch_request_factory=_branch_factory,
        )

        state = DAGExecutionState(graph)
        state.mark_completed("main_0", "r_m0", "resp_m0")

        assert state.assemble_history("branch_0_0") is None

    @pytest.mark.sanity
    def test_branches_at_different_turns(self):
        """
        Branches at different turns in the same graph.

        ## WRITTEN BY AI ##
        """
        main = [_make_request(f"main_{i}") for i in range(6)]
        graph = GenerativeConversationGraph.from_linear_chain_with_branches(
            main_requests=main,
            branches=[
                {"at_turn": 1, "turns": 1},
                {"at_turn": 3, "turns": 2},
            ],
            branch_request_factory=_branch_factory,
        )

        # 6 main + 1 + 2 = 9 nodes
        assert len(graph.nodes) == 9

        # Branch 0 merges at main_2
        edge_map = {
            (e.source_node_id, e.target_node_id): e.history_context
            for e in graph.edges
        }
        assert edge_map[("branch_0_0", "main_2")] == HistoryContext.LAST.value
        assert edge_map[("branch_1_1", "main_4")] == HistoryContext.LAST.value

    @pytest.mark.sanity
    def test_no_branches_produces_linear_graph(self):
        """
        With no branches, should produce the same result as from_linear_chain.

        ## WRITTEN BY AI ##
        """
        main = [_make_request(f"main_{i}") for i in range(3)]
        graph = GenerativeConversationGraph.from_linear_chain_with_branches(
            main_requests=main,
            branches=[],
            branch_request_factory=_branch_factory,
        )

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.root_node_ids == ["main_0"]

    @pytest.mark.smoke
    def test_empty_main_raises(self):
        """
        Empty main request list should raise ValueError.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="empty"):
            GenerativeConversationGraph.from_linear_chain_with_branches(
                main_requests=[],
                branches=[],
                branch_request_factory=_branch_factory,
            )


class TestBranchSpecValidation:
    """Test BranchSpec validation on SyntheticTextDataArgs.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_valid_branch_spec(self):
        """
        Valid branch spec should pass validation.

        ## WRITTEN BY AI ##
        """
        from guidellm.data.deserializers.synthetic import (
            BranchSpec,
            SyntheticTextDataArgs,
        )

        args = SyntheticTextDataArgs(
            kind="synthetic_text",
            prompt_tokens=100,
            turns=5,
            branches=[BranchSpec(at_turn=2, turns=3)],
        )
        assert len(args.branches) == 1
        assert args.branches[0].at_turn == 2

    @pytest.mark.sanity
    def test_branch_at_last_turn_fails(self):
        """
        Branch at the last turn should fail (no merge point).

        ## WRITTEN BY AI ##
        """
        from guidellm.data.deserializers.synthetic import (
            BranchSpec,
            SyntheticTextDataArgs,
        )

        with pytest.raises(Exception, match="at_turn"):
            SyntheticTextDataArgs(
                kind="synthetic_text",
                prompt_tokens=100,
                turns=3,
                branches=[BranchSpec(at_turn=2, turns=1)],
            )

    @pytest.mark.sanity
    def test_branch_json_parsing(self):
        """
        Branch specs should parse from JSON strings (CLI support).

        ## WRITTEN BY AI ##
        """
        from guidellm.data.deserializers.synthetic import SyntheticTextDataArgs

        args = SyntheticTextDataArgs(
            kind="synthetic_text",
            prompt_tokens=100,
            turns=5,
            branches='[{"at_turn": 1, "turns": 2, "agent_id": "researcher"}]',
        )
        assert len(args.branches) == 1
        assert args.branches[0].agent_id == "researcher"

    @pytest.mark.sanity
    def test_single_turn_cannot_have_branches(self):
        """
        A single-turn conversation cannot have branches (no merge point).

        ## WRITTEN BY AI ##
        """
        from guidellm.data.deserializers.synthetic import (
            BranchSpec,
            SyntheticTextDataArgs,
        )

        with pytest.raises(ValueError, match="at_turn"):
            SyntheticTextDataArgs(
                kind="synthetic_text",
                prompt_tokens=100,
                turns=1,
                branches=[BranchSpec(at_turn=0, turns=1)],
            )
