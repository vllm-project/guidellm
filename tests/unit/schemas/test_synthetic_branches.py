"""
Unit tests for synthetic sub-agent branch graph building.
"""

from __future__ import annotations

import pytest

from guidellm.data.deserializers.synthetic import BranchSpec, SyntheticTextDataArgs
from guidellm.scheduler.dag import DAGExecutionState
from guidellm.schemas import GenerationRequest, RequestSettings
from guidellm.schemas.conversation_graph import GenerativeConversationGraph


def _make_request(
    label: str, settings: RequestSettings | None = None
) -> tuple[GenerationRequest, RequestSettings]:
    return (
        GenerationRequest(columns={"text_column": [label]}),
        settings if settings is not None else RequestSettings(),
    )


def _branch_factory(
    b_idx: int, t_idx: int
) -> tuple[GenerationRequest, RequestSettings]:
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
            (e.source_node_id, e.target_node_id): e.history_context for e in graph.edges
        }
        # Spawn: main_1 -> branch_0_0 (new)
        assert edge_map[("main_1", "branch_0_0")] == "new"
        # Branch chain: branch_0_0 -> branch_0_1 (full)
        assert edge_map[("branch_0_0", "branch_0_1")] == "full"
        # Merge: branch_0_1 -> main_2 (last)
        assert edge_map[("branch_0_1", "main_2")] == "last"

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
            if e.history_context == "last"
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
            (e.source_node_id, e.target_node_id): e.history_context for e in graph.edges
        }
        assert edge_map[("branch_0_0", "main_2")] == "last"
        assert edge_map[("branch_1_1", "main_4")] == "last"

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
        with pytest.raises(Exception, match="at_turn"):
            SyntheticTextDataArgs(
                kind="synthetic_text",
                prompt_tokens=100,
                turns=3,
                branches=[BranchSpec(at_turn=2, turns=1)],
            )

    @pytest.mark.sanity
    def test_branches_with_tool_call_turns_accepted(self):
        """
        Branches may be combined with tool_call_turns on the main chain.

        ## WRITTEN BY AI ##
        """
        args = SyntheticTextDataArgs(
            kind="synthetic_text",
            prompt_tokens=100,
            turns=3,
            branches=[BranchSpec(at_turn=0, turns=1)],
            tool_call_turns=[0],
        )
        assert args.tool_call_turns == [0]
        assert len(args.branches) == 1

    @pytest.mark.sanity
    def test_branch_json_parsing(self):
        """
        Branch specs should parse from JSON strings (CLI support).

        ## WRITTEN BY AI ##
        """
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
        with pytest.raises(ValueError, match="at_turn"):
            SyntheticTextDataArgs(
                kind="synthetic_text",
                prompt_tokens=100,
                turns=1,
                branches=[BranchSpec(at_turn=0, turns=1)],
            )


class TestSyntheticBranchesEmitConversationTurns:
    """Branched synthetic rows emit conversation_turns with inline parents.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_branched_row_emits_conversation_turns_topology(self):
        """
        ## WRITTEN BY AI ##
        """
        from unittest.mock import Mock

        from guidellm.data.deserializers.synthetic import _SyntheticTextExamplesIterable
        from guidellm.data.schemas.conversation_graph_data import ConversationGraphData
        from guidellm.utils.imports import json

        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: list(range(max(1, len(text) // 4)))
        tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=False: (
            " ".join(f"tok{t}" for t in tokens)
        )

        config = SyntheticTextDataArgs(
            kind="synthetic_text",
            prompt_tokens=20,
            output_tokens=10,
            turns=3,
            branches=[BranchSpec(at_turn=0, turns=1, agent_id="worker")],
        )
        iterable = _SyntheticTextExamplesIterable(config, tokenizer, random_seed=1)
        _key, row = next(iter(iterable))

        assert set(row) == {"conversation_turns"}
        graph_data = ConversationGraphData.model_validate(
            json.loads(row["conversation_turns"])
        )
        by_id = {turn.node_id: turn for turn in graph_data.turns}
        assert set(by_id) == {"main_0", "main_1", "main_2", "branch_0_0"}
        assert by_id["main_0"].parents == []
        assert by_id["branch_0_0"].agent_id == "worker"
        assert [
            (p.parent_node_id, p.history_context) for p in by_id["branch_0_0"].parents
        ] == [("main_0", "new")]
        assert {
            (p.parent_node_id, p.history_context) for p in by_id["main_1"].parents
        } == {("main_0", "full"), ("branch_0_0", "last")}
        assert [
            (p.parent_node_id, p.history_context) for p in by_id["main_2"].parents
        ] == [("main_1", "full")]

    @pytest.mark.smoke
    def test_branched_row_with_client_tool_turn_expands_injection(self):
        """
        Client tool turns on the main chain expand to tool-call + injection nodes.

        ## WRITTEN BY AI ##
        """
        from unittest.mock import Mock

        from guidellm.data.deserializers.synthetic import _SyntheticTextExamplesIterable
        from guidellm.data.finalizers.generative import (
            GenerativeRequestFinalizer,
            GenerativeRequestFinalizerArgs,
        )
        from guidellm.data.schemas.conversation_graph_data import ConversationGraphData
        from guidellm.utils.imports import json

        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: list(range(max(1, len(text) // 4)))
        tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=False: (
            " ".join(f"tok{t}" for t in tokens)
        )

        config = SyntheticTextDataArgs(
            kind="synthetic_text",
            prompt_tokens=20,
            output_tokens=10,
            turns=3,
            branches=[BranchSpec(at_turn=0, turns=1, agent_id="worker")],
            tool_call_turns=[0],
        )
        iterable = _SyntheticTextExamplesIterable(config, tokenizer, random_seed=1)
        _key, row = next(iter(iterable))
        graph_data = ConversationGraphData.model_validate(
            json.loads(row["conversation_turns"])
        )
        by_id = {turn.node_id: turn for turn in graph_data.turns}

        assert set(by_id) == {
            "main_0",
            "main_0_injection",
            "main_1",
            "main_2",
            "branch_0_0",
        }
        assert "tools_column" in by_id["main_0"].columns
        assert "output_tokens_count_column" not in by_id["main_0"].columns
        assert by_id["main_0_injection"].columns["turn_type_column"] == [
            "tool_response_injection"
        ]
        assert [
            (p.parent_node_id, p.history_context)
            for p in by_id["main_0_injection"].parents
        ] == [("main_0", "full")]
        assert [
            (p.parent_node_id, p.history_context) for p in by_id["branch_0_0"].parents
        ] == [("main_0_injection", "new")]
        assert {
            (p.parent_node_id, p.history_context) for p in by_id["main_1"].parents
        } == {("main_0_injection", "full"), ("branch_0_0", "last")}

        graph = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())(
            [{"conversation_turns_column": [graph_data.model_dump(mode="json")]}]
        )
        assert graph.nodes["main_0"].request.turn_type == "client_tool_call"
        assert (
            graph.nodes["main_0_injection"].request.turn_type
            == "tool_response_injection"
        )
        edge_triples = {
            (e.source_node_id, e.target_node_id, e.history_context) for e in graph.edges
        }
        assert ("main_0", "main_0_injection", "full") in edge_triples
        assert ("main_0_injection", "branch_0_0", "new") in edge_triples
        assert ("main_0_injection", "main_1", "full") in edge_triples
        assert ("branch_0_0", "main_1", "last") in edge_triples

    @pytest.mark.sanity
    def test_branched_row_with_server_tool_turn_no_injection(self):
        """
        Server tool turns stay a single main node with turn_type set.

        ## WRITTEN BY AI ##
        """
        from unittest.mock import Mock

        from guidellm.data.deserializers.synthetic import _SyntheticTextExamplesIterable
        from guidellm.data.schemas.conversation_graph_data import ConversationGraphData
        from guidellm.utils.imports import json

        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: list(range(max(1, len(text) // 4)))
        tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=False: (
            " ".join(f"tok{t}" for t in tokens)
        )

        config = SyntheticTextDataArgs(
            kind="synthetic_text",
            prompt_tokens=20,
            output_tokens=10,
            turns=3,
            branches=[BranchSpec(at_turn=0, turns=1)],
            server_tool_call_turns=[0],
        )
        iterable = _SyntheticTextExamplesIterable(config, tokenizer, random_seed=1)
        _key, row = next(iter(iterable))
        graph_data = ConversationGraphData.model_validate(
            json.loads(row["conversation_turns"])
        )
        by_id = {turn.node_id: turn for turn in graph_data.turns}

        assert "main_0_injection" not in by_id
        assert by_id["main_0"].columns["turn_type_column"] == ["server_tool_call"]
        assert [
            (p.parent_node_id, p.history_context) for p in by_id["branch_0_0"].parents
        ] == [("main_0", "new")]
