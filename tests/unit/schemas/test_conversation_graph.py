"""
Unit tests for conversation graph schemas and validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.scheduler.schemas import (
    ConversationEdge,
    ConversationGraph,
    ConversationNode,
)
from guidellm.schemas import GenerationRequest, RequestInfo, RequestSettings
from guidellm.schemas.conversation_graph import (
    GenerativeConversationGraph,
    GenerativeConversationNode,
)


class TestConversationGraphValidation:
    """Test ConversationGraph Pydantic validation.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_valid_linear_graph(self):
        """
        A simple linear chain should pass validation and compute
        correct root_node_ids.

        ## WRITTEN BY AI ##
        """
        g = ConversationGraph(
            graph_id="test",
            nodes={
                "a": ConversationNode(node_id="a", agent_id="x", request="r1"),
                "b": ConversationNode(node_id="b", agent_id="x", request="r2"),
            },
            edges=[
                ConversationEdge(source_node_id="a", target_node_id="b"),
            ],
        )
        assert g.root_node_ids == ["a"]

    @pytest.mark.smoke
    def test_single_node_graph(self):
        """
        A graph with one node and no edges should be valid.

        ## WRITTEN BY AI ##
        """
        g = ConversationGraph(
            graph_id="single",
            nodes={
                "only": ConversationNode(node_id="only", agent_id="x", request="r"),
            },
            edges=[],
        )
        assert g.root_node_ids == ["only"]

    @pytest.mark.sanity
    def test_cycle_detection(self):
        """
        A graph with a cycle should fail validation.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match="cycle"):
            ConversationGraph(
                graph_id="cycle",
                nodes={
                    "a": ConversationNode(node_id="a", agent_id="x", request="r"),
                    "b": ConversationNode(node_id="b", agent_id="x", request="r"),
                },
                edges=[
                    ConversationEdge(source_node_id="a", target_node_id="b"),
                    ConversationEdge(source_node_id="b", target_node_id="a"),
                ],
            )

    @pytest.mark.sanity
    def test_missing_source_node(self):
        """
        An edge referencing a nonexistent source node should fail.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match="source.*not in nodes"):
            ConversationGraph(
                graph_id="bad",
                nodes={
                    "a": ConversationNode(node_id="a", agent_id="x", request="r"),
                },
                edges=[
                    ConversationEdge(source_node_id="missing", target_node_id="a"),
                ],
            )

    @pytest.mark.sanity
    def test_missing_target_node(self):
        """
        An edge referencing a nonexistent target node should fail.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match="target.*not in nodes"):
            ConversationGraph(
                graph_id="bad",
                nodes={
                    "a": ConversationNode(node_id="a", agent_id="x", request="r"),
                },
                edges=[
                    ConversationEdge(source_node_id="a", target_node_id="missing"),
                ],
            )

    @pytest.mark.sanity
    def test_multiple_full_parents_rejected(self):
        """
        A node with more than one full incoming edge should fail.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match="full.*incoming edges"):
            ConversationGraph(
                graph_id="multi_full",
                nodes={
                    "a": ConversationNode(node_id="a", agent_id="x", request="r"),
                    "b": ConversationNode(node_id="b", agent_id="x", request="r"),
                    "c": ConversationNode(node_id="c", agent_id="x", request="r"),
                },
                edges=[
                    ConversationEdge(
                        source_node_id="a",
                        target_node_id="c",
                        history_context="full",
                    ),
                    ConversationEdge(
                        source_node_id="b",
                        target_node_id="c",
                        history_context="full",
                    ),
                ],
            )

    @pytest.mark.sanity
    def test_root_node_ids_derived_correctly(self):
        """
        root_node_ids should be derived from the graph structure,
        overwriting any user-provided values.

        ## WRITTEN BY AI ##
        """
        g = ConversationGraph(
            graph_id="roots",
            nodes={
                "r1": ConversationNode(node_id="r1", agent_id="x", request="r"),
                "r2": ConversationNode(node_id="r2", agent_id="x", request="r"),
                "child": ConversationNode(node_id="child", agent_id="x", request="r"),
            },
            edges=[
                ConversationEdge(
                    source_node_id="r1",
                    target_node_id="child",
                    history_context="last",
                ),
                ConversationEdge(
                    source_node_id="r2",
                    target_node_id="child",
                    history_context="last",
                ),
            ],
            root_node_ids=["wrong"],
        )
        assert sorted(g.root_node_ids) == ["r1", "r2"]

    @pytest.mark.sanity
    def test_mixed_history_context_types(self):
        """
        A graph with all three history_context types should be valid.

        ## WRITTEN BY AI ##
        """
        g = ConversationGraph(
            graph_id="mixed",
            nodes={
                "orch": ConversationNode(node_id="orch", agent_id="o", request="plan"),
                "w1": ConversationNode(node_id="w1", agent_id="w", request="task"),
                "agg": ConversationNode(node_id="agg", agent_id="o", request="combine"),
            },
            edges=[
                ConversationEdge(
                    source_node_id="orch",
                    target_node_id="w1",
                    history_context="new",
                ),
                ConversationEdge(
                    source_node_id="w1",
                    target_node_id="agg",
                    history_context="last",
                ),
                ConversationEdge(
                    source_node_id="orch",
                    target_node_id="agg",
                    history_context="full",
                ),
            ],
        )
        assert g.root_node_ids == ["orch"]


class TestGenerativeConversationGraph:
    """Test concrete generative graph schemas.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_from_linear_chain(self):
        """
        from_linear_chain should create a valid degenerate graph
        with full edges connecting sequential turns.

        ## WRITTEN BY AI ##
        """
        reqs = [
            GenerationRequest(columns={"text_column": [f"turn {i}"]}) for i in range(3)
        ]
        graph = GenerativeConversationGraph.from_linear_chain(reqs)

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.root_node_ids == ["turn_0"]

        for edge in graph.edges:
            assert edge.history_context == "full"

    @pytest.mark.smoke
    def test_from_linear_chain_single_request(self):
        """
        A single request should produce a graph with one node and no edges.

        ## WRITTEN BY AI ##
        """
        req = GenerationRequest(columns={"text_column": ["hello"]})
        graph = GenerativeConversationGraph.from_linear_chain([req])

        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0
        assert graph.root_node_ids == ["turn_0"]

    @pytest.mark.sanity
    def test_from_linear_chain_empty_raises(self):
        """
        An empty request list should raise ValueError.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="empty"):
            GenerativeConversationGraph.from_linear_chain([])

    @pytest.mark.sanity
    def test_from_linear_chain_inherits_settings(self):
        """
        Node settings should be populated from the request's settings.

        ## WRITTEN BY AI ##
        """
        req = GenerationRequest(
            columns={"text_column": ["hello"]},
            settings=RequestSettings(relative_timestamp=1.5),
        )
        graph = GenerativeConversationGraph.from_linear_chain([req])
        node = graph.nodes["turn_0"]
        assert node.settings.relative_timestamp == 1.5

    @pytest.mark.smoke
    def test_is_conversation_graph_subclass(self):
        """
        GenerativeConversationGraph should be a subclass of ConversationGraph.

        ## WRITTEN BY AI ##
        """
        reqs = [GenerationRequest(columns={"text_column": ["hello"]})]
        graph = GenerativeConversationGraph.from_linear_chain(reqs)
        assert isinstance(graph, ConversationGraph)

    @pytest.mark.smoke
    def test_generative_node_is_conversation_node_subclass(self):
        """
        GenerativeConversationNode should be a subclass of ConversationNode.

        ## WRITTEN BY AI ##
        """
        node = GenerativeConversationNode(
            node_id="test",
            agent_id="agent",
            request=GenerationRequest(columns={"text_column": ["hello"]}),
        )
        assert isinstance(node, ConversationNode)


class TestConversationGraphSubgraphForNodes:
    """Test ConversationGraph.subgraph_for_nodes truncation helper.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def branched_graph(self) -> GenerativeConversationGraph:
        """Build a main chain with one branch for truncation tests."""
        main_reqs = [
            GenerationRequest(columns={"text_column": [f"main {i}"]}) for i in range(3)
        ]

        def branch_factory(branch_index: int, turn_index: int) -> GenerationRequest:
            return GenerationRequest(
                columns={"text_column": [f"branch {branch_index} turn {turn_index}"]}
            )

        return GenerativeConversationGraph.from_linear_chain_with_branches(
            main_requests=main_reqs,
            branches=[{"at_turn": 0, "turns": 1, "agent_id": "worker"}],
            branch_request_factory=branch_factory,
        )

    @pytest.mark.smoke
    def test_subgraph_prefix_main_0_only(self, branched_graph):
        """
        Truncating to the root keeps only main_0 and drops later nodes.

        ## WRITTEN BY AI ##
        """
        branched_graph.request_infos["main_0"] = RequestInfo(
            request_id="r0",
            conversation_id=branched_graph.graph_id,
            graph_id=branched_graph.graph_id,
            node_id="main_0",
            status="queued",
        )
        sub = branched_graph.subgraph_for_nodes({"main_0"})

        assert set(sub.nodes) == {"main_0"}
        assert set(sub.request_infos) == {"main_0"}
        assert sub.edges == []
        assert sub.root_node_ids == ["main_0"]
        assert sub.graph_id == branched_graph.graph_id
        assert "main_1" not in sub.nodes

    @pytest.mark.sanity
    def test_subgraph_prefix_main_0_and_branch(self, branched_graph):
        """
        Truncating to main_0 and its spawn keeps the NEW edge and drops merge.

        ## WRITTEN BY AI ##
        """
        keep = {"main_0", "branch_0_0"}
        for nid in keep:
            branched_graph.request_infos[nid] = RequestInfo(
                request_id=nid,
                conversation_id=branched_graph.graph_id,
                graph_id=branched_graph.graph_id,
                node_id=nid,
                status="queued",
            )
        sub = branched_graph.subgraph_for_nodes(keep)

        assert set(sub.nodes) == keep
        assert set(sub.request_infos) == keep
        assert sub.root_node_ids == ["main_0"]
        assert "main_1" not in sub.nodes
        assert "main_2" not in sub.nodes

        edge_pairs = {
            (e.source_node_id, e.target_node_id, e.history_context) for e in sub.edges
        }
        assert (
            "main_0",
            "branch_0_0",
            "new",
        ) in edge_pairs
        # Merge into main_1 must be gone with main_1 removed
        assert not any(e.target_node_id == "main_1" for e in sub.edges)

    @pytest.mark.smoke
    def test_mid_stop_yield_invariant(self, branched_graph):
        """
        After truncating to queued nodes, nodes and request_infos match.

        ## WRITTEN BY AI ##
        """
        queued_ids = {"main_0"}
        branched_graph.request_infos["main_0"] = RequestInfo(
            request_id="r0",
            conversation_id=branched_graph.graph_id,
            graph_id=branched_graph.graph_id,
            node_id="main_0",
            status="queued",
        )
        # Simulate generator: partial request_infos vs full node set
        assert set(branched_graph.request_infos) != set(branched_graph.nodes)

        yielded = branched_graph.subgraph_for_nodes(queued_ids)
        assert set(yielded.nodes) == set(yielded.request_infos) == queued_ids
        assert "main_1" not in yielded.nodes

    @pytest.mark.sanity
    def test_subgraph_empty_raises(self, branched_graph):
        """
        An empty node set should raise ValueError.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="empty"):
            branched_graph.subgraph_for_nodes(set())

    @pytest.mark.sanity
    def test_subgraph_unknown_ids_raises(self, branched_graph):
        """
        Unknown node IDs should raise ValueError.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="Unknown node IDs"):
            branched_graph.subgraph_for_nodes({"main_0", "missing_node"})
