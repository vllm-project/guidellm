"""
Unit tests for DAG execution utilities.
"""

from __future__ import annotations

import pytest

from guidellm.scheduler.dag import DAGExecutionState
from guidellm.scheduler.schemas import (
    ConversationEdge,
    ConversationGraph,
    ConversationNode,
    HistoryContext,
)


def _make_node(node_id: str, agent_id: str = "agent") -> ConversationNode[str]:
    return ConversationNode(
        node_id=node_id,
        agent_id=agent_id,
        request=f"req_{node_id}",
    )


def _linear_graph(n: int) -> ConversationGraph[str]:
    """Build a linear chain of n nodes connected by full edges."""
    node_ids = [f"n{i}" for i in range(n)]
    nodes = {nid: _make_node(nid) for nid in node_ids}
    edges = [
        ConversationEdge(
            source_node_id=node_ids[i],
            target_node_id=node_ids[i + 1],
            history_context=HistoryContext.FULL,
        )
        for i in range(n - 1)
    ]
    return ConversationGraph(graph_id="linear", nodes=nodes, edges=edges)


def _fork_join_graph() -> ConversationGraph[str]:
    """
    Build a fork/join graph:
    M1 -full-> M2 -full-> M3 -full-> M4
                            |-new-> W1 -last-> M4
                            |-new-> W2 -last-> M4
    """
    nodes = {
        nid: _make_node(nid, "orch" if nid.startswith("M") else "worker")
        for nid in ["M1", "M2", "M3", "W1", "W2", "M4"]
    }
    edges = [
        ConversationEdge(
            source_node_id="M1",
            target_node_id="M2",
            history_context=HistoryContext.FULL,
        ),
        ConversationEdge(
            source_node_id="M2",
            target_node_id="M3",
            history_context=HistoryContext.FULL,
        ),
        ConversationEdge(
            source_node_id="M3",
            target_node_id="M4",
            history_context=HistoryContext.FULL,
        ),
        ConversationEdge(
            source_node_id="M3",
            target_node_id="W1",
            history_context=HistoryContext.NEW,
        ),
        ConversationEdge(
            source_node_id="M3",
            target_node_id="W2",
            history_context=HistoryContext.NEW,
        ),
        ConversationEdge(
            source_node_id="W1",
            target_node_id="M4",
            history_context=HistoryContext.LAST,
        ),
        ConversationEdge(
            source_node_id="W2",
            target_node_id="M4",
            history_context=HistoryContext.LAST,
        ),
    ]
    return ConversationGraph(graph_id="fork_join", nodes=nodes, edges=edges)


def _compaction_graph() -> ConversationGraph[str]:
    """
    Build a compaction graph:
    A -full-> B -full-> C(summarize) -last-> D -full-> E -full-> F
    """
    nodes = {nid: _make_node(nid) for nid in ["A", "B", "C", "D", "E", "F"]}
    edges = [
        ConversationEdge(
            source_node_id="A",
            target_node_id="B",
            history_context=HistoryContext.FULL,
        ),
        ConversationEdge(
            source_node_id="B",
            target_node_id="C",
            history_context=HistoryContext.FULL,
        ),
        ConversationEdge(
            source_node_id="C",
            target_node_id="D",
            history_context=HistoryContext.LAST,
        ),
        ConversationEdge(
            source_node_id="D",
            target_node_id="E",
            history_context=HistoryContext.FULL,
        ),
        ConversationEdge(
            source_node_id="E",
            target_node_id="F",
            history_context=HistoryContext.FULL,
        ),
    ]
    return ConversationGraph(graph_id="compaction", nodes=nodes, edges=edges)


class TestDAGExecutionStateReadiness:
    """Test readiness tracking for DAG nodes.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_initial_ready_nodes_are_roots(self):
        """
        Root nodes (no incoming edges) should be ready immediately.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(3))
        assert state.get_ready_nodes() == ["n0"]

    @pytest.mark.smoke
    def test_completing_node_readies_children(self):
        """
        Completing a node should make its children ready if all
        parents are complete.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(3))
        newly_ready = state.mark_completed("n0", "req_n0", "resp_n0")
        assert newly_ready == ["n1"]

    @pytest.mark.sanity
    def test_fork_join_readiness(self):
        """
        In a fork/join pattern, the join node should not be ready until
        all parents (main chain + workers) have completed.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_fork_join_graph())

        # Only M1 is initially ready
        assert state.get_ready_nodes() == ["M1"]

        state.mark_completed("M1", "req_M1", "resp_M1")
        state.mark_completed("M2", "req_M2", "resp_M2")
        newly_ready = state.mark_completed("M3", "req_M3", "resp_M3")

        # M3 completes -> W1, W2 become ready; M4 not yet (workers pending)
        assert sorted(newly_ready) == ["W1", "W2"]
        assert "M4" not in state.get_ready_nodes()

        state.mark_completed("W1", "req_W1", "resp_W1")
        # M4 still not ready (W2 pending)
        assert "M4" not in state.get_ready_nodes()

        newly_ready = state.mark_completed("W2", "req_W2", "resp_W2")
        assert newly_ready == ["M4"]

    @pytest.mark.smoke
    def test_is_complete(self):
        """
        Graph should report complete after all nodes are done.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(2))
        assert not state.is_complete
        state.mark_completed("n0", "r0", "resp0")
        assert not state.is_complete
        state.mark_completed("n1", "r1", "resp1")
        assert state.is_complete

    @pytest.mark.sanity
    def test_mark_completed_raises_on_unknown_node(self):
        """
        Completing an unknown node should raise ValueError.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(1))
        with pytest.raises(ValueError, match="not in graph"):
            state.mark_completed("nonexistent", "r", "resp")

    @pytest.mark.sanity
    def test_mark_completed_raises_on_duplicate(self):
        """
        Completing the same node twice should raise ValueError.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(1))
        state.mark_completed("n0", "r", "resp")
        with pytest.raises(ValueError, match="already completed"):
            state.mark_completed("n0", "r", "resp")


class TestDAGExecutionStateWalkBack:
    """Test walk-back history assembly.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_root_node_has_no_history(self):
        """
        Root nodes should have None history.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(3))
        assert state.assemble_history("n0") is None

    @pytest.mark.smoke
    def test_linear_chain_full_history(self):
        """
        In a linear chain with full edges, each node should see all
        ancestor (request, response) pairs.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(4))
        state.mark_completed("n0", "r0", "resp0")
        state.mark_completed("n1", "r1", "resp1")
        state.mark_completed("n2", "r2", "resp2")

        hist = state.assemble_history("n3")
        assert hist == [("r0", "resp0"), ("r1", "resp1"), ("r2", "resp2")]

    @pytest.mark.sanity
    def test_new_edge_provides_no_history(self):
        """
        Nodes reached via new edges should have no history.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_fork_join_graph())
        state.mark_completed("M1", "r_M1", "resp_M1")
        state.mark_completed("M2", "r_M2", "resp_M2")
        state.mark_completed("M3", "r_M3", "resp_M3")

        # W1 reached via new edge from M3 -> no history
        assert state.assemble_history("W1") is None

    @pytest.mark.sanity
    def test_fork_join_history_assembly(self):
        """
        The join node should see the full main chain plus worker
        final outputs via last edges.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_fork_join_graph())
        state.mark_completed("M1", "r_M1", "resp_M1")
        state.mark_completed("M2", "r_M2", "resp_M2")
        state.mark_completed("M3", "r_M3", "resp_M3")
        state.mark_completed("W1", "r_W1", "resp_W1")
        state.mark_completed("W2", "r_W2", "resp_W2")

        hist = state.assemble_history("M4")
        expected = [
            ("r_M1", "resp_M1"),
            ("r_M2", "resp_M2"),
            ("r_M3", "resp_M3"),
            ("r_W1", "resp_W1"),
            ("r_W2", "resp_W2"),
        ]
        assert hist == expected

    @pytest.mark.sanity
    def test_compaction_boundary(self):
        """
        Compaction via last edge creates a history boundary. Nodes
        after the boundary should not see pre-compaction history.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_compaction_graph())
        state.mark_completed("A", "rA", "respA")
        state.mark_completed("B", "rB", "respB")

        # C sees full history (A, B)
        hist_c = state.assemble_history("C")
        assert hist_c == [("rA", "respA"), ("rB", "respB")]

        state.mark_completed("C", "summarize", "summary")

        # D sees only C's output (last edge = boundary)
        hist_d = state.assemble_history("D")
        assert hist_d == [("summarize", "summary")]

        state.mark_completed("D", "rD", "respD")

        # E walks back through D, stops at D (no full incoming to D)
        hist_e = state.assemble_history("E")
        assert hist_e == [("rD", "respD")]

        state.mark_completed("E", "rE", "respE")

        # F walks back through E and D
        hist_f = state.assemble_history("F")
        assert hist_f == [("rD", "respD"), ("rE", "respE")]

    @pytest.mark.sanity
    def test_last_only_node(self):
        """
        A node with only last incoming edges should see only direct
        parent outputs, sorted by source_node_id.

        ## WRITTEN BY AI ##
        """
        g = ConversationGraph(
            graph_id="last_only",
            nodes={
                "a": _make_node("a"),
                "b": _make_node("b"),
                "c": _make_node("c"),
            },
            edges=[
                ConversationEdge(
                    source_node_id="a",
                    target_node_id="c",
                    history_context=HistoryContext.LAST,
                ),
                ConversationEdge(
                    source_node_id="b",
                    target_node_id="c",
                    history_context=HistoryContext.LAST,
                ),
            ],
        )
        state = DAGExecutionState(g)
        state.mark_completed("a", "rA", "respA")
        state.mark_completed("b", "rB", "respB")

        hist = state.assemble_history("c")
        # Sorted by source_node_id: a before b
        assert hist == [("rA", "respA"), ("rB", "respB")]


class TestDAGExecutionStateAbort:
    """Test graph abort behavior.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_abort_returns_remaining_nodes(self):
        """
        Aborting should return all incomplete node IDs.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(3))
        state.mark_completed("n0", "r0", "resp0")
        remaining = state.abort()
        assert sorted(remaining) == ["n1", "n2"]

    @pytest.mark.smoke
    def test_abort_prevents_further_ready_nodes(self):
        """
        After abort, no nodes should be reported as ready.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(3))
        state.abort()
        assert state.get_ready_nodes() == []
        assert state.is_aborted

    @pytest.mark.sanity
    def test_abort_marks_complete(self):
        """
        An aborted graph should report as complete.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(2))
        state.abort()
        assert state.is_complete


class TestDAGExecutionStateTopologicalOrder:
    """Test topological ordering.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_linear_topological_order(self):
        """
        Topological order of a linear chain should match the chain order.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_linear_graph(5))
        order = state.topological_order()
        for i in range(4):
            assert order.index(f"n{i}") < order.index(f"n{i+1}")

    @pytest.mark.sanity
    def test_fork_join_topological_order(self):
        """
        In a fork/join graph, all predecessors should appear before
        their dependents in topological order.

        ## WRITTEN BY AI ##
        """
        state = DAGExecutionState(_fork_join_graph())
        order = state.topological_order()

        # M1 before M2 before M3
        assert order.index("M1") < order.index("M2") < order.index("M3")
        # M3 before W1, W2
        assert order.index("M3") < order.index("W1")
        assert order.index("M3") < order.index("W2")
        # W1, W2, M3 all before M4
        assert order.index("W1") < order.index("M4")
        assert order.index("W2") < order.index("M4")
        assert order.index("M3") < order.index("M4")
