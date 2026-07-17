"""
Unit tests for conversation graph data interchange and assembler.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.schemas import GenerationRequest, RequestSettings
from guidellm.schemas.conversation_graph import (
    GenerativeConversationGraph,
    GenerativeConversationNode,
)


class TestConversationGraphData:
    """Validate ConversationGraphData schema rules.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_duplicate_node_ids_rejected(self):
        """
        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match="unique"):
            ConversationGraphData(
                turns=[
                    ConversationTurnData(node_id="a", columns={}),
                    ConversationTurnData(node_id="a", columns={}),
                ]
            )

    @pytest.mark.smoke
    def test_root_and_child_parents(self):
        """
        ## WRITTEN BY AI ##
        """
        data = ConversationGraphData(
            turns=[
                ConversationTurnData(node_id="root", columns={}),
                ConversationTurnData(
                    node_id="child",
                    parents=[
                        ConversationParentRef(
                            parent_node_id="root",
                            history_context="full",
                        )
                    ],
                    columns={},
                ),
            ]
        )
        assert data.turns[0].parents == []
        assert data.turns[1].parents[0].parent_node_id == "root"


class TestFromNodesWithParents:
    """Validate GenerativeConversationGraph.from_nodes_with_parents.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_assembles_fork_join_edges(self):
        """
        ## WRITTEN BY AI ##
        """
        nodes = {
            "main_0": GenerativeConversationNode(
                node_id="main_0",
                agent_id="default",
                request=GenerationRequest(columns={"text_column": ["a"]}),
                settings=RequestSettings(),
            ),
            "main_1": GenerativeConversationNode(
                node_id="main_1",
                agent_id="default",
                request=GenerationRequest(columns={"text_column": ["b"]}),
                settings=RequestSettings(),
            ),
            "branch_0_0": GenerativeConversationNode(
                node_id="branch_0_0",
                agent_id="worker",
                request=GenerationRequest(columns={"text_column": ["c"]}),
                settings=RequestSettings(),
            ),
        }
        graph = GenerativeConversationGraph.from_nodes_with_parents(
            nodes=nodes,
            parents_by_node={
                "main_0": [],
                "main_1": [
                    ("main_0", "full"),
                    ("branch_0_0", "last"),
                ],
                "branch_0_0": [("main_0", "new")],
            },
            graph_id="g1",
        )
        assert graph.graph_id == "g1"
        triples = {
            (e.source_node_id, e.target_node_id, e.history_context) for e in graph.edges
        }
        assert ("main_0", "main_1", "full") in triples
        assert ("main_0", "branch_0_0", "new") in triples
        assert ("branch_0_0", "main_1", "last") in triples

    @pytest.mark.sanity
    def test_missing_parent_raises(self):
        """
        ## WRITTEN BY AI ##
        """
        nodes = {
            "main_0": GenerativeConversationNode(
                node_id="main_0",
                agent_id="default",
                request=GenerationRequest(columns={}),
                settings=RequestSettings(),
            ),
        }
        with pytest.raises(ValueError, match="not in the node map"):
            GenerativeConversationGraph.from_nodes_with_parents(
                nodes=nodes,
                parents_by_node={"main_0": [("missing", "full")]},
            )

    @pytest.mark.sanity
    def test_empty_nodes_raises(self):
        """
        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="empty node map"):
            GenerativeConversationGraph.from_nodes_with_parents(
                nodes={},
                parents_by_node={},
            )
