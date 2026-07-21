"""
Unit tests for guidellm.data.finalizers module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.data.finalizers import (
    FinalizerRegistry,
    GenerativeRequestFinalizer,
)
from guidellm.data.finalizers.generative import GenerativeRequestFinalizerArgs
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.schemas import GenerationRequest, RequestSettings
from guidellm.schemas.conversation_graph import GenerativeConversationGraph


def _ordered_requests(
    graph: GenerativeConversationGraph,
) -> list[GenerationRequest]:
    """Return graph node requests ordered by turn index suffix.

    ## WRITTEN BY AI ##
    """
    return [
        graph.nodes[nid].request
        for nid in sorted(
            graph.nodes,
            key=lambda nid: int(nid.rsplit("_", 1)[-1]),
        )
    ]


class TestGenerativeRequestFinalizerTokenAggregation:
    """Test cases for GenerativeRequestFinalizer token aggregation.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of GenerativeRequestFinalizer.

        ### WRITTEN BY AI ###
        """
        return GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

    @pytest.mark.smoke
    def test_finalize_single_turn_prompt_tokens(self, valid_instances):
        """Test finalize with single prompt token count.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {"prompt_tokens_count_column": [100]}

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        assert gen_req.input_metrics.text_tokens == 100

    @pytest.mark.smoke
    def test_finalize_multi_turn_prompt_tokens(self, valid_instances):
        """Test finalize with multiple prompt token counts (sums them).

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {"prompt_tokens_count_column": [50, 75, 100]}

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        assert gen_req.input_metrics.text_tokens == 225  # 50 + 75 + 100

    @pytest.mark.smoke
    def test_finalize_multi_turn_output_tokens(self, valid_instances):
        """Test finalize with multiple output token counts (sums them).

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {"output_tokens_count_column": [20, 30, 40]}

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        assert gen_req.output_metrics.text_tokens == 90  # 20 + 30 + 40

    @pytest.mark.sanity
    def test_finalize_with_none_values_in_list(self, valid_instances):
        """Test finalize skips None values when summing.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {"prompt_tokens_count_column": [50, None, 100]}

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        assert gen_req.input_metrics.text_tokens == 150  # 50 + 100, skips None

    @pytest.mark.regression
    def test_finalize_with_empty_column_lists(self, valid_instances):
        """Test finalize with empty column lists.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "prompt_tokens_count_column": [],
            "output_tokens_count_column": [],
        }

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        assert gen_req.input_metrics.text_tokens is None
        assert gen_req.output_metrics.text_tokens is None


class TestGenerativeRequestFinalizerMultimodal:
    """Test cases for GenerativeRequestFinalizer multimodal aggregation.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of GenerativeRequestFinalizer.

        ### WRITTEN BY AI ###
        """
        return GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

    @pytest.mark.sanity
    def test_finalize_multi_value_text_columns(self, valid_instances):
        """Test finalize accumulates text metrics for multiple text values.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "text_column": [
                "Hello world",
                "How are you?",
                "I am fine",
            ],
        }

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        # Should accumulate metrics from all text values
        assert gen_req.input_metrics.text_words > 0
        assert gen_req.input_metrics.text_characters > 0

    @pytest.mark.sanity
    def test_finalize_multi_value_image_columns(self, valid_instances):
        """Test finalize sums image pixels and bytes across multiple images.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "image_column": [
                {"image_pixels": 1000, "image_bytes": 5000},
                {"image_pixels": 2000, "image_bytes": 10000},
                {"image_pixels": 1500, "image_bytes": 7500},
            ],
        }

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        assert gen_req.input_metrics.image_pixels == 4500  # 1000 + 2000 + 1500
        assert gen_req.input_metrics.image_bytes == 22500  # 5000 + 10000 + 7500

    @pytest.mark.regression
    def test_finalize_preserves_columns(self, valid_instances):
        """Test finalize preserves input columns in result.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "text_column": ["Hello"],
            "prompt_tokens_count_column": [50],
            "output_tokens_count_column": [25],
        }

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        # Original columns should be preserved
        assert gen_req.columns == columns
        # And metrics should be set
        assert gen_req.input_metrics.text_tokens == 50
        assert gen_req.output_metrics.text_tokens == 25


class TestFinalizerTopLevel:
    """Test cases for top-level finalizer interface.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of GenerativeRequestFinalizer.

        ### WRITTEN BY AI ###
        """
        return GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

    @pytest.mark.smoke
    def test_finalizer_returns_conversation_graph(self, valid_instances):
        """Test __call__ returns a GenerativeConversationGraph.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        items = [
            {"prompt_tokens_count_column": [50]},
            {"prompt_tokens_count_column": [75]},
            {"prompt_tokens_count_column": [100]},
        ]

        result = instance(items)

        assert isinstance(result, GenerativeConversationGraph)
        requests = _ordered_requests(result)
        assert len(requests) == 3
        assert all(isinstance(r, GenerationRequest) for r in requests)
        assert [r.input_metrics.text_tokens for r in requests] == [50, 75, 100]

    @pytest.mark.sanity
    def test_finalizer_handles_empty_list(self, valid_instances):
        """Test __call__ handles empty list.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        items = []

        result = instance(items)

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.sanity
    def test_finalizer_aggregates_multimodal_metrics(self, valid_instances):
        """Test finalize aggregates all multimodal metrics correctly.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        columns = {
            "text_column": ["Hello world"],
            "image_column": [{"image_pixels": 1920 * 1080, "image_bytes": 50000}],
            "video_column": [
                {"video_frames": 120, "video_seconds": 4.0, "video_bytes": 1000000}
            ],
            "audio_column": [
                {"audio_samples": 48000, "audio_seconds": 1.0, "audio_bytes": 96000}
            ],
        }

        gen_req, req_settings = instance.finalize_turn(columns)

        assert isinstance(gen_req, GenerationRequest)
        assert isinstance(req_settings, RequestSettings)
        # Text metrics
        assert gen_req.input_metrics.text_words == 2
        assert gen_req.input_metrics.text_characters == 11

        # Image metrics
        assert gen_req.input_metrics.image_pixels == 1920 * 1080
        assert gen_req.input_metrics.image_bytes == 50000

        # Video metrics
        assert gen_req.input_metrics.video_frames == 120
        assert gen_req.input_metrics.video_seconds == 4.0
        assert gen_req.input_metrics.video_bytes == 1000000

        # Audio metrics
        assert gen_req.input_metrics.audio_samples == 48000
        assert gen_req.input_metrics.audio_seconds == 1.0
        assert gen_req.input_metrics.audio_bytes == 96000


class TestFinalizerRegistry:
    """Test cases for FinalizerRegistry.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_registry_has_generative(self):
        """Test registry has 'generative' finalizer registered.

        ### WRITTEN BY AI ###
        """
        finalizer_cls = FinalizerRegistry.get_registered_object("generative")

        assert finalizer_cls is not None
        assert finalizer_cls == GenerativeRequestFinalizer

    @pytest.mark.sanity
    def test_protocol_conformance(self):
        """Test GenerativeRequestFinalizer conforms to DatasetFinalizer protocol.

        ### WRITTEN BY AI ###
        """
        instance = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

        # Should have __call__ method
        assert callable(instance)

        # Test it works as expected
        result = instance([{"text_column": ["test"]}])
        assert isinstance(result, GenerativeConversationGraph)


class TestFinalizerTurnType:
    """Verify GenerativeRequestFinalizer sets turn_type correctly
    and splits tool-calling turns into tool_call + injection pairs.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def finalizer(self):
        """
        ## WRITTEN BY AI ##
        """
        return GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

    @pytest.mark.smoke
    def test_tool_turn_produces_tool_call_plus_injection(self, finalizer):
        """A turn with tools_column produces a tool_call request followed
        by a tool_response_injection request. Output metrics are moved
        to the injection turn so the tool call turn is unconstrained.

        ## WRITTEN BY AI ##
        """
        items = [
            {
                "text_column": ["hello"],
                "tools_column": ['[{"type": "function"}]'],
                "tool_response_column": ['{"status": "ok"}'],
                "output_tokens_count_column": [20],
            },
            {"text_column": ["world"]},
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 3
        assert results[0].turn_type == "client_tool_call"
        assert "tool_response_column" not in results[0].columns
        assert results[0].output_metrics.text_tokens is None
        assert results[1].turn_type == "tool_response_injection"
        assert results[1].columns["tool_response_column"] == ['{"status": "ok"}']
        assert results[1].output_metrics.text_tokens == 20
        assert results[2].turn_type == "standard"

    @pytest.mark.smoke
    def test_all_turns_with_tools_all_produce_pairs(self, finalizer):
        """When every turn has tools_column, each produces a pair.

        ## WRITTEN BY AI ##
        """
        items = [
            {"text_column": ["hello"], "tools_column": ['[{"type": "function"}]']},
            {"text_column": ["world"], "tools_column": ['[{"type": "function"}]']},
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 4
        assert results[0].turn_type == "client_tool_call"
        assert results[1].turn_type == "tool_response_injection"
        assert results[2].turn_type == "client_tool_call"
        assert results[3].turn_type == "tool_response_injection"

    @pytest.mark.sanity
    def test_standard_turns_without_tools(self, finalizer):
        """Turns without tools_column have turn_type='standard' and no split.

        ## WRITTEN BY AI ##
        """
        items = [
            {"text_column": ["hello"]},
            {"text_column": ["world"]},
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 2
        assert results[0].turn_type == "standard"
        assert results[1].turn_type == "standard"

    @pytest.mark.sanity
    def test_single_turn_with_tools_produces_pair(self, finalizer):
        """A single-turn conversation with tools produces a pair.

        ## WRITTEN BY AI ##
        """
        items = [
            {"text_column": ["hello"], "tools_column": ['[{"type": "function"}]']},
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 2
        assert results[0].turn_type == "client_tool_call"
        assert results[1].turn_type == "tool_response_injection"


class TestFinalizerTurnTypeColumn:
    """Verify turn_type_column overrides automatic turn type inference.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def finalizer(self):
        """
        ## WRITTEN BY AI ##
        """
        return GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

    @pytest.mark.smoke
    def test_turn_type_column_sets_server_tool_call(self, finalizer):
        """turn_type_column="server_tool_call" overrides default inference.

        ## WRITTEN BY AI ##
        """
        items = [
            {
                "text_column": ["hello"],
                "turn_type_column": ["server_tool_call"],
            },
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 1
        assert results[0].turn_type == "server_tool_call"

    @pytest.mark.smoke
    def test_turn_type_column_takes_priority_over_tools_column(self, finalizer):
        """turn_type_column takes priority even when tools_column is present.

        ## WRITTEN BY AI ##
        """
        items = [
            {
                "text_column": ["hello"],
                "tools_column": ['[{"type": "function"}]'],
                "turn_type_column": ["server_tool_call"],
            },
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 1
        assert results[0].turn_type == "server_tool_call"

    @pytest.mark.sanity
    def test_server_tool_call_no_injection_turn(self, finalizer):
        """server_tool_call turns do not produce injection turns.

        ## WRITTEN BY AI ##
        """
        items = [
            {
                "text_column": ["hello"],
                "turn_type_column": ["server_tool_call"],
            },
            {
                "text_column": ["world"],
            },
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 2
        assert results[0].turn_type == "server_tool_call"
        assert results[1].turn_type == "standard"

    @pytest.mark.sanity
    def test_empty_turn_type_column_falls_through(self, finalizer):
        """Empty turn_type_column falls back to tools_column inference.

        ## WRITTEN BY AI ##
        """
        items = [
            {
                "text_column": ["hello"],
                "turn_type_column": [],
                "tools_column": ['[{"type": "function"}]'],
            },
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 2
        assert results[0].turn_type == "client_tool_call"
        assert results[1].turn_type == "tool_response_injection"


class TestFinalizerToolCallMode:
    """Verify tool_call_mode config controls turn type for tools_column turns.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_tool_call_mode_client_default(self):
        """Default tool_call_mode is 'client'.

        ## WRITTEN BY AI ##
        """
        config = GenerativeRequestFinalizerArgs()
        assert config.tool_call_mode == "client"

    @pytest.mark.smoke
    def test_tool_call_mode_client_produces_client_tool_call(self):
        """tool_call_mode='client' produces client_tool_call + injection.

        ## WRITTEN BY AI ##
        """
        finalizer = GenerativeRequestFinalizer(
            GenerativeRequestFinalizerArgs(tool_call_mode="client")
        )
        items = [
            {
                "text_column": ["hello"],
                "tools_column": ['[{"type": "function"}]'],
                "tool_response_column": ['{"status": "ok"}'],
            },
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 2
        assert results[0].turn_type == "client_tool_call"
        assert results[1].turn_type == "tool_response_injection"

    @pytest.mark.smoke
    def test_tool_call_mode_server_produces_server_tool_call(self):
        """tool_call_mode='server' produces server_tool_call without injection.

        ## WRITTEN BY AI ##
        """
        finalizer = GenerativeRequestFinalizer(
            GenerativeRequestFinalizerArgs(tool_call_mode="server")
        )
        items = [
            {
                "text_column": ["hello"],
                "tools_column": ['[{"type": "function"}]'],
                "tool_response_column": ['{"status": "ok"}'],
            },
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 1
        assert results[0].turn_type == "server_tool_call"

    @pytest.mark.sanity
    def test_tool_call_mode_server_strips_tool_columns(self):
        """tool_call_mode='server' strips tools_column and tool_response_column.

        ## WRITTEN BY AI ##
        """
        finalizer = GenerativeRequestFinalizer(
            GenerativeRequestFinalizerArgs(tool_call_mode="server")
        )
        items = [
            {
                "text_column": ["hello"],
                "tools_column": ['[{"type": "function"}]'],
                "tool_response_column": ['{"status": "ok"}'],
            },
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 1
        assert "tools_column" not in results[0].columns
        assert "tool_response_column" not in results[0].columns

    @pytest.mark.sanity
    def test_tool_call_mode_server_no_effect_on_standard_turns(self):
        """tool_call_mode='server' has no effect on turns without tools_column.

        ## WRITTEN BY AI ##
        """
        finalizer = GenerativeRequestFinalizer(
            GenerativeRequestFinalizerArgs(tool_call_mode="server")
        )
        items = [
            {"text_column": ["hello"]},
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 1
        assert results[0].turn_type == "standard"

    @pytest.mark.sanity
    def test_tool_call_mode_server_mixed_turns(self):
        """tool_call_mode='server' with mixed tool/non-tool turns.

        ## WRITTEN BY AI ##
        """
        finalizer = GenerativeRequestFinalizer(
            GenerativeRequestFinalizerArgs(tool_call_mode="server")
        )
        items = [
            {
                "text_column": ["tool turn"],
                "tools_column": ['[{"type": "function"}]'],
            },
            {"text_column": ["standard turn"]},
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        assert len(results) == 2
        assert results[0].turn_type == "server_tool_call"
        assert results[1].turn_type == "standard"

    @pytest.mark.sanity
    def test_turn_type_column_overrides_tool_call_mode(self):
        """turn_type_column takes priority over tool_call_mode config.

        ## WRITTEN BY AI ##
        """
        finalizer = GenerativeRequestFinalizer(
            GenerativeRequestFinalizerArgs(tool_call_mode="server")
        )
        items = [
            {
                "text_column": ["hello"],
                "tools_column": ['[{"type": "function"}]'],
                "turn_type_column": ["client_tool_call"],
            },
        ]
        graph = finalizer(items)
        assert isinstance(graph, GenerativeConversationGraph)
        results = _ordered_requests(graph)

        # turn_type_column says client_tool_call, so it should create
        # a client_tool_call + injection pair despite server mode
        assert len(results) == 2
        assert results[0].turn_type == "client_tool_call"
        assert results[1].turn_type == "tool_response_injection"


class TestGenerativeRequestFinalizerRequestSettings:
    """Verify relative_timestamp_column maps to GenerationRequest.settings.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def finalizer(self):
        """### WRITTEN BY AI ###"""
        return GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())

    @pytest.mark.smoke
    def test_relative_timestamp_column_sets_settings(self, finalizer):
        """### WRITTEN BY AI ###"""
        _gen_req, req_settings = finalizer.finalize_turn(
            {"relative_timestamp_column": [2.5]}
        )

        assert req_settings == RequestSettings(relative_timestamp=2.5)

    @pytest.mark.smoke
    def test_missing_relative_timestamp_column_uses_empty_settings(self, finalizer):
        """### WRITTEN BY AI ###"""
        _gen_req, req_settings = finalizer.finalize_turn({"text_column": ["hello"]})

        assert req_settings == RequestSettings()

    @pytest.mark.smoke
    def test_none_relative_timestamp_column_uses_empty_settings(self, finalizer):
        """### WRITTEN BY AI ###"""
        _gen_req, req_settings = finalizer.finalize_turn(
            {"relative_timestamp_column": [None]}
        )

        assert req_settings == RequestSettings()


class TestFinalizerConversationGraph:
    """Verify finalizer wraps request pairs into conversation graphs.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_linear_chain_without_branches(self):
        """Non-empty rows without branches become a linear conversation graph.

        ## WRITTEN BY AI ##
        """
        finalizer = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())
        items = [
            {"text_column": ["hello"]},
            {"text_column": ["world"]},
        ]

        graph = finalizer(items)

        assert isinstance(graph, GenerativeConversationGraph)
        assert set(graph.nodes) == {"turn_0", "turn_1"}
        assert len(graph.edges) == 1
        assert graph.edges[0].source_node_id == "turn_0"
        assert graph.edges[0].target_node_id == "turn_1"
        assert graph.nodes["turn_0"].settings == RequestSettings()

    @pytest.mark.sanity
    def test_graph_preserves_request_settings_from_pairs(self):
        """Node settings come from RequestSettings pairs, not the request.

        ## WRITTEN BY AI ##
        """
        finalizer = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())
        items = [
            {
                "text_column": ["hello"],
                "relative_timestamp_column": [1.5],
                "requeue_delay_column": [0.25],
            },
        ]

        graph = finalizer(items)

        assert isinstance(graph, GenerativeConversationGraph)
        node = graph.nodes["turn_0"]
        assert node.settings == RequestSettings(
            relative_timestamp=1.5,
            requeue_delay=0.25,
        )

    @pytest.mark.smoke
    def test_conversation_turns_column_builds_fork_join_graph(self):
        """conversation_turns_column assembles a fork/join graph from inline parents.

        ## WRITTEN BY AI ##
        """
        graph_data = ConversationGraphData(
            turns=[
                ConversationTurnData(
                    node_id="main_0",
                    columns={"text_column": ["main_0"]},
                ),
                ConversationTurnData(
                    node_id="main_1",
                    parents=[
                        ConversationParentRef(
                            parent_node_id="main_0",
                            history_context="full",
                        ),
                        ConversationParentRef(
                            parent_node_id="branch_0_0",
                            history_context="last",
                        ),
                    ],
                    columns={
                        "text_column": ["main_1"],
                        "output_tokens_count_column": [5],
                    },
                ),
                ConversationTurnData(
                    node_id="branch_0_0",
                    agent_id="sub_agent",
                    parents=[
                        ConversationParentRef(
                            parent_node_id="main_0",
                            history_context="new",
                        )
                    ],
                    columns={
                        "text_column": ["branch"],
                        "output_tokens_count_column": [5],
                    },
                ),
            ]
        )
        finalizer = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())
        graph = finalizer(
            [{"conversation_turns_column": [graph_data.model_dump(mode="json")]}]
        )

        assert isinstance(graph, GenerativeConversationGraph)
        assert set(graph.nodes) == {"main_0", "main_1", "branch_0_0"}
        assert graph.nodes["branch_0_0"].agent_id == "sub_agent"
        edge_triples = {
            (e.source_node_id, e.target_node_id, e.history_context) for e in graph.edges
        }
        assert (
            "main_0",
            "branch_0_0",
            "new",
        ) in edge_triples
        assert (
            "branch_0_0",
            "main_1",
            "last",
        ) in edge_triples
        assert (
            "main_0",
            "main_1",
            "full",
        ) in edge_triples
