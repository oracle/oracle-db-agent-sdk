##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for the public parallel composition pattern."""

import json

import pytest
from pyagentspec import Agent
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import (
    AgentNode,
    EndNode,
    LlmNode,
    ParallelFlowNode,
    StartNode,
    ToolNode,
)

from oracle_db_agent_sdk import (
    IncompatibleParallelContractError,
    PatternCompilationError,
    llm_node,
    parallel,
    plsql_tool,
    sequential,
    tool_node,
)
from oracle_db_agent_sdk._serialization import _serialize_workflow
from oracle_db_agent_sdk._wrappers import Executable, LlmConfig, _unwrap_executable


def _unwrap_flow(value: Executable) -> Flow:
    component = _unwrap_executable(value)
    assert isinstance(component, Flow)
    return component


def _unwrap_agent(value: Executable) -> Agent:
    component = _unwrap_executable(value)
    assert isinstance(component, Agent)
    return component


def test_parallel_wraps_agents_in_standard_subflows(make_agent) -> None:
    medical = make_agent(
        "medical",
        inputs=(("claim_id", "string"), ("medical_notes", "string")),
        outputs=(("medical_result", "string"),),
    )
    policy = make_agent(
        "policy",
        inputs=(("policy_number", "string"),),
        outputs=(("policy_result", "string"),),
    )

    flow = _unwrap_flow(parallel(name="parallel_test", steps=[medical, policy]))

    assert [type(node) for node in flow.nodes] == [
        StartNode,
        ParallelFlowNode,
        EndNode,
    ]
    parallel_node = flow.nodes[1]
    assert isinstance(parallel_node, ParallelFlowNode)
    assert len(parallel_node.subflows) == 2
    assert [[type(node) for node in subflow.nodes] for subflow in parallel_node.subflows] == [
        [StartNode, AgentNode, EndNode],
        [StartNode, AgentNode, EndNode],
    ]
    assert parallel_node.subflows[0].nodes[1].agent == _unwrap_agent(medical)
    assert parallel_node.subflows[1].nodes[1].agent == _unwrap_agent(policy)
    assert {property_.title for property_ in flow.inputs or ()} == {
        "claim_id",
        "medical_notes",
        "policy_number",
    }
    assert {property_.title for property_ in flow.outputs or ()} == {
        "medical_result",
        "policy_result",
    }
    assert [(edge.from_node.name, edge.to_node.name) for edge in flow.control_flow_connections] == [
        ("parallel-start", "parallel-branches"),
        ("parallel-branches", "parallel-end"),
    ]
    assert [
        (
            edge.source_node.name,
            edge.source_output,
            edge.destination_node.name,
            edge.destination_input,
        )
        for edge in flow.data_flow_connections or ()
    ] == [
        ("parallel-start", "claim_id", "parallel-branches", "claim_id"),
        ("parallel-start", "medical_notes", "parallel-branches", "medical_notes"),
        ("parallel-start", "policy_number", "parallel-branches", "policy_number"),
        ("parallel-branches", "medical_result", "parallel-end", "medical_result"),
        ("parallel-branches", "policy_result", "parallel-end", "policy_result"),
    ]

    medical_subflow, policy_subflow = parallel_node.subflows
    assert [
        (
            edge.source_node.name,
            edge.source_output,
            edge.destination_node.name,
            edge.destination_input,
        )
        for edge in medical_subflow.data_flow_connections or ()
    ] == [
        (
            "parallel-branch-000-start",
            "claim_id",
            "parallel-branch-000-MEDICAL",
            "claim_id",
        ),
        (
            "parallel-branch-000-start",
            "medical_notes",
            "parallel-branch-000-MEDICAL",
            "medical_notes",
        ),
        (
            "parallel-branch-000-MEDICAL",
            "medical_result",
            "parallel-branch-000-end",
            "medical_result",
        ),
    ]
    assert [
        (
            edge.source_node.name,
            edge.source_output,
            edge.destination_node.name,
            edge.destination_input,
        )
        for edge in policy_subflow.data_flow_connections or ()
    ] == [
        (
            "parallel-branch-001-start",
            "policy_number",
            "parallel-branch-001-POLICY",
            "policy_number",
        ),
        (
            "parallel-branch-001-POLICY",
            "policy_result",
            "parallel-branch-001-end",
            "policy_result",
        ),
    ]


def test_parallel_preserves_existing_flow_branch(make_agent) -> None:
    first = make_agent(
        "first",
        inputs=(("source", "string"),),
        outputs=(("intermediate", "string"),),
    )
    second = make_agent(
        "second",
        inputs=(("intermediate", "string"),),
        outputs=(("flow_result", "string"),),
    )
    existing_workflow = sequential(
        name="review_sequence",
        steps=[first, second],
    )
    existing_flow = _unwrap_flow(existing_workflow)
    other = make_agent(
        "other",
        inputs=(("other_input", "string"),),
        outputs=(("other_result", "string"),),
    )

    outer = _unwrap_flow(parallel(name="parallel_test", steps=[existing_workflow, other]))

    parallel_node = outer.nodes[1]
    assert isinstance(parallel_node, ParallelFlowNode)
    assert parallel_node.subflows[0] == existing_flow
    assert parallel_node.subflows[0] is not existing_flow


def test_parallel_deduplicates_compatible_shared_inputs(make_agent) -> None:
    medical = make_agent(
        "medical",
        inputs=(("claim_id", "string"), ("medical_notes", "string")),
        outputs=(("medical_result", "string"),),
    )
    fraud = make_agent(
        "fraud",
        inputs=(("claim_id", "string"), ("customer_id", "string")),
        outputs=(("fraud_result", "string"),),
    )

    flow = _unwrap_flow(parallel(name="parallel_test", steps=[medical, fraud]))

    input_names = [property_.title for property_ in flow.inputs or ()]
    assert input_names.count("claim_id") == 1
    assert set(input_names) == {"claim_id", "medical_notes", "customer_id"}


def test_parallel_composes_tool_and_llm_node_branches(
    database_llm: LlmConfig,
) -> None:
    status = tool_node(
        name="claim_status",
        tool=plsql_tool(
            name="get_claim_status",
            procedure="CLAIMS_SERVICE.GET_STATUS",
            inputs={"claim_id": "string"},
            outputs={"status_result": "string"},
        ),
    )
    policy = llm_node(
        name="policy_review",
        llm=database_llm,
        prompt="Review policy {{policy_id}}.",
        inputs={"policy_id": "string"},
        outputs={"policy_result": "string"},
    )

    flow = _unwrap_flow(parallel(name="parallel_test", steps=[status, policy]))
    parallel_node = flow.nodes[1]

    assert isinstance(parallel_node, ParallelFlowNode)
    assert isinstance(parallel_node.subflows[0].nodes[1], ToolNode)
    assert isinstance(parallel_node.subflows[1].nodes[1], LlmNode)
    assert {property_.title for property_ in flow.inputs or ()} == {
        "claim_id",
        "policy_id",
    }


def test_parallel_rejects_conflicting_shared_input_types(make_agent) -> None:
    medical = make_agent(
        "medical",
        inputs=(("claim_id", "string"),),
        outputs=(("medical_result", "string"),),
    )
    fraud = make_agent(
        "fraud",
        inputs=(("claim_id", "integer"),),
        outputs=(("fraud_result", "string"),),
    )

    with pytest.raises(
        IncompatibleParallelContractError,
        match="claim_id.*incompatible types",
    ):
        parallel(name="parallel_test", steps=[medical, fraud])


def test_parallel_rejects_duplicate_branch_outputs(make_agent) -> None:
    medical = make_agent(
        "medical",
        inputs=(("medical_notes", "string"),),
        outputs=(("review_result", "string"),),
    )
    fraud = make_agent(
        "fraud",
        inputs=(("transaction_history", "string"),),
        outputs=(("review_result", "string"),),
    )

    with pytest.raises(
        IncompatibleParallelContractError,
        match="duplicate output property 'review_result'",
    ):
        parallel(name="parallel_test", steps=[medical, fraud])


def test_parallel_construction_does_not_mutate_sources(make_agent) -> None:
    medical = make_agent(
        "medical",
        inputs=(("medical_notes", "string"),),
        outputs=(("medical_result", "string"),),
    )
    fraud = make_agent(
        "fraud",
        inputs=(("transaction_history", "string"),),
        outputs=(("fraud_result", "string"),),
    )
    raw_agents = [_unwrap_agent(medical), _unwrap_agent(fraud)]
    before = [value.model_copy(deep=True) for value in raw_agents]

    parallel(name="parallel_test", steps=[medical, fraud])

    assert raw_agents == before


@pytest.mark.parametrize("count", [0, 1])
def test_parallel_requires_at_least_two_steps(make_agent, count) -> None:
    with pytest.raises(PatternCompilationError, match="at least two"):
        parallel(name="parallel_test", steps=[make_agent("medical")] * count)


@pytest.mark.parametrize("name", ["", None, 123])
def test_parallel_rejects_invalid_names(make_agent, name) -> None:
    with pytest.raises(PatternCompilationError, match="name must be a non-empty string"):
        parallel(name=name, steps=[make_agent("medical"), make_agent("policy")])


@pytest.mark.parametrize("steps", [None, "ab", b"ab", {}, iter(())])
def test_parallel_rejects_non_sequence_steps(steps) -> None:
    with pytest.raises(PatternCompilationError, match="steps must be a sequence"):
        parallel(name="parallel_test", steps=steps)


@pytest.mark.parametrize("collection", [list, tuple])
def test_parallel_preserves_user_name_in_serialized_flow(make_agent, collection) -> None:
    steps = collection(
        [
            make_agent("medical", outputs=(("medical_result", "string"),)),
            make_agent("policy", outputs=(("policy_result", "string"),)),
        ]
    )
    workflow = parallel(name="claim_review", steps=steps)

    assert _unwrap_flow(workflow).name == "claim_review"
    assert json.loads(_serialize_workflow(workflow))["name"] == "claim_review"


def test_parallel_rejects_legacy_positional_arguments(make_agent) -> None:
    with pytest.raises(TypeError):
        parallel(make_agent("medical"), make_agent("policy"))  # type: ignore[call-arg]


def test_parallel_rejects_non_building_block_branches(make_agent) -> None:
    with pytest.raises(PatternCompilationError, match="oracle_db_agent_sdk building block"):
        parallel(name="parallel_test", steps=[make_agent("medical"), object()])  # type: ignore[arg-type]


def test_parallel_returns_opaque_executable(make_agent) -> None:
    medical = make_agent(
        "medical",
        outputs=(("medical_result", "string"),),
    )
    fraud = make_agent(
        "fraud",
        outputs=(("fraud_result", "string"),),
    )

    workflow = parallel(name="parallel_test", steps=[medical, fraud])
    assert type(workflow) is Executable
    assert isinstance(_unwrap_executable(workflow), Flow)
