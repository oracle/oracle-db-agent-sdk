##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for the public routing composition pattern."""

import json

import pytest
from pyagentspec import Agent
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import (
    AgentNode,
    BranchingNode,
    EndNode,
    FlowNode,
    LlmNode,
    StartNode,
    ToolNode,
)

from oracle_db_agent_sdk import (
    IncompatibleRoutingContractError,
    PatternCompilationError,
    llm_node,
    parallel,
    plsql_tool,
    routing,
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


@pytest.mark.parametrize("name", ["", " ", "\t\n", None, 123])
def test_routing_rejects_invalid_names(name) -> None:
    with pytest.raises(PatternCompilationError, match="name must be a non-empty string"):
        routing(router=object(), name=name, route_key="route", routes={}, default=object())


def test_routing_preserves_supplied_name(make_agent) -> None:
    selector = make_agent("selector", outputs=(("route", "string"),))
    target = make_agent("target")
    workflow = routing(
        name=" ticket_routing ",
        router=selector,
        route_key="route",
        routes={"a": target},
        default=target,
    )
    assert _unwrap_flow(workflow).name == " ticket_routing "
    assert json.loads(_serialize_workflow(workflow))["name"] == " ticket_routing "


def test_routing_requires_name(make_agent) -> None:
    selector = make_agent("selector", outputs=(("route", "string"),))
    target = make_agent("target")
    with pytest.raises(TypeError, match="name"):
        routing(selector, route_key="route", routes={"a": target}, default=target)  # type: ignore[call-arg]


def test_routing_builds_standard_branching_flow(make_agent) -> None:
    router = make_agent(
        "router",
        inputs=(("ticket_text", "string"), ("customer_id", "string")),
        outputs=(("issue_type", "string"),),
    )
    billing = make_agent(
        "billing",
        inputs=(("ticket_text", "string"), ("customer_id", "string")),
        outputs=(("resolution", "string"),),
    )
    technical = make_agent(
        "technical",
        inputs=(("ticket_text", "string"),),
        outputs=(("resolution", "string"),),
    )
    general = make_agent(
        "general",
        inputs=(("ticket_text", "string"),),
        outputs=(("resolution", "string"),),
    )

    flow = _unwrap_flow(
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={
                "technical": technical,
                "billing": billing,
            },
            default=general,
        )
    )

    assert [type(node) for node in flow.nodes] == [
        StartNode,
        AgentNode,
        BranchingNode,
        AgentNode,
        AgentNode,
        AgentNode,
        EndNode,
        EndNode,
        EndNode,
    ]
    assert {property_.title for property_ in flow.inputs or ()} == {
        "ticket_text",
        "customer_id",
    }
    assert [property_.title for property_ in flow.outputs or ()] == ["resolution"]

    branching_node = flow.nodes[2]
    assert isinstance(branching_node, BranchingNode)
    assert branching_node.mapping == {
        "billing": "routing-route-000",
        "technical": "routing-route-001",
    }

    assert [
        (
            edge.from_node.name,
            edge.from_branch,
            edge.to_node.name,
        )
        for edge in flow.control_flow_connections
    ] == [
        ("routing-start", None, "routing-router-ROUTER"),
        ("routing-router-ROUTER", None, "routing-branch"),
        ("routing-branch", "routing-route-000", "routing-target-000-BILLING"),
        ("routing-target-000-BILLING", None, "routing-end-000"),
        ("routing-branch", "routing-route-001", "routing-target-001-TECHNICAL"),
        ("routing-target-001-TECHNICAL", None, "routing-end-001"),
        ("routing-branch", "default", "routing-default-GENERAL"),
        ("routing-default-GENERAL", None, "routing-default-end"),
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
        (
            "routing-start",
            "customer_id",
            "routing-router-ROUTER",
            "customer_id",
        ),
        (
            "routing-start",
            "ticket_text",
            "routing-router-ROUTER",
            "ticket_text",
        ),
        (
            "routing-router-ROUTER",
            "issue_type",
            "routing-branch",
            "issue_type",
        ),
        (
            "routing-start",
            "customer_id",
            "routing-target-000-BILLING",
            "customer_id",
        ),
        (
            "routing-start",
            "ticket_text",
            "routing-target-000-BILLING",
            "ticket_text",
        ),
        (
            "routing-target-000-BILLING",
            "resolution",
            "routing-end-000",
            "resolution",
        ),
        (
            "routing-start",
            "ticket_text",
            "routing-target-001-TECHNICAL",
            "ticket_text",
        ),
        (
            "routing-target-001-TECHNICAL",
            "resolution",
            "routing-end-001",
            "resolution",
        ),
        (
            "routing-start",
            "ticket_text",
            "routing-default-GENERAL",
            "ticket_text",
        ),
        (
            "routing-default-GENERAL",
            "resolution",
            "routing-default-end",
            "resolution",
        ),
    ]


def test_routing_accepts_flow_router_and_flow_target(make_agent) -> None:
    classifier = make_agent(
        "classifier",
        inputs=(("ticket_text", "string"),),
        outputs=(("normalized_ticket", "string"),),
    )
    decision = make_agent(
        "decision",
        inputs=(("normalized_ticket", "string"),),
        outputs=(("issue_type", "string"),),
    )
    router_workflow = sequential(
        name="router_sequence",
        steps=[classifier, decision],
    )
    router_flow = _unwrap_flow(router_workflow)

    diagnose = make_agent(
        "diagnose",
        inputs=(("ticket_text", "string"),),
        outputs=(("diagnosis", "string"),),
    )
    resolve = make_agent(
        "resolve",
        inputs=(("diagnosis", "string"),),
        outputs=(("resolution", "string"),),
    )
    technical_workflow = sequential(
        name="technical_sequence",
        steps=[diagnose, resolve],
    )
    technical_flow = _unwrap_flow(technical_workflow)
    general = make_agent(
        "general",
        inputs=(("ticket_text", "string"),),
        outputs=(("resolution", "string"),),
    )

    flow = _unwrap_flow(
        routing(
            router_workflow,
            name="routing_test",
            route_key="issue_type",
            routes={"technical": technical_workflow},
            default=general,
        )
    )

    assert isinstance(flow.nodes[1], FlowNode)
    assert flow.nodes[1].subflow == router_flow
    assert flow.nodes[1].subflow is not router_flow
    assert isinstance(flow.nodes[3], FlowNode)
    assert flow.nodes[3].subflow == technical_flow
    assert flow.nodes[3].subflow is not technical_flow


def test_routing_accepts_parallel_flow_target(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    summary = make_agent(
        "summary",
        outputs=(("summary_result", "string"),),
    )
    sentiment = make_agent(
        "sentiment",
        outputs=(("sentiment_result", "string"),),
    )
    parallel_workflow = parallel(name="parallel_test", steps=[summary, sentiment])
    parallel_target = _unwrap_flow(parallel_workflow)
    general = make_agent(
        "general",
        outputs=(
            ("summary_result", "string"),
            ("sentiment_result", "string"),
        ),
    )

    flow = _unwrap_flow(
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"review": parallel_workflow},
            default=general,
        )
    )

    target_node = flow.nodes[3]
    assert isinstance(target_node, FlowNode)
    assert target_node.subflow == parallel_target
    assert target_node.subflow is not parallel_target


def test_routing_default_can_reuse_route_target(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    billing = make_agent(
        "billing",
        outputs=(("resolution", "string"),),
    )

    flow = _unwrap_flow(
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"billing": billing},
            default=billing,
        )
    )

    route_node = flow.nodes[3]
    default_node = flow.nodes[4]
    assert isinstance(route_node, AgentNode)
    assert isinstance(default_node, AgentNode)
    assert route_node is not default_node
    assert route_node.agent == _unwrap_agent(billing)
    assert default_node.agent == _unwrap_agent(billing)


def test_routing_composes_llm_router_with_tool_and_llm_destinations(
    database_llm: LlmConfig,
) -> None:
    router = llm_node(
        name="classify_claim",
        llm=database_llm,
        prompt="Classify claim {{claim_id}} as status or coverage.",
        inputs={"claim_id": "string"},
        outputs={"request_type": "string"},
    )
    status = tool_node(
        name="claim_status",
        tool=plsql_tool(
            name="get_claim_status",
            procedure="CLAIMS_SERVICE.GET_STATUS",
            inputs={"claim_id": "string"},
            outputs={"answer": "string"},
        ),
    )
    coverage = llm_node(
        name="coverage_review",
        llm=database_llm,
        prompt="Explain coverage for claim {{claim_id}}.",
        inputs={"claim_id": "string"},
        outputs={"answer": "string"},
    )

    flow = _unwrap_flow(
        routing(
            router,
            name="routing_test",
            route_key="request_type",
            routes={"status": status},
            default=coverage,
        )
    )

    assert isinstance(flow.nodes[1], LlmNode)
    assert isinstance(flow.nodes[3], ToolNode)
    assert isinstance(flow.nodes[4], LlmNode)


def test_routing_requires_non_empty_routes(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    general = make_agent("general")

    with pytest.raises(PatternCompilationError, match="non-empty mapping"):
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={},
            default=general,
        )


def test_routing_rejects_invalid_components(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    general = make_agent("general")

    with pytest.raises(PatternCompilationError, match="router"):
        routing(  # type: ignore[arg-type]
            object(),
            name="routing_test",
            route_key="issue_type",
            routes={"general": general},
            default=general,
        )
    with pytest.raises(PatternCompilationError, match="destinations"):
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"general": object()},  # type: ignore[dict-item]
            default=general,
        )
    with pytest.raises(PatternCompilationError, match="default"):
        routing(  # type: ignore[arg-type]
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"general": general},
            default=object(),
        )


def test_routing_rejects_invalid_route_key_arguments(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    general = make_agent("general")

    with pytest.raises(PatternCompilationError, match="route_key"):
        routing(
            router,
            name="routing_test",
            route_key="",
            routes={"general": general},
            default=general,
        )
    with pytest.raises(PatternCompilationError, match="route values"):
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"": general},
            default=general,
        )


def test_routing_requires_declared_string_route_key(make_agent) -> None:
    missing = make_agent(
        "missing",
        outputs=(("category", "string"),),
    )
    integer = make_agent(
        "integer",
        outputs=(("issue_type", "integer"),),
    )
    general = make_agent("general")

    with pytest.raises(
        IncompatibleRoutingContractError,
        match="does not expose route key",
    ):
        routing(
            missing,
            name="routing_test",
            route_key="issue_type",
            routes={"general": general},
            default=general,
        )
    with pytest.raises(
        IncompatibleRoutingContractError,
        match="must be a string",
    ):
        routing(
            integer,
            name="routing_test",
            route_key="issue_type",
            routes={"general": general},
            default=general,
        )


def test_routing_rejects_missing_or_incompatible_target_inputs(make_agent) -> None:
    router = make_agent(
        "router",
        inputs=(("ticket_text", "string"),),
        outputs=(("issue_type", "string"),),
    )
    missing = make_agent(
        "missing",
        inputs=(("customer_id", "string"),),
        outputs=(("resolution", "string"),),
    )
    incompatible = make_agent(
        "incompatible",
        inputs=(("ticket_text", "integer"),),
        outputs=(("resolution", "string"),),
    )
    general = make_agent(
        "general",
        inputs=(("ticket_text", "string"),),
        outputs=(("resolution", "string"),),
    )

    with pytest.raises(
        IncompatibleRoutingContractError,
        match="not provided by the routing workflow input contract",
    ):
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"missing": missing},
            default=general,
        )
    with pytest.raises(
        IncompatibleRoutingContractError,
        match="incompatible input types",
    ):
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"incompatible": incompatible},
            default=general,
        )


def test_routing_rejects_incompatible_target_outputs(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    billing = make_agent(
        "billing",
        outputs=(("resolution", "string"),),
    )
    missing = make_agent(
        "missing",
        outputs=(("answer", "string"),),
    )
    incompatible = make_agent(
        "incompatible",
        outputs=(("resolution", "integer"),),
    )

    with pytest.raises(
        IncompatibleRoutingContractError,
        match="missing outputs.*additional outputs",
    ):
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"billing": billing, "missing": missing},
            default=billing,
        )
    with pytest.raises(
        IncompatibleRoutingContractError,
        match="incompatible output types",
    ):
        routing(
            router,
            name="routing_test",
            route_key="issue_type",
            routes={"billing": billing, "incompatible": incompatible},
            default=billing,
        )


def test_routing_construction_does_not_mutate_sources(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    billing = make_agent(
        "billing",
        outputs=(("resolution", "string"),),
    )
    general = make_agent(
        "general",
        outputs=(("resolution", "string"),),
    )
    raw_agents = [
        _unwrap_agent(router),
        _unwrap_agent(billing),
        _unwrap_agent(general),
    ]
    before = [value.model_copy(deep=True) for value in raw_agents]

    routing(
        router,
        name="routing_test",
        route_key="issue_type",
        routes={"billing": billing},
        default=general,
    )

    assert raw_agents == before


def test_routing_returns_opaque_executable(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    general = make_agent("general")

    workflow = routing(
        router,
        name="routing_test",
        route_key="issue_type",
        routes={"general": general},
        default=general,
    )
    assert type(workflow) is Executable
    assert isinstance(_unwrap_executable(workflow), Flow)
