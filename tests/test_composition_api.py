##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for public pattern construction."""

import inspect

import pytest
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import FlowNode

import oracle_db_agent_sdk
from oracle_db_agent_sdk import (
    PatternCompilationError,
    parallel,
    routing,
    sequential,
)
from oracle_db_agent_sdk._wrappers import Executable, _unwrap_executable


def test_public_pattern_factories_use_lowercase_function_names() -> None:
    assert list(inspect.signature(sequential).parameters) == ["name", "steps"]
    assert list(inspect.signature(parallel).parameters) == ["name", "steps"]
    assert all(
        parameter.kind == inspect.Parameter.KEYWORD_ONLY
        for parameter in inspect.signature(parallel).parameters.values()
    )
    assert parallel.__name__ == "parallel"
    assert routing.__name__ == "routing"
    assert list(inspect.signature(routing).parameters) == [
        "router",
        "name",
        "route_key",
        "routes",
        "default",
    ]
    assert not hasattr(oracle_db_agent_sdk, "Sequential")
    assert not hasattr(oracle_db_agent_sdk, "SequentialAgent")
    assert not hasattr(oracle_db_agent_sdk, "Parallel")
    assert not hasattr(oracle_db_agent_sdk, "Routing")


@pytest.mark.parametrize("factory", [sequential, parallel])
@pytest.mark.parametrize("name", ["", " ", "\t\n", None, 123])
def test_step_patterns_reject_invalid_names(factory, name) -> None:
    with pytest.raises(PatternCompilationError, match="name must be a non-empty string"):
        factory(name=name, steps=[])


def test_sequential_requires_at_least_two_steps(make_agent) -> None:
    with pytest.raises(PatternCompilationError, match="at least two"):
        sequential(name="sequence", steps=[make_agent("a1")])


def test_sequential_rejects_non_building_block_steps(make_agent) -> None:
    with pytest.raises(PatternCompilationError, match="oracle_db_agent_sdk building block"):
        sequential(
            name="sequence",
            steps=[make_agent("a1"), object()],  # type: ignore[list-item]
        )


def test_sequential_returns_opaque_executable(make_agent) -> None:
    workflow = sequential(
        name="sequence",
        steps=[make_agent("a1"), make_agent("a2")],
    )

    assert type(workflow) is Executable
    assert isinstance(_unwrap_executable(workflow), Flow)


def test_sequential_accepts_an_eagerly_built_nested_flow(make_agent) -> None:
    a1, a2, a3 = make_agent("a1"), make_agent("a2"), make_agent("a3")
    inner = sequential(name="inner_sequence", steps=[a2, a3])
    outer = sequential(name="outer_sequence", steps=[a1, inner])
    inner_flow = _unwrap_executable(inner)
    outer_flow = _unwrap_executable(outer)

    assert isinstance(inner_flow, Flow)
    assert isinstance(outer_flow, Flow)
    assert isinstance(outer_flow.nodes[2], FlowNode)
    assert outer_flow.nodes[2].subflow == inner_flow
    assert outer_flow.nodes[2].subflow is not inner_flow


def test_patterns_reject_raw_pyagentspec_components(make_agent) -> None:
    first = make_agent("first")
    second = make_agent("second")
    router = make_agent("router", outputs=(("route", "string"),))
    raw_first = _unwrap_executable(first)

    with pytest.raises(PatternCompilationError, match="oracle_db_agent_sdk building block"):
        sequential(
            name="sequence",
            steps=[raw_first, second],  # type: ignore[list-item]
        )
    with pytest.raises(PatternCompilationError, match="oracle_db_agent_sdk building block"):
        parallel(name="parallel_test", steps=[raw_first, second])  # type: ignore[arg-type]
    with pytest.raises(PatternCompilationError, match="oracle_db_agent_sdk building block"):
        routing(  # type: ignore[arg-type]
            raw_first,
            name="routing_test",
            route_key="route",
            routes={"second": second},
            default=router,
        )
