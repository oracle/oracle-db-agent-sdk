##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Compose SDK executables into database-executed workflows.

Patterns validate component contracts during construction. They do not execute
database calls or start Python threads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._contracts import Contract
from ._loop_compiler import build_loop_flow
from ._orchestrator_compiler import build_orchestrator_flow
from ._pattern_compiler import (
    build_parallel_flow,
    build_routing_flow,
    build_sequential_flow,
)
from ._wrappers import (
    Executable,
    LlmConfig,
    RuntimeComponent,
    Tool,
    _EmbeddedAgentTool,
    _unwrap_executable,
    _wrap_executable,
)
from .exceptions import PatternCompilationError


def _unwrap_pattern_component(
    executable: Executable,
    *,
    role: str,
) -> RuntimeComponent:
    """Require an SDK executable and report invalid inputs as pattern errors."""
    if not isinstance(executable, Executable):
        raise PatternCompilationError(
            f"{role} must be created by an oracle_db_agent_sdk building block."
        )
    return _unwrap_executable(executable)


def sequential(
    *,
    name: str,
    steps: Sequence[Executable],
) -> Executable:
    """Connect steps in order, passing each step's outputs to the next.

    Defines a workflow without executing its steps. Adjacent output and input
    field names must match exactly, with compatible types; extra outputs are not
    discarded. Earlier inputs are not automatically forwarded to later steps.
    Nested patterns follow the same contract rules.

    :param name: Name of the workflow. Must be a nonblank string.
        Example: ``"order_support"``.
    :type name: str
    :param steps: Ordered sequence of at least two SDK-owned executables returned
        by building-block or pattern functions. Example: ``[lookup_step, explain_step]``.
    :type steps: Sequence[Executable]

    Returns:
        ``Executable`` representing a sequential workflow with the first step's
        input contract and the last step's output contract.

    Raises:
        IncompatibleSequentialContractError: Adjacent output and input names
            differ, or matching properties have incompatible types.
        PatternCompilationError: Arguments are invalid or compilation fails."""
    if not isinstance(name, str) or not name.strip():
        raise PatternCompilationError("sequential name must be a non-empty string.")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise PatternCompilationError("sequential steps must be a sequence.")
    if len(steps) < 2:
        raise PatternCompilationError("sequential requires at least two steps.")
    components = [
        _unwrap_pattern_component(
            step,
            role="Each sequential step",
        )
        for step in steps
    ]
    return _wrap_executable(
        build_sequential_flow(components, name=name),
    )


def parallel(
    *,
    name: str,
    steps: Sequence[Executable],
) -> Executable:
    """Compose independent branches with combined inputs and outputs.

    Defines independent branches without executing them. Branches may share
    compatible inputs, but their output names must be unique. Branches do not
    feed each other. The database schedules execution; performance depends on
    available resources and the work each branch performs.

    :param name: Name identifying the parallel workflow. Must be a nonblank string.
        Example: ``"claim_review"``.
    :type name: str
    :param steps: Sequence of at least two SDK-owned executables.
        Example: ``[medical_review, policy_review]``. Each step becomes an
        independent branch and may be an agent, a node, or a composed workflow.
        List order does not determine execution order.
    :type steps: Sequence[Executable]

    Returns:
        ``Executable`` representing a parallel workflow. Its input contract
        combines the inputs required by the branches; its output contract
        combines their uniquely named outputs.

    Raises:
        IncompatibleParallelContractError: Shared inputs have incompatible types
            or branches declare duplicate output names.
        PatternCompilationError: Arguments are invalid or compilation fails."""
    if not isinstance(name, str) or not name.strip():
        raise PatternCompilationError("parallel name must be a non-empty string.")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise PatternCompilationError("parallel steps must be a sequence.")
    if len(steps) < 2:
        raise PatternCompilationError("parallel requires at least two steps.")

    components = [_unwrap_pattern_component(step, role="Each parallel step") for step in steps]
    return _wrap_executable(
        build_parallel_flow(components, name=name),
    )


def routing(
    router: Executable,
    *,
    name: str,
    route_key: str,
    routes: Mapping[str, Executable],
    default: Executable,
) -> Executable:
    """Run a router and then one matching destination or the default.

    Defines routing without executing any steps. The router produces a string
    used to select one destination. Destinations receive values from the original
    workflow inputs, not the router's outputs. Each destination may consume a
    compatible subset of the router's input contract. All destinations, including
    the default, must expose matching output names with compatible types.

    :param router: SDK-owned executable whose outputs include the string property
        identified by ``route_key``.
    :type router: Executable
    :param name: Nonblank name identifying the routing workflow.
        Example: ``"ticket_routing"``.
    :type name: str
    :param route_key: Name of the router output used to select a destination.
        Must be a nonempty string. Example: ``"issue_type"``.
    :type route_key: str
    :param routes: Nonempty mapping of nonempty route strings to SDK-owned
        executables. Example: ``{"billing": billing_agent, "technical": technical_agent}``.
    :type routes: Mapping[str, Executable]
    :param default: SDK-owned executable to run when the route value does not
        match any key in ``routes``. A default destination is required.
    :type default: Executable

    Returns:
        ``Executable`` representing a routing workflow with the router's input
        contract and the destinations' common output contract. During execution,
        only the selected destination runs after the router.

    Raises:
        IncompatibleRoutingContractError: The route key, destination inputs, or
            common destination outputs have incompatible contracts.
        PatternCompilationError: Arguments are invalid or compilation fails."""
    if not isinstance(name, str) or not name.strip():
        raise PatternCompilationError("routing name must be a non-empty string.")
    router_component = _unwrap_pattern_component(router, role="routing router")
    if not isinstance(route_key, str) or not route_key:
        raise PatternCompilationError("routing route_key must be a non-empty string.")
    if not isinstance(routes, Mapping) or not routes:
        raise PatternCompilationError("routing routes must be a non-empty mapping.")
    destination_components = {}
    for route_value, destination in routes.items():
        if not isinstance(route_value, str) or not route_value:
            raise PatternCompilationError("routing route values must be non-empty strings.")
        destination_components[route_value] = _unwrap_pattern_component(
            destination,
            role="routing destinations",
        )
    default_component = _unwrap_pattern_component(default, role="routing default")

    return _wrap_executable(
        build_routing_flow(
            router_component,
            name=name,
            route_key=route_key,
            routes=destination_components,
            default=default_component,
        ),
    )


def orchestrator(
    *,
    name: str,
    llm: LlmConfig,
    instruction: str,
    inputs: Contract,
    outputs: Contract,
    tools: Sequence[Tool | _EmbeddedAgentTool],
    description: str | None = None,
) -> Executable:
    """Create a model-driven coordinator with declared capabilities.

    The manager chooses a capability, receives its result, and chooses again or
    finishes. Use ``agent_tool(name=..., agent=...)`` for a Python-defined agent
    or workflow, and ``plsql_tool()`` for a procedure. All rounds run in the
    database. Capabilities may be repeated until the manager chooses to finish;
    the SDK does not impose a delegation-count limit.

    :param name: Name identifying the orchestrator.
    :type name: str
    :param llm: Database LLM configuration created by ``llm_config()``.
    :type llm: LlmConfig
    :param instruction: Model instruction template using ``{{input_name}}``
        placeholders for the declared inputs.
    :type instruction: str
    :param inputs: Nonempty named input types or JSON schemas.
    :type inputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param outputs: Nonempty named output types or JSON schemas.
    :type outputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param tools: At least one SDK-owned PL/SQL tool or agent-backed tool
        available to the model.
    :type tools: Sequence[Tool]
    :param description: Human-readable description of the coordinator.
    :type description: str, optional

    Returns:
        ``Executable`` usable directly or inside another pattern.

    Raises:
        PatternCompilationError: If configuration or capabilities are invalid."""
    if not isinstance(name, str) or not name.strip():
        raise PatternCompilationError("orchestrator name must be a non-empty string.")
    if not isinstance(instruction, str) or not instruction.strip():
        raise PatternCompilationError("orchestrator instruction must be a non-empty string.")
    if not isinstance(inputs, Mapping) or not inputs:
        raise PatternCompilationError("orchestrator requires a non-empty input contract.")
    if not isinstance(outputs, Mapping) or not outputs:
        raise PatternCompilationError("orchestrator requires a non-empty output contract.")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)) or not tools:
        raise PatternCompilationError("orchestrator requires at least one tool in a sequence.")
    return _wrap_executable(
        build_orchestrator_flow(
            name=name,
            llm=llm,
            instruction=instruction,
            inputs=inputs,
            outputs=outputs,
            tools=tools,
            description=description,
        )
    )


def loop(
    *,
    name: str,
    body: Executable,
    state_key: str,
    continue_value: str,
    max_iterations: int,
) -> Executable:
    """Repeat a body in the database up to a configured iteration limit.

    The body's input and output field names and types must match. The database
    passes outputs to the next iteration as inputs. Before each iteration,
    state_key is compared with continue_value. A different value finishes the
    loop, including before the first run. Finishing on the last permitted
    iteration succeeds. Requesting another iteration after max_iterations
    raises LoopExecutionError. Database failures raise DatabaseExecutionError;
    an absent response raises InvalidDatabaseResponseError.

    Keep limits small to control resource use, especially for nested loops.
    The caller owns the connection and transaction.

    :param name: Name identifying the loop workflow.
    :type name: str
    :param body: Package-owned node, agent, or pattern to repeat.
    :type body: Executable
    :param state_key: String field in the body contract controlling repetition.
    :type state_key: str
    :param continue_value: Nonempty string value that means continue.
    :type continue_value: str
    :param max_iterations: Maximum number of body executions, from 1 to 256.
    :type max_iterations: int

    Returns:
        ``Executable`` accepting and returning the body's named state fields.

    Raises:
        IncompatibleLoopContractError: If feedback contracts or control field differ.
        PatternCompilationError: If loop configuration or body is invalid.

    Example::

        workflow = loop(name="review", body=review_step, state_key="done",
                        continue_value="N", max_iterations=10)
        result = execute(connection=connection, workflow=workflow,
                         inputs={"done": "N", "counter": 0})

    Runtime value validation follows the database engine's rules. Python does
    not inspect intermediate state, repeat calls, retry, commit, or roll back."""
    if not isinstance(name, str) or not name.strip():
        raise PatternCompilationError("loop name must be a non-empty string.")
    component = _unwrap_pattern_component(body, role="loop body")
    if not isinstance(state_key, str) or not state_key:
        raise PatternCompilationError("loop state_key must be a non-empty string.")
    if not isinstance(continue_value, str) or not continue_value:
        raise PatternCompilationError("loop continue_value must be a non-empty string.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or not 1 <= max_iterations <= 256
    ):
        raise PatternCompilationError("loop max_iterations must be an integer from 1 to 256.")
    return _wrap_executable(
        build_loop_flow(
            component,
            name=name,
            state_key=state_key,
            continue_value=continue_value,
            max_iterations=max_iterations,
        ),
    )
