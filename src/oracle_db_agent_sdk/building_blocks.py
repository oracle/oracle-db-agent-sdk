##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Create SDK building blocks for workflows executed by Oracle Database.

These factories construct workflow definitions without calling the database or an
LLM. Application code passes the returned objects to other SDK functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Literal, overload

from pyagentspec import Agent as PyAgentSpecAgent
from pyagentspec.flows.nodes import LlmNode as PyAgentSpecLlmNode
from pyagentspec.flows.nodes import ToolNode as PyAgentSpecToolNode
from pyagentspec.llms import DbmsVectorChainLlmConfig, LlmGenerationConfig
from pyagentspec.tools import BuiltinTool

from ._contracts import Contract, contract_properties, preserve_contract_schemas
from ._flow_status import fresh_copy
from ._json_types import JSONValue
from ._wrappers import (
    Executable,
    LlmConfig,
    Tool,
    _EmbeddedAgentTool,
    _unwrap_executable,
    _unwrap_llm,
    _unwrap_tool,
    _wrap_executable,
    _wrap_llm,
    _wrap_tool,
)
from .exceptions import BuildingBlockError

_PLSQL_EXECUTOR_NAME = "Oracle In-DB Agent Executor"
_PLSQL_TOOL_VERSION = "1.0"


def llm_config(
    *,
    provider: str,
    model_id: str,
    url: str,
    credential_name: str | None = None,
    host: Literal["local"] | None = None,
    transfer_timeout: int | None = None,
    generation: Mapping[str, JSONValue] | None = None,
) -> LlmConfig:
    """Configure an LLM for database-executed agents and workflow steps.

    Creates a reusable LLM configuration without contacting the provider. Pass the
    result to ``llm_node()`` or ``agent()``. Construction does not verify endpoint
    connectivity, credential access, or model availability; confirm these for
    your database deployment before execution.

    :param provider: LLM provider identifier supported by the database.
        Example: ``"openai"``.
    :type provider: str
    :param model_id: Model identifier supported by the configured provider.
    :type model_id: str
    :param url: Provider endpoint reachable from the database.
    :type url: str
    :param credential_name: Name of a database-managed credential, not an API key.
        Defaults to ``None``; supply a credential when the provider requires one.
    :type credential_name: str or None, optional
    :param host: Optional database host setting. The supported explicit value is
        ``"local"``. Defaults to ``None``.
    :type host: Literal["local"] or None, optional
    :param transfer_timeout: Optional database LLM transfer timeout setting.
        Defaults to ``None``, leaving the setting unspecified.
    :type transfer_timeout: int or None, optional
    :param generation: Model generation settings.
        Example: ``{"temperature": 0.2, "max_tokens": 512}``.
        Supported settings also depend on the database and provider.
        Defaults to ``None``.
    :type generation: Mapping[str, JSONValue] or None, optional

    Returns:
        ``LlmConfig`` containing the SDK-owned LLM configuration. This is a
        configuration object, not an executable workflow or a model response.

    Raises:
        BuildingBlockError: The LLM configuration is invalid."""
    try:
        generation_config = (
            LlmGenerationConfig.model_validate(dict(generation)) if generation is not None else None
        )
        config = DbmsVectorChainLlmConfig(
            name=f"{provider}-{model_id}",
            model_id=model_id,
            provider=provider,
            url=url,
            credential_name=credential_name,
            host=host,
            transfer_timeout=transfer_timeout,
            default_generation_parameters=generation_config,
        )
        return _wrap_llm(config)
    except Exception:
        # Validation diagnostics can contain supplied configuration values.
        raise BuildingBlockError("The LLM configuration is invalid.") from None


def plsql_tool(
    *,
    name: str,
    procedure: str,
    inputs: Contract,
    outputs: Contract,
    description: str | None = None,
    schema: str | None = None,
    package: str | None = None,
) -> Tool:
    """Describe an existing PL/SQL procedure as a workflow tool.

    Creates a tool definition for direct invocation with ``tool_node()`` or
    selection by an agent. It neither creates nor executes the PL/SQL procedure.
    The procedure must satisfy the database agent engine's tool interface and
    access requirements. The definition does not request interactive confirmation
    before tool execution.

    :param name: Tool name exposed in the workflow.
    :type name: str
    :param procedure: Procedure identifier understood by the database agent engine.
        Example: ``"GET_STATUS"`` with ``package="ORDER_SERVICE"``.
    :type procedure: str
    :param inputs: Mapping of input names to JSON type strings or JSON Schema
        mappings. Example: ``{"order_id": "string"}``.
    :type inputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param outputs: Mapping of output names to JSON type strings or JSON Schema
        mappings. Example: ``{"order_status": "string"}``.
    :type outputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param description: Human-readable description to help an agent select the
        tool. Defaults to ``None``.
    :type description: str or None, optional
    :param schema: Schema containing the procedure. Defaults to ``None``.
    :type schema: str or None, optional
    :param package: Package containing the procedure. Defaults to ``None``.
    :type package: str or None, optional

    Returns:
        ``Tool`` containing the SDK-owned procedure definition for ``tool_node()``
        or an agent's ``tools`` argument. The tool is not executable on its own.

    Raises:
        BuildingBlockError: The tool configuration or a contract is invalid.
    """
    if not isinstance(name, str) or not name.strip():
        raise BuildingBlockError("The PL/SQL tool name must be a non-empty string.")
    references = [procedure, *(value for value in (schema, package) if value is not None)]
    if any(not isinstance(value, str) or not value.strip() for value in references):
        raise BuildingBlockError("PL/SQL tool references must be non-empty strings.")
    try:
        configuration: dict[str, JSONValue] = {"procedure_name": procedure}
        if schema is not None:
            configuration["schema_name"] = schema
        if package is not None:
            configuration["package_name"] = package

        input_properties = contract_properties(inputs)
        output_properties = contract_properties(outputs)
        with preserve_contract_schemas(input_properties, output_properties):
            tool = BuiltinTool(
                name=name,
                description=description,
                inputs=input_properties,
                outputs=output_properties,
                requires_confirmation=False,
                tool_type="plsql_exec",
                executor_name=_PLSQL_EXECUTOR_NAME,
                tool_version=_PLSQL_TOOL_VERSION,
                configuration=configuration,
            )
        return _wrap_tool(tool)
    except BuildingBlockError:
        raise
    except Exception:
        raise BuildingBlockError("The PL/SQL tool configuration is invalid.") from None


@overload
def agent_tool(
    *,
    name: str,
    agent: Executable,
    description: str | None = None,
) -> _EmbeddedAgentTool: ...


@overload
def agent_tool(
    *,
    name: str,
    agent_name: str,
    schema: str,
    inputs: Contract,
    outputs: Contract,
    description: str | None = None,
) -> Tool: ...


def agent_tool(
    *,
    name: str,
    agent: Executable | None = None,
    agent_name: str | None = None,
    schema: str | None = None,
    inputs: Contract | None = None,
    outputs: Contract | None = None,
    description: str | None = None,
) -> Tool | _EmbeddedAgentTool:
    """Expose a Python-defined agent or a stored agent to an orchestrator.

    Pass ``agent=...`` to use an SDK agent or workflow and inherit its input/output
    contracts. You do not need to register a Python-defined agent in the database.
    Alternatively, supply schema, agent_name, inputs and outputs for an existing
    database agent. Check that your database version supports calling stored agents.
    Use either form with ``orchestrator()``, not ``agent()``. Defining a tool does
    not run it.

    :param name: Capability name presented to the model.
    :type name: str
    :param agent: SDK-owned executable to embed; mutually exclusive with stored
        agent identity and contracts. Its inputs and outputs are inherited.
    :type agent: Executable or None, optional
    :param agent_name: Name of the existing database agent.
    :type agent_name: str
    :param schema: Owner of the database agent.
    :type schema: str
    :param inputs: Named input types or JSON schemas for the capability.
    :type inputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param outputs: Named output types or JSON schemas for the capability.
    :type outputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param description: Description helping the model decide when to call it.
    :type description: str, optional

    Returns:
        An orchestrator capability embedding an executable or referencing a stored agent.

    Raises:
        BuildingBlockError: If the reference or contracts are invalid."""
    if not isinstance(name, str) or not name.strip():
        raise BuildingBlockError("The agent tool reference must have a non-empty name.")
    if description is not None and not isinstance(description, str):
        raise BuildingBlockError("The agent tool description must be text.")
    if agent is not None:
        if any(value is not None for value in (agent_name, schema, inputs, outputs)):
            raise BuildingBlockError("Choose an embedded agent or a stored agent reference.")
        try:
            return _EmbeddedAgentTool(name, description, fresh_copy(_unwrap_executable(agent)))
        except Exception:
            raise BuildingBlockError("The embedded agent tool is invalid.") from None
    if any(not isinstance(value, str) or not value.strip() for value in (agent_name, schema)):
        raise BuildingBlockError("The stored agent reference must contain non-empty names.")
    if inputs is None or outputs is None:
        raise BuildingBlockError("A stored agent reference requires input and output contracts.")
    try:
        input_properties = contract_properties(inputs)
        output_properties = contract_properties(outputs)
        with preserve_contract_schemas(input_properties, output_properties):
            return _wrap_tool(
                BuiltinTool(
                    name=name,
                    description=description,
                    inputs=input_properties,
                    outputs=output_properties,
                    requires_confirmation=False,
                    tool_type="agent_exec",
                    executor_name=_PLSQL_EXECUTOR_NAME,
                    tool_version=_PLSQL_TOOL_VERSION,
                    configuration={"schema_name": schema, "agent_name": agent_name},
                )
            )
    except BuildingBlockError:
        raise
    except Exception:
        raise BuildingBlockError("The stored agent tool configuration is invalid.") from None


def tool_node(*, name: str, tool: Tool) -> Executable:
    """Define a step that invokes a specified tool without agent selection.

    Creates an executable workflow step that inherits the tool's input and output
    contracts. Construction does not invoke the tool; it runs when the resulting
    workflow is executed in the database.

    :param name: Name of the workflow step. Example: ``"lookup_order"``.
    :type name: str
    :param tool: SDK-owned tool definition returned by ``plsql_tool()`` or the
        stored-reference form of ``agent_tool()``. Embedded agent tools require
        ``orchestrator()``.
        Create tools with the SDK tool functions.
    :type tool: Tool

    Returns:
        ``Executable`` representing a direct tool-call step. Pass it to
        ``execute()`` or include it in a composition pattern.

    Raises:
        BuildingBlockError: The tool or step configuration is invalid."""
    if not isinstance(name, str) or not name.strip():
        raise BuildingBlockError("The tool node name must be a non-empty string.")
    if isinstance(tool, _EmbeddedAgentTool):
        raise BuildingBlockError("Embedded agent tools must be used with orchestrator().")
    try:
        pyagentspec_tool = deepcopy(_unwrap_tool(tool))
        with preserve_contract_schemas(pyagentspec_tool):
            return _wrap_executable(
                PyAgentSpecToolNode(
                    name=name,
                    tool=pyagentspec_tool,
                )
            )
    except BuildingBlockError:
        raise
    except Exception:
        raise BuildingBlockError("The tool node is invalid.") from None


def llm_node(
    *,
    name: str,
    llm: LlmConfig,
    prompt: str,
    inputs: Contract,
    outputs: Contract,
) -> Executable:
    """Define a step that performs one direct LLM request.

    Creates an LLM step without sending a request. Input values are substituted
    into the prompt template during database execution. This step does not select
    tools; use ``agent()`` when tool selection is required.

    :param name: Name of the workflow step.
    :type name: str
    :param llm: SDK-owned LLM configuration returned by ``llm_config()``.
    :type llm: LlmConfig
    :param prompt: Prompt template using ``{{input_name}}`` placeholders for
        declared inputs. Example: ``"Explain this status: {{order_status}}"``.
    :type prompt: str
    :param inputs: Mapping of input names to JSON type strings or JSON Schema
        mappings. Example: ``{"order_status": "string"}``.
    :type inputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param outputs: Mapping of output names to JSON type strings or JSON Schema
        mappings. Example: ``{"answer": "string"}``.
    :type outputs: Mapping[str, str | Mapping[str, JSONValue]]

    Returns:
        ``Executable`` representing a direct LLM-request step for ``execute()``
        or a composition pattern.

    Raises:
        BuildingBlockError: The LLM, prompt, or contracts are invalid.
    """
    if not isinstance(name, str) or not name.strip():
        raise BuildingBlockError("The LLM node name must be a non-empty string.")
    try:
        input_properties = contract_properties(inputs)
        output_properties = contract_properties(outputs)
        with preserve_contract_schemas(input_properties, output_properties):
            return _wrap_executable(
                PyAgentSpecLlmNode(
                    name=name,
                    llm_config=_unwrap_llm(llm),
                    prompt_template=prompt,
                    inputs=input_properties,
                    outputs=output_properties,
                )
            )
    except BuildingBlockError:
        raise
    except Exception:
        raise BuildingBlockError("The LLM node is invalid.") from None


def agent(
    *,
    name: str,
    llm: LlmConfig,
    instruction: str,
    inputs: Contract,
    outputs: Contract,
    tools: Sequence[Tool] = (),
    description: str | None = None,
) -> Executable:
    """Define an LLM-driven agent with optional PL/SQL tool access.

    Creates an agent definition without running it. During database execution,
    the agent may select from the supplied tools; supplying a tool does not
    guarantee it will be called. Only expose tools suited to the task and database
    privileges. Interactive approval is not enabled.

    :param name: Name of the agent.
    :type name: str
    :param llm: SDK-owned LLM configuration returned by ``llm_config()``.
    :type llm: LlmConfig
    :param instruction: Instruction template using ``{{input_name}}`` placeholders
        for declared inputs. Example: ``"Answer the question: {{question}}"``.
    :type instruction: str
    :param inputs: Mapping of input names to JSON type strings or JSON Schema
        mappings. Example: ``{"question": "string"}``.
    :type inputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param outputs: Mapping of output names to JSON type strings or JSON Schema
        mappings. Example: ``{"answer": "string"}``.
    :type outputs: Mapping[str, str | Mapping[str, JSONValue]]
    :param tools: SDK-owned PL/SQL tool definitions available for agent selection.
        Supply objects returned by ``plsql_tool()``. Agent-backed tools from
        ``agent_tool()`` require ``orchestrator()``. Defaults to an empty tuple.
    :type tools: Sequence[Tool], optional
    :param description: Human-readable description of the agent's purpose.
        Defaults to ``None``.
    :type description: str or None, optional

    Returns:
        ``Executable`` representing an agent for ``execute()`` or a composition
        pattern. It is a workflow definition, not an execution result.

    Raises:
        BuildingBlockError: The agent configuration or a supplied building block
            is invalid, or an agent-backed tool is supplied.
    """
    return _build_agent(
        name=name,
        llm=llm,
        instruction=instruction,
        inputs=inputs,
        outputs=outputs,
        tools=tools,
        description=description,
        allow_agent_tools=False,
    )


def _build_agent(
    *,
    name: str,
    llm: LlmConfig,
    instruction: str,
    inputs: Contract,
    outputs: Contract,
    tools: Sequence[Tool],
    description: str | None,
    allow_agent_tools: bool,
) -> Executable:
    """Build an agent while enforcing the calling API's allowed tool kinds."""
    if not isinstance(name, str) or not name.strip():
        raise BuildingBlockError("The agent name must be a non-empty string.")
    try:
        if any(isinstance(tool, _EmbeddedAgentTool) for tool in tools):
            raise BuildingBlockError("Embedded agent tools require orchestrator().")
        pyagentspec_tools = [_unwrap_tool(tool) for tool in tools]
        if not allow_agent_tools and any(
            isinstance(tool, BuiltinTool) and tool.tool_type == "agent_exec"
            for tool in pyagentspec_tools
        ):
            raise BuildingBlockError(
                "agent() does not accept agent_tool() tools; use orchestrator() instead."
            )
        pyagentspec_tools = deepcopy(pyagentspec_tools)
        input_properties = contract_properties(inputs)
        output_properties = contract_properties(outputs)
        with preserve_contract_schemas(input_properties, output_properties, pyagentspec_tools):
            return _wrap_executable(
                PyAgentSpecAgent(
                    name=name,
                    description=description,
                    llm_config=_unwrap_llm(llm),
                    system_prompt=instruction,
                    inputs=input_properties,
                    outputs=output_properties,
                    tools=pyagentspec_tools,
                    human_in_the_loop=False,
                )
            )
    except BuildingBlockError:
        raise
    except Exception:
        raise BuildingBlockError("The agent is invalid.") from None
