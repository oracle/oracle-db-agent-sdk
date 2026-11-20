##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for the public authoring building blocks."""

from __future__ import annotations

import json
import traceback
from copy import deepcopy

import pytest
from pyagentspec import Agent as PyAgentSpecAgent
from pyagentspec import AgentSpecSerializer
from pyagentspec.flows.nodes import LlmNode as PyAgentSpecLlmNode
from pyagentspec.flows.nodes import ToolNode as PyAgentSpecToolNode
from pyagentspec.llms import DbmsVectorChainLlmConfig
from pyagentspec.tools import BuiltinTool

from oracle_db_agent_sdk import (
    BuildingBlockError,
    agent,
    agent_tool,
    llm_config,
    llm_node,
    plsql_tool,
    tool_node,
)
from oracle_db_agent_sdk._wrappers import (
    Executable,
    LlmConfig,
    Tool,
    _unwrap_executable,
    _unwrap_llm,
    _unwrap_tool,
)


@pytest.fixture
def database_llm() -> LlmConfig:
    """Construct representative LLM settings without contacting the provider."""
    return llm_config(
        provider="openai",
        model_id="gpt-4o",
        url="https://api.openai.com/v1/chat/completions",
        credential_name="OPENAI_CREDENTIAL",
        transfer_timeout=120,
        generation={"temperature": 0.2, "max_tokens": 512},
    )


@pytest.fixture
def add_tool() -> Tool:
    """Describe an illustrative PL/SQL tool without requiring it in a database."""
    return plsql_tool(
        name="add_numbers",
        description="Add two numbers and return their sum.",
        schema="APP_USER",
        package="MATH_PACKAGE",
        procedure="ADD_NUMBERS",
        inputs={"left": "number", "right": "number"},
        outputs={"sum": "number"},
    )


def test_llm_config_builds_dbms_vector_chain_config(database_llm: LlmConfig) -> None:
    config = _unwrap_llm(database_llm)

    assert isinstance(config, DbmsVectorChainLlmConfig)
    assert config.name == "openai-gpt-4o"
    assert config.model_id == "gpt-4o"
    assert config.provider == "openai"
    assert config.url == "https://api.openai.com/v1/chat/completions"
    assert config.credential_name == "OPENAI_CREDENTIAL"
    assert config.transfer_timeout == 120
    assert config.default_generation_parameters is not None
    assert config.default_generation_parameters.temperature == 0.2
    assert config.default_generation_parameters.max_tokens == 512


def test_llm_config_serializes_canonical_model_id_without_translation(
    database_llm: LlmConfig,
) -> None:
    document = json.loads(AgentSpecSerializer().to_json(_unwrap_llm(database_llm)))

    assert document["model_id"] == "gpt-4o"
    assert "model" not in document


def test_llm_config_supports_local_provider_without_credential() -> None:
    config = _unwrap_llm(
        llm_config(
            provider="ollama",
            model_id="llama3.2",
            url="http://localhost:11434/api/chat",
            host="local",
        )
    )

    assert config.host == "local"
    assert config.credential_name is None


def test_llm_config_wraps_pyagentspec_validation() -> None:
    with pytest.raises(BuildingBlockError, match="LLM configuration") as error:
        llm_config(
            provider="openai",
            model_id="gpt-4o",
            url="https://api.openai.com/v1/chat/completions",
        )

    assert error.value.__cause__ is None


def test_invalid_generation_value_is_absent_from_traceback() -> None:
    sensitive_value = "synthetic-sensitive-value"

    with pytest.raises(BuildingBlockError, match="LLM configuration") as caught:
        llm_config(
            provider="openai",
            model_id="example-model",
            url="https://example.invalid/chat",
            credential_name="TEST_CREDENTIAL",
            generation={"temperature": sensitive_value},
        )

    assert sensitive_value not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None


def test_invalid_contract_default_is_absent_from_traceback() -> None:
    sensitive_value = "synthetic-sensitive-value"

    with pytest.raises(BuildingBlockError, match="contract is invalid") as caught:
        plsql_tool(
            name="lookup",
            procedure="LOOKUP",
            inputs={"": {"type": "string", "default": sensitive_value}},
            outputs={},
        )

    assert sensitive_value not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (plsql_tool, {"procedure": "LOOKUP", "inputs": {}, "outputs": {}}),
        (tool_node, {}),
        (llm_node, {"prompt": "Explain.", "inputs": {}, "outputs": {}}),
        (agent, {"instruction": "Explain.", "inputs": {}, "outputs": {}}),
    ],
    ids=["plsql_tool", "tool_node", "llm_node", "agent"],
)
def test_factory_validation_values_are_absent_from_traceback(
    database_llm, add_tool, factory, kwargs
) -> None:
    sensitive_value = "synthetic-sensitive-value"
    arguments = dict(kwargs)
    arguments["name"] = {"private": sensitive_value}
    if factory is tool_node:
        arguments["tool"] = add_tool
    elif factory in (llm_node, agent):
        arguments["llm"] = database_llm

    with pytest.raises(BuildingBlockError) as caught:
        factory(**arguments)

    assert sensitive_value not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None


def test_plsql_tool_builds_builtin_tool(add_tool: Tool) -> None:
    tool = _unwrap_tool(add_tool)

    assert isinstance(tool, BuiltinTool)
    assert tool.name == "add_numbers"
    assert tool.description == "Add two numbers and return their sum."
    assert tool.tool_type == "plsql_exec"
    assert tool.executor_name == "Oracle In-DB Agent Executor"
    assert tool.tool_version == "1.0"
    assert tool.requires_confirmation is False
    assert tool.configuration == {
        "schema_name": "APP_USER",
        "package_name": "MATH_PACKAGE",
        "procedure_name": "ADD_NUMBERS",
    }
    assert [(prop.title, prop.type) for prop in tool.inputs or []] == [
        ("left", "number"),
        ("right", "number"),
    ]
    assert [(prop.title, prop.type) for prop in tool.outputs or []] == [
        ("sum", "number"),
    ]


def test_plsql_tool_omits_unspecified_schema_and_package() -> None:
    tool = _unwrap_tool(
        plsql_tool(
            name="normalize",
            procedure="NORMALIZE_ORDER",
            inputs={"order": "string"},
            outputs={"normalized_order": "string"},
        )
    )

    assert tool.configuration == {"procedure_name": "NORMALIZE_ORDER"}


def test_full_json_schema_contract_is_copied_and_titled() -> None:
    order_schema = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    }
    original = deepcopy(order_schema)

    tool = _unwrap_tool(
        plsql_tool(
            name="normalize",
            procedure="NORMALIZE_ORDER",
            inputs={"order": order_schema},
            outputs={"normalized_order": "string"},
        )
    )

    assert order_schema == original
    assert tool.inputs is not None
    assert tool.inputs[0].json_schema == {"title": "order", **original}


def test_tool_node_inherits_tool_contract(add_tool: Tool) -> None:
    node = _unwrap_executable(tool_node(name="add", tool=add_tool))

    assert isinstance(node, PyAgentSpecToolNode)
    assert node.tool == _unwrap_tool(add_tool)
    assert node.tool is not _unwrap_tool(add_tool)
    assert [prop.title for prop in node.inputs or []] == ["left", "right"]
    assert [prop.title for prop in node.outputs or []] == ["sum"]


def test_llm_node_builds_direct_llm_step(database_llm: LlmConfig) -> None:
    node = _unwrap_executable(
        llm_node(
            name="summarize",
            llm=database_llm,
            prompt="Explain {{sum}}.",
            inputs={"sum": "number"},
            outputs={"answer": "string"},
        )
    )

    assert isinstance(node, PyAgentSpecLlmNode)
    assert node.llm_config is _unwrap_llm(database_llm)
    assert node.prompt_template == "Explain {{sum}}."
    assert [prop.title for prop in node.inputs or []] == ["sum"]
    assert [prop.title for prop in node.outputs or []] == ["answer"]


def test_agent_builds_noninteractive_tool_calling_agent(
    database_llm: LlmConfig,
    add_tool: Tool,
) -> None:
    component = _unwrap_executable(
        agent(
            name="calculator",
            description="Perform database calculations.",
            llm=database_llm,
            instruction="Use add_numbers to add {{left}} and {{right}}.",
            tools=[add_tool],
            inputs={"left": "number", "right": "number"},
            outputs={"sum": "number"},
        )
    )

    assert isinstance(component, PyAgentSpecAgent)
    assert component.llm_config is _unwrap_llm(database_llm)
    assert component.system_prompt == "Use add_numbers to add {{left}} and {{right}}."
    assert component.tools == [_unwrap_tool(add_tool)]
    assert component.human_in_the_loop is False


def test_public_factories_return_only_opaque_package_types(
    database_llm: LlmConfig,
    add_tool: Tool,
) -> None:
    executable = tool_node(name="add", tool=add_tool)

    assert type(database_llm) is LlmConfig
    assert type(add_tool) is Tool
    assert type(executable) is Executable
    assert not isinstance(executable, PyAgentSpecToolNode)


def test_factories_reject_raw_pyagentspec_objects(database_llm: LlmConfig) -> None:
    raw_tool = BuiltinTool(
        name="raw",
        tool_type="plsql_exec",
        configuration={"procedure_name": "RAW_TOOL"},
    )

    with pytest.raises(BuildingBlockError, match="tool must be created"):
        tool_node(name="raw", tool=raw_tool)  # type: ignore[arg-type]

    with pytest.raises(BuildingBlockError, match="tool must be created"):
        agent(
            name="raw_agent",
            llm=database_llm,
            instruction="Run the raw tool.",
            inputs={},
            outputs={},
            tools=[raw_tool],  # type: ignore[list-item]
        )


def _named_block(kind, name, llm, tool):
    contract = {"value": "string"}
    factories = {
        "agent": lambda: agent(
            name=name, llm=llm, instruction="Process {{value}}.", inputs=contract, outputs=contract
        ),
        "llm_node": lambda: llm_node(
            name=name, llm=llm, prompt="Process {{value}}.", inputs=contract, outputs=contract
        ),
        "tool_node": lambda: tool_node(name=name, tool=tool),
        "plsql_tool": lambda: plsql_tool(
            name=name, procedure="EXAMPLE", inputs=contract, outputs=contract
        ),
        "agent_tool": lambda: agent_tool(
            name=name, agent_name="EXAMPLE", schema="APP", inputs=contract, outputs=contract
        ),
    }
    return factories[kind]()


@pytest.mark.parametrize("kind", ["agent", "llm_node", "tool_node", "plsql_tool", "agent_tool"])
@pytest.mark.parametrize("name", ["", " ", "\t\n", None, 123])
def test_building_blocks_reject_invalid_names(database_llm, add_tool, kind, name):
    with pytest.raises(BuildingBlockError):
        _named_block(kind, name, database_llm, add_tool)


@pytest.mark.parametrize("kind", ["agent", "llm_node", "tool_node", "plsql_tool", "agent_tool"])
def test_building_blocks_preserve_nonblank_names(database_llm, add_tool, kind):
    name = " example name "
    block = _named_block(kind, name, database_llm, add_tool)
    component = _unwrap_tool(block) if isinstance(block, Tool) else _unwrap_executable(block)
    assert component.name == name


@pytest.mark.parametrize(
    ("field", "value"),
    [("procedure", None)]
    + [
        (field, value)
        for field in ("procedure", "package", "schema")
        for value in (123, [], {}, "", " \t")
    ],
)
def test_plsql_tool_rejects_malformed_reference_shapes(field, value):
    arguments = {
        "name": "lookup",
        "procedure": "EXAMPLE",
        "inputs": {},
        "outputs": {},
        field: value,
    }
    with pytest.raises(BuildingBlockError, match="references must be non-empty strings"):
        plsql_tool(**arguments)


def test_plsql_tool_preserves_reference_text_and_omits_optional_none():
    procedure = '"My Package"."My Procedure"'
    result = plsql_tool(name="lookup", procedure=procedure, inputs={}, outputs={})
    assert _unwrap_tool(result).configuration == {"procedure_name": procedure}
    package = '"My Package"'
    schema = '"My Schema"'
    result = plsql_tool(
        name="lookup",
        procedure='"My Procedure"',
        package=package,
        schema=schema,
        inputs={},
        outputs={},
    )
    assert _unwrap_tool(result).configuration == {
        "procedure_name": '"My Procedure"',
        "package_name": package,
        "schema_name": schema,
    }
