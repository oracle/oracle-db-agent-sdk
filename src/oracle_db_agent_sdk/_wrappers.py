##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""SDK-owned values for model configurations, tools, and executables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

from pyagentspec.agenticcomponent import AgenticComponent
from pyagentspec.flows.node import Node
from pyagentspec.llms.llmconfig import LlmConfig as PyAgentSpecLlmConfig
from pyagentspec.tools.tool import Tool as PyAgentSpecTool

from .exceptions import BuildingBlockError

_T = TypeVar("_T")
_CONSTRUCTION_TOKEN = object()


class _OpaqueBuildingBlock(Generic[_T]):
    """Store an internal PyAgentSpec object without exposing public construction."""

    __slots__ = ("_value",)

    def __init__(self, value: _T, *, _token: object) -> None:
        """Require the factory token to prevent unsupported direct construction."""
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("Use an oracle_db_agent_sdk building-block function.")
        self._value = value


class LlmConfig(_OpaqueBuildingBlock[PyAgentSpecLlmConfig]):
    """SDK-owned LLM configuration returned by llm_config()."""


class Tool(_OpaqueBuildingBlock[PyAgentSpecTool]):
    """Opaque tool definition returned by a tool factory."""


RuntimeComponent: TypeAlias = AgenticComponent | Node


class Executable(_OpaqueBuildingBlock[RuntimeComponent]):
    """Opaque executable returned by a node, agent, or pattern factory."""


@dataclass(frozen=True)
class _EmbeddedAgentTool:
    """Compiler-owned snapshot of an executable exposed to an orchestrator."""

    name: str
    description: str | None
    component: RuntimeComponent


def _wrap_llm(value: PyAgentSpecLlmConfig) -> LlmConfig:
    """Store an LLM configuration behind the SDK's authoring interface."""
    return LlmConfig(value, _token=_CONSTRUCTION_TOKEN)


def _wrap_tool(value: PyAgentSpecTool) -> Tool:
    """Store a tool definition behind the SDK's authoring interface."""
    return Tool(value, _token=_CONSTRUCTION_TOKEN)


def _wrap_executable(value: RuntimeComponent) -> Executable:
    """Store a database component behind the public API."""
    return Executable(value, _token=_CONSTRUCTION_TOKEN)


def _unwrap_llm(value: LlmConfig) -> PyAgentSpecLlmConfig:
    """Recover an SDK LLM configuration or reject a foreign object."""
    if not isinstance(value, LlmConfig):
        raise BuildingBlockError("llm must be created by oracle_db_agent_sdk.llm_config().")
    return value._value


def _unwrap_tool(value: Tool) -> PyAgentSpecTool:
    """Recover an SDK tool definition or reject a foreign object."""
    if not isinstance(value, Tool):
        raise BuildingBlockError("tool must be created by an oracle_db_agent_sdk tool factory.")
    return value._value


def _unwrap_executable(value: Executable) -> RuntimeComponent:
    """Recover an SDK executable or reject a foreign object."""
    if not isinstance(value, Executable):
        raise BuildingBlockError(
            "workflow must be created by an oracle_db_agent_sdk building block."
        )
    return value._value
