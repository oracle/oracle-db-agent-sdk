##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Public API for building and executing Oracle Database agent workflows."""

from ._json_types import JSONValue
from .building_blocks import agent, agent_tool, llm_config, llm_node, plsql_tool, tool_node
from .credentials import create_api_key_credential
from .exceptions import (
    AgentSpecSerializationError,
    BuildingBlockError,
    CredentialConfigurationError,
    CredentialCreationError,
    DatabaseCleanupError,
    DatabaseExecutionError,
    IncompatibleLoopContractError,
    IncompatibleParallelContractError,
    IncompatibleRoutingContractError,
    IncompatibleSequentialContractError,
    InputSerializationError,
    InvalidDatabaseResponseError,
    LoopExecutionError,
    LoopRuntimeSetupError,
    OracleDBAgentsError,
    OrchestrationExecutionError,
    PatternCompilationError,
)
from .executor import execute
from .patterns import loop, orchestrator, parallel, routing, sequential

__all__ = [
    "AgentSpecSerializationError",
    "BuildingBlockError",
    "CredentialConfigurationError",
    "CredentialCreationError",
    "DatabaseCleanupError",
    "DatabaseExecutionError",
    "IncompatibleLoopContractError",
    "IncompatibleParallelContractError",
    "IncompatibleRoutingContractError",
    "IncompatibleSequentialContractError",
    "InputSerializationError",
    "InvalidDatabaseResponseError",
    "JSONValue",
    "LoopRuntimeSetupError",
    "LoopExecutionError",
    "OracleDBAgentsError",
    "OrchestrationExecutionError",
    "PatternCompilationError",
    "agent",
    "agent_tool",
    "create_api_key_credential",
    "execute",
    "llm_config",
    "llm_node",
    "loop",
    "orchestrator",
    "parallel",
    "plsql_tool",
    "routing",
    "sequential",
    "tool_node",
]
