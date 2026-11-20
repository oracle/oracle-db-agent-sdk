##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Public SDK errors for configuration, compilation, and database operations.

Catch OracleDBAgentsError for SDK failures or a subclass for targeted recovery.
Database operation errors include the original database or driver diagnostic text.
Authoring and serialization errors may suppress underlying validation details.
"""


class OracleDBAgentsError(Exception):
    """Base class for the SDK's public exception hierarchy."""


class AgentSpecSerializationError(OracleDBAgentsError):
    """Raised when a workflow cannot be prepared for database execution."""


class BuildingBlockError(OracleDBAgentsError):
    """Raised when a public authoring building block cannot be constructed."""


class InputSerializationError(OracleDBAgentsError):
    """Raised when execution inputs cannot be serialized as JSON."""


class CredentialConfigurationError(OracleDBAgentsError):
    """Raised when credential configuration arguments are invalid."""


class PatternCompilationError(OracleDBAgentsError):
    """Raised when an authoring pattern cannot be compiled."""


class IncompatibleSequentialContractError(PatternCompilationError):
    """Raised when adjacent sequential steps have incompatible contracts."""


class IncompatibleParallelContractError(PatternCompilationError):
    """Raised when parallel branches have incompatible contracts."""


class IncompatibleRoutingContractError(PatternCompilationError):
    """Raised when routing components have incompatible contracts."""


class IncompatibleLoopContractError(PatternCompilationError):
    """Raised when a loop body cannot carry its outputs into its inputs."""


class LoopRuntimeSetupError(OracleDBAgentsError):
    """Retained for import compatibility; not raised by loop()."""


class LoopExecutionError(OracleDBAgentsError):
    """Raised when a loop still requests continuation at its iteration limit."""


class OrchestrationExecutionError(OracleDBAgentsError):
    """Raised when a manager selects an unknown capability."""


class DatabaseExecutionError(OracleDBAgentsError):
    """Raised when database execution fails or its completed response cannot be read."""


class DatabaseCleanupError(OracleDBAgentsError):
    """Raised when cursor cleanup fails after an otherwise successful operation."""


class InvalidDatabaseResponseError(OracleDBAgentsError):
    """Raised for an absent, blank, non-text, or invalid JSON database response."""


class CredentialCreationError(OracleDBAgentsError):
    """Raised when database credential creation fails."""
