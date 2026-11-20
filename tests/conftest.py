##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Shared workflow factories that require no database or LLM access."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from oracle_db_agent_sdk import agent, llm_config
from oracle_db_agent_sdk._wrappers import Executable, LlmConfig

PropertyDefinition = tuple[str, str]
AgentFactory = Callable[
    [str, Sequence[PropertyDefinition], Sequence[PropertyDefinition] | None], Executable
]


@pytest.fixture
def database_llm() -> LlmConfig:
    """Build an LLM configuration with a deliberately non-routable test endpoint."""
    return llm_config(
        provider="openai",
        model_id="example-model",
        url="https://example.invalid/chat",
        credential_name="TEST_CREDENTIAL",
    )


@pytest.fixture
def make_agent(database_llm: LlmConfig) -> AgentFactory:
    """Return a public-API agent factory for testing configurable contracts."""

    def factory(
        identifier: str,
        inputs: Sequence[PropertyDefinition] = (("value", "string"),),
        outputs: Sequence[PropertyDefinition] | None = None,
    ) -> Executable:
        """Build an agent whose outputs default to the supplied input contract."""
        input_contract = dict(inputs)
        output_definitions = inputs if outputs is None else outputs
        output_contract = dict(output_definitions)
        placeholders = " ".join(f"{{{{{title}}}}}" for title, _ in inputs)
        return agent(
            name=identifier.upper(),
            llm=database_llm,
            instruction=f"Process {placeholders}".rstrip(),
            inputs=input_contract,
            outputs=output_contract,
        )

    return factory
