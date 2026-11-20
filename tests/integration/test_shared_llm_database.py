##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Opt-in database tests for shared LLM configurations across parallel branches.

Use SDK_ORCHESTRATOR_CONNECTION_FILE and SDK_ORCHESTRATOR_MODEL_FILE for the
connection and existing model configuration, plus SDK_FIXED_LOOP_CLIENT_LIB
when thick mode is required. These tests create no database objects.
"""

import hashlib
import json
import os
from pathlib import Path

import oracledb
import pytest

import oracle_db_agent_sdk.executor as sdk_executor
from oracle_db_agent_sdk import agent, execute, llm_config, llm_node, parallel

CONNECTION_FILE = os.environ.get("SDK_ORCHESTRATOR_CONNECTION_FILE")
MODEL_FILE = os.environ.get("SDK_ORCHESTRATOR_MODEL_FILE")
pytestmark = pytest.mark.skipif(
    not (CONNECTION_FILE and MODEL_FILE), reason="live database/model is opt-in"
)


@pytest.fixture(scope="module")
def connection():
    if library := os.environ.get("SDK_FIXED_LOOP_CLIENT_LIB"):
        oracledb.init_oracle_client(lib_dir=library)
    settings = json.loads(Path(CONNECTION_FILE).read_text())
    with oracledb.connect(**{key: settings[key] for key in ("user", "password", "dsn")}) as conn:
        conn.call_timeout = 180000
        yield conn


@pytest.mark.parametrize("kind", ["llm", "agent"])
@pytest.mark.parametrize("shape", ["parallel", "nested", "nested_local"])
def test_shared_model_executes_in_parallel(connection, monkeypatch, kind, shape):
    settings = json.loads(Path(MODEL_FILE).read_text())
    shared = llm_config(**settings)

    def step(name, model):
        kwargs = dict(name=name, llm=model, inputs={"text": "string"}, outputs={name: "string"})
        if kind == "agent":
            return agent(
                instruction="Copy {{text}} exactly into the declared output field using "
                "the required output mechanism. Do not add words or punctuation.",
                **kwargs,
            )
        return llm_node(
            prompt="Copy {{text}} exactly. Return only plain text, with no JSON or quotes.",
            **kwargs,
        )

    workflow = parallel(name="inner", steps=[step("first", shared), step("second", shared)])
    expected = {"first": "orchid", "second": "orchid"}
    if shape != "parallel":
        other = llm_config(**settings) if shape == "nested_local" else shared
        workflow = parallel(name="outer", steps=[workflow, step("third", other)])
        expected["third"] = "orchid"

    if artifacts := os.environ.get("SDK_LIVE_ARTIFACTS"):
        folder = Path(artifacts) / "shared-llm" / f"{shape}-{kind}"
        folder.mkdir(parents=True, exist_ok=True)
        original = sdk_executor.execute_immediate

        def observed(conn, *, agent_spec_json, inputs_json):
            (folder / "agentspec.json").write_text(agent_spec_json)
            (folder / "inputs.json").write_text(inputs_json)
            (folder / "submission.json").write_text(
                json.dumps(
                    {
                        "sha256": hashlib.sha256(agent_spec_json.encode()).hexdigest(),
                    }
                )
            )
            raw = original(conn, agent_spec_json=agent_spec_json, inputs_json=inputs_json)
            (folder / "response.json").write_text(raw if isinstance(raw, str) else "null")
            return raw

        monkeypatch.setattr(sdk_executor, "execute_immediate", observed)

    result = execute(connection=connection, workflow=workflow, inputs={"text": "orchid"})
    assert result == expected
    assert connection.is_healthy()
