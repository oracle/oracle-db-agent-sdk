##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Opt-in DB/model proof of embedded orchestration and actual result feedback.

Set SDK_ORCHESTRATOR_CONNECTION_FILE (user/password/dsn JSON),
SDK_ORCHESTRATOR_MODEL_FILE (llm_config keyword JSON referencing an existing
credential), and optionally SDK_FIXED_LOOP_CLIENT_LIB for thick mode. The schema
needs CREATE TABLE/PROCEDURE and agent execution privileges. No stored agents,
credentials, privileges, proxy settings or ACLs are created or changed.
SDK_ORCHESTRATOR_EVIDENCE_DIR optionally retains each exact SDK submission.
"""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import oracledb
import pytest

import oracle_db_agent_sdk.executor as sdk_executor
from oracle_db_agent_sdk import (
    agent,
    agent_tool,
    execute,
    llm_config,
    orchestrator,
    plsql_tool,
)

CONNECTION_FILE = os.environ.get("SDK_ORCHESTRATOR_CONNECTION_FILE")
MODEL_FILE = os.environ.get("SDK_ORCHESTRATOR_MODEL_FILE")
pytestmark = pytest.mark.skipif(
    not (CONNECTION_FILE and MODEL_FILE),
    reason="live database/model is opt-in",
)
RESULT = {"bumped": "number", "receipt": "string"}
FINAL = {"bumped": "number", "first_receipt": "string", "second_receipt": "string"}


@pytest.fixture(scope="module")
def connection():
    if lib_dir := os.environ.get("SDK_FIXED_LOOP_CLIENT_LIB"):
        oracledb.init_oracle_client(lib_dir=lib_dir)
    config = json.loads(Path(CONNECTION_FILE).read_text())
    with oracledb.connect(**{key: config[key] for key in ("user", "password", "dsn")}) as conn:
        conn.call_timeout = int(os.environ.get("SDK_LIVE_CALL_TIMEOUT_MS", "180000"))
        yield conn
        conn.rollback()


@pytest.fixture(scope="module")
def resources(connection):
    suffix = uuid4().hex[:12].upper()
    package, events = "SDK_EO_P_" + suffix, "SDK_EO_E_" + suffix
    created = []
    with connection.cursor() as cursor:
        try:
            cursor.execute(f"""create table {events}(event_id timestamp default systimestamp,
                phase varchar2(10),p_value number,prior_receipt varchar2(32),
                bumped number,receipt varchar2(32))""")
            created.append(("table", events))
            cursor.execute(f"""create package {package} authid current_user as
                procedure first_bump(p_value in number,bumped out number,receipt out varchar2);
                procedure second_bump(p_value in number,prior_receipt in varchar2,
                    bumped out number,receipt out varchar2);
                end;""")
            created.append(("package", package))
            cursor.execute(f"""create package body {package} as
                procedure first_bump(p_value in number,bumped out number,receipt out varchar2) is
                    pragma autonomous_transaction;
                begin
                    bumped:=p_value+1; receipt:=rawtohex(sys_guid());
                    insert into {events}(phase,p_value,bumped,receipt)
                        values('first',p_value,bumped,receipt); commit;
                end;
                procedure second_bump(p_value in number,prior_receipt in varchar2,
                    bumped out number,receipt out varchar2) is
                    pragma autonomous_transaction;
                begin
                    bumped:=p_value+1; receipt:=rawtohex(sys_guid());
                    insert into {events}(phase,p_value,prior_receipt,bumped,receipt)
                        values('second',p_value,prior_receipt,bumped,receipt); commit;
                end;
                end;""")
            assert not cursor.execute(
                "select line,text from user_errors where name=:name",
                name=package,
            ).fetchall()
            model = llm_config(**json.loads(Path(MODEL_FILE).read_text()))
            first_tool = plsql_tool(
                name="first_bump",
                package=package,
                procedure="first_bump",
                inputs={"p_value": "number"},
                outputs=RESULT,
                description="Increment p_value and return a fresh database receipt.",
            )
            first = agent(
                name="first_specialist",
                llm=model,
                instruction="Call first_bump exactly once with p_value={{p_value}}. "
                "Return its bumped and receipt fields exactly. Never invent receipts.",
                inputs={"p_value": "number"},
                outputs=RESULT,
                tools=[first_tool],
            )
            second = agent(
                name="second_specialist",
                llm=model,
                instruction="Call second_bump exactly once with p_value={{p_value}} and "
                "prior_receipt={{prior_receipt}}. Return its bumped and receipt exactly.",
                inputs={"p_value": "number", "prior_receipt": "string"},
                outputs=RESULT,
                tools=[
                    plsql_tool(
                        name="second_bump",
                        package=package,
                        procedure="second_bump",
                        inputs={"p_value": "number", "prior_receipt": "string"},
                        outputs=RESULT,
                    )
                ],
            )
            yield {
                "model": model,
                "events": events,
                "first_tool": first_tool,
                "tools": [
                    agent_tool(
                        name="first_specialist",
                        agent=first,
                        description="First increment and fresh receipt.",
                    ),
                    agent_tool(
                        name="second_specialist",
                        agent=second,
                        description="Second increment, carrying the earlier receipt.",
                    ),
                ],
            }
        finally:
            for kind, name in reversed(created):
                cursor.execute(f"drop {kind} {name}" + (" purge" if kind == "table" else ""))
            assert (
                cursor.execute(
                    "select count(*) from user_objects where object_name in (:p,:e)",
                    p=package,
                    e=events,
                ).fetchone()[0]
                == 0
            )


@pytest.fixture
def evidence(connection, resources, monkeypatch, request, tmp_path):
    with connection.cursor() as cursor:
        cursor.execute(f"delete from {resources['events']}")
    connection.commit()
    root = Path(os.environ.get("SDK_ORCHESTRATOR_EVIDENCE_DIR", str(tmp_path))) / request.node.name
    root.mkdir(parents=True, exist_ok=True)
    original = sdk_executor.execute_immediate
    submissions = []

    def observed(conn, *, agent_spec_json, inputs_json):
        submissions.append(agent_spec_json)
        (root / "agentspec.json").write_text(agent_spec_json)
        (root / "inputs.json").write_text(inputs_json)
        (root / "submission.sha256").write_text(
            hashlib.sha256(agent_spec_json.encode()).hexdigest()
        )
        raw = original(conn, agent_spec_json=agent_spec_json, inputs_json=inputs_json)
        (root / "raw-response.json").write_text(raw if isinstance(raw, str) else "null")
        return raw

    monkeypatch.setattr(sdk_executor, "execute_immediate", observed)

    def rows():
        with connection.cursor() as cursor:
            data = [
                dict(
                    zip(
                        ("phase", "p_value", "prior_receipt", "bumped", "receipt"), row, strict=True
                    )
                )
                for row in cursor.execute(
                    f"select phase,p_value,prior_receipt,bumped,receipt "
                    f"from {resources['events']} order by event_id"
                )
            ]
        (root / "events.json").write_text(json.dumps(data, indent=2))
        return data

    yield rows
    rows()
    assert len(submissions) == 1


def coordinate(connection, resources, instruction, *, tools=None, outputs=None):
    workflow = orchestrator(
        name="embedded_coordinator",
        llm=resources["model"],
        instruction=instruction,
        inputs={"p_value": "number"},
        outputs=FINAL if outputs is None else outputs,
        tools=resources["tools"] if tools is None else tools,
    )
    return execute(connection=connection, workflow=workflow, inputs={"p_value": 7})


@pytest.mark.parametrize("direct", [False, True], ids=["two_agents", "procedure_and_agent"])
def test_result_and_receipt_feed_the_next_capability(connection, resources, evidence, direct):
    first_name = "first_bump" if direct else "first_specialist"
    tools = [resources["first_tool"] if direct else resources["tools"][0], resources["tools"][1]]
    result = coordinate(
        connection,
        resources,
        f"Start with p_value={{{{p_value}}}}. First call {first_name} once with that value. "
        "Then call second_specialist once using the earlier bumped as p_value and receipt "
        "as prior_receipt. Obtain real receipts from both calls, then finish. Final bumped "
        "is the second bumped; first_receipt and second_receipt are their exact receipts. "
        "Do not repeat either capability.",
        tools=tools,
    )
    rows = evidence()
    assert [r["phase"] for r in rows] == ["first", "second"]
    assert [r["p_value"] for r in rows] == [7, 8]
    assert rows[1]["prior_receipt"] == rows[0]["receipt"]
    assert result == {
        "bumped": 9,
        "first_receipt": rows[0]["receipt"],
        "second_receipt": rows[1]["receipt"],
    }


def test_repeated_agent_beyond_eight_calls(connection, resources, evidence):
    calls = 9
    result = coordinate(
        connection,
        resources,
        f"Start with p_value={{{{p_value}}}}. Call first_specialist exactly {calls} times. "
        "Feed each returned bumped into the next p_value. Then finish and report the final bumped "
        "and receipt. Do not call second_specialist.",
        outputs=RESULT,
    )
    rows = evidence()
    assert [r["phase"] for r in rows] == ["first"] * calls
    assert [r["p_value"] for r in rows] == list(range(7, 7 + calls))
    assert result == {"bumped": 7 + calls, "receipt": rows[-1]["receipt"]}


def test_finish_without_delegation(connection, resources, evidence):
    result = coordinate(
        connection,
        resources,
        "Finish immediately without calling any capabilities. The final bumped is "
        "the input p_value={{p_value}} and receipt is the literal string unused.",
        outputs=RESULT,
    )
    assert result == {"bumped": 7, "receipt": "unused"}
    assert evidence() == []


def test_manager_can_return_to_an_earlier_agent(connection, resources, evidence):
    result = coordinate(
        connection,
        resources,
        "Start with p_value={{p_value}}. Call first_specialist once, then "
        "second_specialist once using that bumped and receipt as p_value and prior_receipt. "
        "Then call first_specialist again using the second bumped as p_value. "
        "Finish and return the third call's bumped and receipt exactly. "
        "Keep this ordered three-call task and its progress in task memory.",
        outputs=RESULT,
    )
    rows = evidence()
    assert [row["phase"] for row in rows] == ["first", "second", "first"]
    assert [row["p_value"] for row in rows] == [7, 8, 9]
    assert rows[1]["prior_receipt"] == rows[0]["receipt"]
    assert result == {"bumped": 10, "receipt": rows[-1]["receipt"]}
