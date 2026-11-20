##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Define a standalone calculator agent that calls a PL/SQL tool.

Use orchestrator() with agent_tool() when a coordinator needs to call sub-agents.

Running this module constructs definitions only; it does not call a database or
LLM. Before executing them, configure a supported provider/model, provision the
named database credential, and supply the referenced PL/SQL tool procedure with
the database agent engine's expected interface. Schema, package, credential, and
provider settings here are illustrative, not resources created by this example.
"""

from oracle_db_agent_sdk import (
    agent,
    llm_config,
    llm_node,
    plsql_tool,
    tool_node,
)

database_llm = llm_config(
    provider="openai",
    model_id="gpt-5.4",
    url="https://api.openai.com/v1/chat/completions",
    credential_name="OPENAI_CREDENTIAL",
)

add_numbers = plsql_tool(
    name="add_numbers",
    description="Add two numbers and return their sum.",
    schema="APP_USER",
    package="MATH_PACKAGE",
    procedure="ADD_NUMBERS",
    inputs={"left": "number", "right": "number"},
    outputs={"sum": "number"},
)

add_step = tool_node(
    name="add",
    tool=add_numbers,
)

summarize_step = llm_node(
    name="summarize",
    llm=database_llm,
    prompt="Explain the calculation result {{sum}}.",
    inputs={"sum": "number"},
    outputs={"answer": "string"},
)

calculator = agent(
    name="calculator",
    description="Perform calculations using database tools.",
    llm=database_llm,
    instruction="Use add_numbers to add {{left}} and {{right}}.",
    tools=[add_numbers],
    inputs={"left": "number", "right": "number"},
    outputs={"sum": "number"},
)
