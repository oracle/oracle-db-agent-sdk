# Agent Guide: oracle-db-agent-sdk

## Purpose

Help AI agents understand the Oracle Database Agent SDK and use its public Python
API to build and run agents and workflows.

## Setup

- Use Python 3.10–3.14. Install from a local checkout with
  `python -m pip install .` in a virtual environment.
- Import the package as `oracle_db_agent_sdk`.
- Confirm that the database user can run agent workflows and that the database
  can reach the chosen model provider.
- Depending on the user's environment, guide them through any required proxy or
  network ACL setup using the [Agent Engine getting started guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/get-started-with-oracle-ai-database-agent-engine.html).
  Obtain the user's explicit consent before configuring or changing proxy
  settings or network ACLs.
- Reference database-managed credentials by name in model configuration. Keep
  real API keys and database passwords out of generated code and workflow content.

## API Surface

- Model configuration: `llm_config`.
- Tool definitions: `plsql_tool`, `agent_tool`.
- Workflow steps: `tool_node`, `llm_node`.
- Agents: `agent`.
- Workflow patterns: `sequential`, `parallel`, `routing`, `orchestrator`, `loop`.
- Execution: `execute`.
- Credential setup: `create_api_key_credential`.
- Values and errors: `JSONValue`, `OracleDBAgentsError`, and specific exceptions
  listed in the [API reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/python-api-reference.html).

## Capability Workflows

- Model configuration: create an `llm_config()` using a provider, model, and
  endpoint supported by the database. Pass it to an agent or model step.
- Agents: use `agent()` to answer a question or perform a task, optionally
  choosing from PL/SQL tools. It does not accept `agent_tool()` references.
- Procedure calls: use `plsql_tool()` to describe an existing PL/SQL procedure.
  Pass it to `tool_node()` for a direct call or to an agent's `tools` list for
  model selection. The procedure must already exist and be accessible.
- Model steps: use `llm_node()` for a prompt-driven task without tool selection.
- Sequential workflows: use `sequential()` when each step consumes the previous
  step's outputs. Adjacent output/input names and types must match.
- Independent tasks: use `parallel(name=..., steps=[...])` to run branches and
  combine their results. List order does not determine execution order.
  Shared input names must have compatible types; output names must be unique.
- Routing: use `routing(router=..., name=..., route_key=..., routes=..., default=...)`
  to select a branch from fixed route values, with a
  default for unmatched values. Branches receive the original workflow inputs
  and must return the same output fields with compatible types.
- Coordination: use `orchestrator()` when the model needs to coordinate tools
  and sub-agents. Wrap a Python-defined agent or workflow with
  `agent_tool(name=..., agent=...)`; its input/output contracts are inherited.
  To use an existing database agent, supply its schema, name, and matching
  contracts instead. Supply nonempty inputs, outputs, and tools, with unique
  capability names other than `finish`. The manager receives actual previous results and can
  choose another capability or finish; all stages execute in one database call.
- Repeated work: use `loop()` when a task should update its state and repeat.
  The body's input/output names and types must match. Supply initial values
  as named inputs and set a string control field, continuation value, and
  iteration limit from 1 to 256. Repetitions execute in the database.
  Keep limits small to control resource use, especially for nested loops.
- Execution: call `execute(connection=..., workflow=..., inputs=...)` with an
  open `oracledb.Connection`. Defining a workflow does not run it.
- Credential setup: use `create_api_key_credential()` once if an API-key
  credential is needed and the connection user can create it. Reuse an existing
  credential by name instead of trying to create it again.

## Inputs and Results

- Define input and output fields with names and JSON type names, such as
  `inputs={"question": "string"}`. Supply actual values when running the workflow,
  such as `inputs={"question": "How can I contact support?"}`.
- Reference input fields in instructions and prompts with `{{field_name}}`.
- Build workflows from objects returned by public SDK functions.
- Inspect the returned Python value before accessing output fields. Ordinary
  execution preserves the database response structure.
- A loop returns its final named fields directly. Continued work after the
  iteration limit raises `LoopExecutionError`. Database execution failures raise
  `DatabaseExecutionError`; an absent database response raises
  `InvalidDatabaseResponseError`. Python does not coordinate iterations.
- An invalid orchestrator choice raises
  `OrchestrationExecutionError`, including inside enclosing compositions.
- Manage connection closure and transactions in application code.
- Validate external inputs and limit available tools and database privileges.
  See [input and tool security](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/python-api-reference.html).

## Validation

- Verify suggested code against the public function signatures and examples in
  this repository. Do not invent parameters or copy another SDK's API syntax.
- Check required input values, template variables, and the field names and types
  connecting workflow steps.

## Quick Start

- Start with the complete agent example in [README.md](README.md#quickstart).
- Use [Oracle quickstart](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/quickstart-build-an-ai-agent-with-python.html) for installation, model settings,
  and optional credential setup.
- Adapt the complete example to the user's task. Keep imports and explanations
  focused on what is needed to run it.

## Workflow Pointers

- [API reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/python-api-reference.html): signatures, parameters, results, and errors.
- [Order support](examples/sequential_order_support.py): a procedure call followed
  by a model explanation.
- [Claim review](examples/parallel_claim_review.py): independent review branches.
- [Ticket routing](examples/routing_support_ticket.py): classification followed
  by one selected support branch.
- [Document review](examples/loop_document_review.py): repeated draft improvement.
- [Case coordination](examples/orchestrator_case_support.py): a lookup tool and
  a Python-defined specialist embedded as an agent tool.

The example modules expose `run_*` functions taking a connection and inputs.
Configure their model settings and required database resources before calling
them; running the module alone does not execute its workflow.

## Troubleshooting

- Invalid configuration: check model settings, function arguments, template
  variables, and declared input/output fields.
- Incompatible workflow steps: check field names and types at each connection.
- Connection failures: check database login settings and network access.
- Database execution failures: check agent support, permissions, credential
  access, and network access from the database to the model provider.
- Unexpected responses: inspect the result instead of assuming an `answer` field.
- Before retrying a failed operation, check whether its tools already changed data.
