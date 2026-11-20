# Oracle Database Agent SDK

**Build AI agents and workflows on Oracle Database — from Python.**

Define agent behavior, choose tools, and connect steps through a simple Python
API. Oracle Database runs the agent steps and tool calls through your
application's connection.

[Why this SDK?](#why-this-sdk) · [Quickstart](#quickstart) ·
[Examples](#examples) · [Documentation](#documentation) ·
[Contributing](#contributing)

---

<a id="why-this-sdk"></a>

## 💡 Why Oracle Database Agent SDK?

- **Build agents in Python.** Choose a model, define instructions and input/output
  fields, then run the agent from your application.
- **Compose workflows with built-in patterns.** Run steps in sequence or in
  parallel, select a branch, or repeat work in a bounded loop.
- **Execute workflows inside Oracle Database.** Let the database run your
  workflow, using existing PL/SQL procedures as tools to retrieve information
  and perform database operations.

### How workflows execute

Define agents and workflows in Python. The SDK uses AgentSpec to describe agents
and workflow steps for execution in Oracle Database through your application's
connection.

Call `execute()` to run the entire workflow in Oracle Database. See the
[API reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/python-api-reference.html)
for execution details.

---

<a id="quickstart"></a>

## 🚀 Quickstart: Build an AI Agent with Python

### Requirements

- **Python:** 3.10–3.14.
- **Oracle Database:** access to `DBMS_AI_AGENT.EXECUTE_IMMEDIATE` and permission
  to run agent workflows. Ask your database administrator to confirm availability.
- **LLM access:** a database-supported provider and model, an endpoint reachable
  from the database, and a database-managed credential when required.

### Installation

Create a virtual environment and install the SDK from PyPI:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install oracle-db-agent-sdk
```

Import the package as `oracle_db_agent_sdk`.

To install from source instead, run this command from the root of a local checkout
with the virtual environment activated:

```bash
python -m pip install .
```

### Build and run your first agent

This example uses OpenAI and an existing database credential named
`OPENAI_CREDENTIAL`. See
[configuration and credential setup](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/quickstart-build-an-ai-agent-with-python.html)
to prepare the credential or use another provider.

Replace the connection placeholders with your database settings:

```python
import oracledb

from oracle_db_agent_sdk import agent, execute, llm_config

support_llm = llm_config(
    provider="openai",
    model_id="gpt-5.4",
    url="https://api.openai.com/v1/chat/completions",
    credential_name="OPENAI_CREDENTIAL",
)

support_agent = agent(
    name="customer_support",
    llm=support_llm,
    instruction="Answer the customer's question: {{question}}",
    inputs={"question": "string"},
    outputs={"answer": "string"},
)

connection = oracledb.connect(
    user="<db_user>",
    password="<db_password>",
    dsn="<host>:<port>/<service_name>",
)
try:
    result = execute(
        connection=connection,
        workflow=support_agent,
        inputs={"question": "What details should I include in a support request?"},
    )
    print("Database response:", result)
finally:
    connection.close()
```

The script prints the full database response. `execute()` preserves the database
response structure, so check it before accessing fields. See
[result types](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/python-api-reference.html) and
[inspecting the result](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/quickstart-build-an-ai-agent-with-python.html).

[Read the full quickstart →](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/quickstart-build-an-ai-agent-with-python.html)

---

<a id="examples"></a>

## 🧩 See what you can build

Choose a task to explore in the examples below.

### Order support

Look up an order with a PL/SQL procedure, then use an LLM to turn its status
into a customer-facing explanation. This workflow chains a tool call and a
model response.

[View the order-support example →](https://github.com/oracle/oracle-db-agent-sdk/blob/main/examples/sequential_order_support.py)

### Independent claim reviews

Compose separate medical, policy, and fraud review prompts as parallel branches.
Each branch uses the inputs it needs and contributes an assessment to the
workflow's results.

[View the claim-review example →](https://github.com/oracle/oracle-db-agent-sdk/blob/main/examples/parallel_claim_review.py)

### Support-ticket routing

Classify a ticket and route it to a billing, security, or technical support
branch. A general-support fallback handles unmatched classifications.

[View the ticket-routing example →](https://github.com/oracle/oracle-db-agent-sdk/blob/main/examples/routing_support_ticket.py)

### Document review

Refine a draft through repeated reviews, carrying the draft and completion flag
between steps. Reviews run in the database. The loop raises an error if review
is still incomplete at its iteration limit.

[View the loop example →](https://github.com/oracle/oracle-db-agent-sdk/blob/main/examples/loop_document_review.py) ·
[Learn about loops →](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/python-api-reference.html)

---

<a id="documentation"></a>

## 📚 Documentation

- **[Getting started](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/quickstart-build-an-ai-agent-with-python.html)** — install, configure, and run your first agent.
- **[Python API reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/odbae/oracle-database-agent-sdk/python-api-reference.html)** — look up functions, parameters, and errors.
- **[Workflow examples](https://github.com/oracle/oracle-db-agent-sdk/tree/main/examples)** — explore the included workflows.

---

<a id="contributing"></a>

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidance.

## Security

Follow [Oracle's security reporting instructions](SECURITY.md) to report a vulnerability privately.

## License and notices

See [LICENSE.txt](LICENSE.txt) for the SDK license, [THIRD_PARTY_LICENSE.txt](THIRD_PARTY_LICENSE.txt)
for third-party licenses, and [NOTICE.txt](NOTICE.txt) for notices.
