# Loop database tests

These tests run SDK loops on an agent-enabled database. Procedure invocation
counts verify that loops repeat as requested and respect their configured limits.

Use a dedicated test schema with CREATE TABLE, CREATE PROCEDURE, quota, and
DBMS_AI_AGENT execution privileges. Tests create uniquely named tables/packages,
record body invocation counts, and drop their objects afterwards. Tests own
commits and rollbacks on their dedicated connection. They do not change proxy
settings, ACLs, provider credentials, or existing application objects.

Store connection parameters in a protected JSON file containing `user`,
`password`, and `dsn`, then run:

```sh
SDK_FIXED_LOOP_CONNECTION_FILE=/path/to/protected-connection.json \
  python -m pytest -v tests/integration/test_fixed_loop_database.py
```

For thick mode, also set `SDK_FIXED_LOOP_CLIENT_LIB` to the Oracle client library
directory and configure the usual `LD_LIBRARY_PATH`, `TNS_ADMIN`, and Oracle
environment for that database. Never put a password in command-line arguments.

Coverage includes zero/multiple iterations, finishing exactly at the limit,
no extra body execution after the limit, body failures, sequential/routing/
parallel/nested composition, repeated entry into the same loop, and caller
transaction control. These tests use deterministic PL/SQL procedures and run
without a model provider. Without the connection-file environment
variable, the regular unit suite skips this module.

The suite also checks two independent parallel loops, exhaustion before a step
with different output fields, and Unicode feedback. Set
`SDK_FIXED_LOOP_ARTIFACTS` to a directory to retain test results and call records.
These artifacts contain test values only.
