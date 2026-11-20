##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Statements used for database operations."""

from typing import Final

EXECUTE_IMMEDIATE_PLSQL: Final[str] = """\
BEGIN
  :result := DBMS_AI_AGENT.EXECUTE_IMMEDIATE(
      :agent_spec,
      :inputs
  );
END;"""

CREATE_CREDENTIAL_PLSQL: Final[str] = """\
BEGIN
  DBMS_VECTOR_CHAIN.CREATE_CREDENTIAL(
      credential_name => :credential_name,
      params => JSON(:params_json)
  );
END;"""
