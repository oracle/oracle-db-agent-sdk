##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Review a document with bounded iteration executed inside the database.

Before calling run_document_review(), replace the illustrative model and endpoint,
provision REVIEW_LLM_CREDENTIAL, and grant the required body execution privileges.
The caller supplies the connection. Importing this module does not execute the workflow.
"""

from __future__ import annotations

from collections.abc import Mapping

import oracledb

from oracle_db_agent_sdk import JSONValue, execute, llm_config, llm_node, loop


def run_document_review(
    connection: oracledb.Connection, inputs: Mapping[str, JSONValue]
) -> JSONValue:
    """Improve a draft until complete, with at most five body executions.

    Reviews run in the database. The connection user needs permission to run
    the review step and access its model credential.

    :param connection: Application-owned connection to the agent-enabled database.
    :type connection: oracledb.Connection
    :param inputs: Mapping with ``task``, ``draft``, and ``done`` string fields.
        ``done='N'`` requests continued review.
    :type inputs: Mapping[str, JSONValue]

    Returns:
        ``JSONValue`` containing the final named fields on successful completion.

    Raises:
        OracleDBAgentsError: If configuration or execution fails,
            including failure to finish within the iteration bound.
    """
    state_fields = {"task": "string", "draft": "string", "done": "string"}
    review_step = llm_node(
        name="review_step",
        llm=llm_config(
            provider="openai",
            model_id="gpt-5",
            url="https://LLM_URL",
            credential_name="REVIEW_LLM_CREDENTIAL",
        ),
        prompt=(
            "Review {{draft}} for task {{task}}. Previous completion flag: {{done}}. "
            "Improve the draft and set done to 'Y' if complete, else 'N'. "
            "Return one JSON object with the string fields task, draft, and done. "
            "Copy task exactly and put the revised text in draft. "
            "Return no other text or Markdown formatting."
        ),
        inputs=state_fields,
        outputs=state_fields,
    )
    workflow = loop(
        name="review_until_done",
        body=review_step,
        state_key="done",
        continue_value="N",
        max_iterations=5,
    )
    return execute(connection=connection, workflow=workflow, inputs=inputs)
