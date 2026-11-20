##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Exercise example workflow construction without a database or LLM."""

import importlib.util
from pathlib import Path

import pytest
from pyagentspec import AgentSpecDeserializer
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import LlmNode, ParallelFlowNode, ToolNode

from oracle_db_agent_sdk._serialization import _serialize_workflow
from oracle_db_agent_sdk._wrappers import Executable

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_INPUTS = {
    "sequential_order_support": {"order_id": "ORDER-100"},
    "parallel_claim_review": {
        "claim_id": "CLAIM-100",
        "medical_notes": "Synthetic outpatient visit.",
        "policy_number": "POLICY-100",
        "transaction_history": "One consultation charge.",
    },
    "routing_support_ticket": {
        "ticket_text": "Review my invoice",
        "customer_id": "example-customer",
    },
    "loop_document_review": {"task": "Polish the draft.", "draft": "call support", "done": "N"},
    "orchestrator_case_support": {"case_id": "CASE-100"},
}


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.py")), ids=lambda path: path.stem)
def test_example_workflows_construct_and_serialize(path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    connection = object()
    inputs = EXAMPLE_INPUTS.get(path.stem, {})
    result = {"example": "result"}
    graphs = []

    def fake_execute(**kwargs):
        assert kwargs["connection"] is connection
        assert kwargs["inputs"] is inputs
        graph = AgentSpecDeserializer().from_json(_serialize_workflow(kwargs["workflow"]))
        assert {prop.title for prop in graph.inputs} == set(inputs)
        graphs.append(graph)
        return result

    monkeypatch.setattr(module, "execute", fake_execute, raising=False)
    for name, value in vars(module).items():
        if name.startswith("run_"):
            assert value(connection, inputs) is result
        elif isinstance(value, Executable):
            graphs.extend(
                AgentSpecDeserializer().from_json(spec) for spec in [_serialize_workflow(value)]
            )

    assert graphs
    if path.stem == "sequential_order_support":
        lookup = next(node for node in graphs[0].nodes if isinstance(node, ToolNode))
        assert lookup.tool.configuration["procedure_name"] == "GET_STATUS"
        assert lookup.tool.configuration["package_name"] == "ORDER_SERVICE"
    if path.stem == "parallel_claim_review":
        parallel = next(node for node in graphs[0].nodes if isinstance(node, ParallelFlowNode))
        configs = [
            node.llm_config
            for branch in parallel.subflows
            for node in branch.nodes
            if isinstance(node, LlmNode)
        ]
        assert len(configs) == 3
        assert len({config.id for config in configs}) == 1
    if path.stem == "routing_support_ticket":
        assert isinstance(graphs[0], Flow)
        assert {prop.title for prop in graphs[0].inputs} == {"ticket_text", "customer_id"}
        assert "{{customer_id}}" in graphs[0].nodes[1].prompt_template
