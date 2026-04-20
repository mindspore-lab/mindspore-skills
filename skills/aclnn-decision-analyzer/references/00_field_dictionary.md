# 00 Field Dictionary

This document is the single source of truth for field definitions.
All other step documents must reference this file and must not redefine field meanings.

## Field Definitions

| Field | Type | Required | Produced In | Meaning |
| --- | --- | --- | --- | --- |
| `scope.user_request` | string | yes | Step 3 | Short restatement of user intent. |
| `scope.primary_apis` | string[] | yes | Step 1 | APIs explicitly requested by user. |
| `scope.related_apis` | string[] | yes | Step 1 | Family APIs auto-expanded by analysis. |
| `mapping.api` | string | yes | Step 1 | Frontend API linked to this mapping record. |
| `mapping.op_name` | string | yes | Step 1 | Internal operator name for decision unit. |
| `mapping.yaml_path` | string | yes | Step 1 | op yaml path used for dispatch and schema evidence. |
| `mapping.dispatch_ascend` | string | yes | Step 1 | `dispatch.Ascend` value; empty string is valid. |
| `mapping.api_def_ascend` | string | no | Step 1 | API-level Ascend setting from api_def. |
| `mapping.api_def_path` | string | no | Step 1 | API definition file path. |
| `mapping.api_def_py_method` | string | no | Step 1 | API-level py_method from api_def when available. |
| `evidence.op_name` | string | yes | Step 1 | Operator key for evidence block. |
| `evidence.config_evidence` | string[] | yes | Step 1 | Paths from `api_def` and `op_def`. |
| `evidence.pyboost_evidence` | string[] | yes | Step 1 | PyBoost-side backend evidence; `[]` is valid. |
| `evidence.kbk_evidence` | string[] | yes | Step 1 | KBK-side backend evidence; `[]` is valid. |
| `evidence.other_evidence` | string[] | no | Step 1 | Optional bprop/tests/docs evidence. |
| `path_decision.op_name` | string | yes | Step 2 | Operator key for decision record. |
| `path_decision.label` | enum | yes | Step 2 | `PATH1_AUTO`, `PATH2_CUSTOMIZE`, or `UNKNOWN`. |
| `path_decision.reason` | string | yes | Step 2 | One-sentence reason from decision rules. |
| `path_decision.interface_five_elements` | object | yes | Step 2 | Required five-element interface analysis block. |
| `risk_flags` | string[] | yes | Step 2/3 | Explicit risks or evidence gaps. |
| `change_scope.yaml` | string[] | yes | Step 2/3 | YAML files to check/change. |
| `change_scope.pyboost` | string[] | yes | Step 2/3 | PyBoost files to check/change. |
| `change_scope.kbk` | string[] | yes | Step 2/3 | KBK files to check/change. |
| `change_scope.bprop` | string[] | yes | Step 2/3 | Bprop files to check/change. |
| `change_scope.tests_docs` | string[] | yes | Step 2/3 | Test/doc files to check/change. |
| `next_checks` | string[] | yes | Step 3 | 1-3 concrete next actions. |
