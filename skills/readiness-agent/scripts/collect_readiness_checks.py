#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Optional


def describe_probe_source(probe_source: Optional[str]) -> str:
    mapping = {
        "selected_env": "selected environment",
        "explicit_env": "selected environment",
        "workspace_env": "selected environment",
        "selected_env_missing_python": "selected environment",
        "workspace_env_missing": "workspace-local environment",
        "explicit_python": "selected Python interpreter",
    }
    return mapping.get(probe_source or "", "selected Python interpreter")


def make_check(
    check_id: str,
    status: str,
    summary: str,
    *,
    category_hint: Optional[str] = None,
    severity: Optional[str] = None,
    remediable: Optional[bool] = None,
    remediation_owner: Optional[str] = None,
    revalidation_scope: Optional[List[str]] = None,
    evidence: Optional[List[str]] = None,
    **extra: object,
) -> dict:
    payload = {
        "id": check_id,
        "status": status,
        "summary": summary,
        "evidence": evidence or [],
    }
    if category_hint is not None:
        payload["category_hint"] = category_hint
    if severity is not None:
        payload["severity"] = severity
    if remediable is not None:
        payload["remediable"] = remediable
    if remediation_owner is not None:
        payload["remediation_owner"] = remediation_owner
    if revalidation_scope is not None:
        payload["revalidation_scope"] = revalidation_scope
    payload.update(extra)
    return payload


def resolve_runtime_package_names(runtime_layer: dict, missing_runtime: List[str]) -> List[str]:
    profile = runtime_layer.get("implicit_dependency_profile") or []
    mapping = {}
    for item in profile:
        if not isinstance(item, dict):
            continue
        import_name = str(item.get("import_name") or "").strip()
        package_name = str(item.get("package_name") or "").strip()
        if import_name and package_name:
            mapping[import_name] = package_name

    package_names: List[str] = []
    for import_name in missing_runtime:
        package_name = mapping.get(import_name, import_name)
        if package_name not in package_names:
            package_names.append(package_name)
    return package_names


ASSET_METADATA_FIELDS = (
    "asset_kind",
    "asset_provider",
    "asset_repo_id",
    "asset_repo_type",
    "asset_local_path",
    "dataset_split",
    "template_path",
    "example_recipe_id",
    "reference_transformers_version",
)


def asset_check_fields(asset: dict, key: str) -> dict:
    payload = {
        "asset_kind": key,
        "asset_local_path": asset.get("path"),
    }
    if asset.get("asset_provider"):
        payload["asset_provider"] = asset.get("asset_provider")
    if asset.get("repo_id"):
        payload["asset_repo_id"] = asset.get("repo_id")
    if asset.get("repo_type"):
        payload["asset_repo_type"] = asset.get("repo_type")
    if asset.get("dataset_split"):
        payload["dataset_split"] = asset.get("dataset_split")
    if asset.get("template_path"):
        payload["template_path"] = asset.get("template_path")
    if asset.get("example_recipe_id"):
        payload["example_recipe_id"] = asset.get("example_recipe_id")
    if asset.get("reference_transformers_version"):
        payload["reference_transformers_version"] = asset.get("reference_transformers_version")
    return payload


def collect_checks(target: dict, closure: dict) -> List[dict]:
    checks: List[dict] = []
    target_type = target.get("target_type") or "unknown"
    framework_layer = closure.get("layers", {}).get("framework", {})
    framework_path = framework_layer.get("framework_path", "unknown")
    runtime_layer = closure.get("layers", {}).get("runtime_dependencies", {})
    system = closure.get("layers", {}).get("system", {})
    python_env = closure.get("layers", {}).get("python_environment", {})
    remote_assets = closure.get("layers", {}).get("remote_assets", {})
    workspace = closure.get("layers", {}).get("workspace_assets", {})
    selected_env_root = python_env.get("selected_env_root")
    selected_python = python_env.get("selected_python")
    selection_status = python_env.get("selection_status")
    selection_reason = python_env.get("selection_reason")
    probe_source = python_env.get("probe_source")
    probe_python_path = python_env.get("probe_python_path")
    selected_python_ready = bool(probe_python_path)

    if target_type in {"training", "inference"}:
        checks.append(
            make_check(
                "target-stability",
                "ok",
                f"Execution target is resolved as {target_type}.",
                evidence=[f"target_type={target_type}"],
            )
        )
    else:
        checks.append(
            make_check(
                "target-stability",
                "warn",
                "Execution target remains ambiguous.",
                category_hint="unknown",
                severity="medium",
                evidence=["target_type is unresolved"],
            )
        )

    if system.get("requires_ascend"):
        ascend_env_script_path = system.get("ascend_env_script_path") or "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        probe_env_source = system.get("probe_env_source")
        probe_env_error = system.get("probe_env_error")
        ascend_env_active = bool(system.get("ascend_env_active"))
        if not system.get("device_paths_present"):
            checks.append(
                make_check(
                    "system-device",
                    "block",
                    "Ascend device visibility is missing for the selected target.",
                    category_hint="system",
                    severity="fatal",
                    remediable=False,
                    remediation_owner="manual-system",
                    revalidation_scope=["system"],
                    evidence=["/dev/davinci* not found"],
                )
            )
        else:
            checks.append(
                make_check(
                    "system-device",
                    "ok",
                    "Ascend device visibility evidence is present.",
                    evidence=["/dev/davinci* exists"],
                )
            )

        evidence = []
        if ascend_env_script_path:
            evidence.append(f"set_env.sh={ascend_env_script_path}")
        if system.get("ascend_env_selection_source"):
            evidence.append(f"selection_source={system.get('ascend_env_selection_source')}")
        if system.get("cann_path_input"):
            evidence.append(f"cann_path_input={system.get('cann_path_input')}")
        if probe_env_source:
            evidence.append(f"probe_env_source={probe_env_source}")
        if probe_env_error:
            evidence.append(f"probe_env_error={probe_env_error}")

        if ascend_env_active:
            checks.append(
                make_check(
                    "system-ascend-env",
                    "ok",
                    "Ascend runtime environment is already active.",
                    evidence=evidence or ["current Ascend environment is active"],
                )
            )
        elif not system.get("ascend_env_script_present"):
            checks.append(
                make_check(
                    "system-ascend-env",
                    "block",
                    "Ascend environment sourcing script is missing.",
                    category_hint="system",
                    severity="fatal",
                    remediable=False,
                    remediation_owner="manual-system",
                    revalidation_scope=["system"],
                    evidence=[f"{ascend_env_script_path} missing"],
                )
            )
        elif probe_env_source == "sourced_script" and not probe_env_error:
            checks.append(
                make_check(
                    "system-ascend-env",
                    "ok",
                    "Ascend environment script sourced successfully for runtime probing.",
                    evidence=evidence,
                )
            )
        else:
            checks.append(
                make_check(
                    "system-ascend-env",
                    "block",
                    "Ascend environment script is present but unusable for runtime probing.",
                    category_hint="system",
                    severity="fatal",
                    remediable=False,
                    remediation_owner="manual-system",
                    revalidation_scope=["system"],
                    evidence=evidence,
                )
            )

    uv_available = python_env.get("tooling", {}).get("uv_available", False)
    if uv_available:
        checks.append(
            make_check(
                "python-uv",
                "ok",
                "uv is directly resolvable.",
                evidence=[str(python_env.get("tooling", {}).get("uv_path"))],
            )
        )
    else:
        checks.append(
            make_check(
                "python-uv",
                "block",
                "uv is missing from the selected execution path.",
                category_hint="env",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["tool-resolution", "python-environment"],
                evidence=["uv_path not found"],
            )
        )

    if selection_status == "selected" and probe_python_path:
        checks.append(
            make_check(
                "python-selected-python",
                "ok",
                "Selected Python is resolved and probeable.",
                evidence=[
                    f"selected_python={selected_python}",
                    f"probe_python_path={probe_python_path}",
                    f"probe_source={probe_source}",
                ],
            )
        )
    else:
        evidence = []
        if selected_env_root:
            evidence.append(f"selected_env_root={selected_env_root}")
        if selected_python:
            evidence.append(f"selected_python={selected_python}")
        if probe_source:
            evidence.append(f"probe_source={probe_source}")
        if selection_reason:
            evidence.append(f"selection_reason={selection_reason}")
        checks.append(
            make_check(
                "python-selected-python",
                "block",
                "Selected Python is unavailable or unusable for readiness checks.",
                category_hint="env",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["python-environment", "framework"],
                evidence=evidence or ["selected_python is unresolved"],
            )
        )
    if not selected_env_root:
        evidence = ["workspace-local selected environment is unresolved", "system_python_fallback=forbidden"]
        if python_env.get("selection_source"):
            evidence.append(f"selection_source={python_env.get('selection_source')}")
        checks.append(
            make_check(
                "python-selected-env",
                "block",
                "No usable workspace-local Python environment is available for readiness checks. Do not use system python or pip for remediation.",
                category_hint="env",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["python-environment", "framework"],
                evidence=evidence,
            )
        )

    if framework_path in {"mindspore", "pta"}:
        checks.append(
            make_check(
                "framework-path",
                "ok",
                f"Framework path is resolved as {framework_path}.",
                evidence=[f"framework_path={framework_path}"],
            )
        )
    elif framework_path == "mixed":
        checks.append(
            make_check(
                "framework-path",
                "warn",
                "Framework evidence is mixed and requires manual confirmation.",
                category_hint="unknown",
                severity="medium",
                evidence=["framework_path=mixed"],
            )
        )
    else:
        checks.append(
            make_check(
                "framework-path",
                "warn",
                "Framework path is not yet resolved.",
                category_hint="unknown",
                severity="medium",
                evidence=["framework_path=unknown"],
            )
        )

    required_framework = framework_layer.get("required_packages", [])
    framework_probes = framework_layer.get("import_probes", {})
    framework_probe_source = framework_layer.get("probe_source") or probe_source or "selected_env"
    framework_probe_label = describe_probe_source(framework_probe_source)
    framework_probe_error = framework_layer.get("probe_error")
    framework_smoke = framework_layer.get("smoke_prerequisite") or {}
    missing_framework = [pkg for pkg in required_framework if not framework_probes.get(pkg, False)]
    if not selected_python_ready and required_framework:
        checks.append(
            make_check(
                "framework-importability",
                "warn",
                "Framework importability is deferred until a workspace-local Python environment is resolved. Do not install framework packages into system Python.",
                category_hint="env",
                severity="medium",
                evidence=[f"required_framework={','.join(required_framework)}", "system_python_fallback=forbidden"],
            )
        )
    elif required_framework and not missing_framework:
        checks.append(
            make_check(
                "framework-importability",
                "ok",
                f"Required framework packages are importable in the {framework_probe_label}.",
                evidence=[f"probe_source={framework_probe_source}", *required_framework],
            )
        )
    elif missing_framework:
        evidence = [f"probe_source={framework_probe_source}", *missing_framework]
        if framework_probe_error:
            evidence.append(f"probe_error={framework_probe_error}")
        checks.append(
            make_check(
                "framework-importability",
                "block",
                f"Required framework packages are unavailable in the {framework_probe_label}: {', '.join(missing_framework)}.",
                category_hint="framework",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["framework", "task-smoke"],
                evidence=evidence,
            )
        )

    smoke_status = framework_smoke.get("status")
    smoke_details = framework_smoke.get("details") or []
    smoke_error = framework_smoke.get("error")
    if not selected_python_ready and framework_path in {"mindspore", "pta", "mixed"}:
        checks.append(
            make_check(
                "framework-smoke-prerequisite",
                "skipped",
                "Framework smoke prerequisite is deferred until a workspace-local Python environment is resolved.",
                evidence=["system_python_fallback=forbidden"],
            )
        )
    elif smoke_status == "passed":
        checks.append(
            make_check(
                "framework-smoke-prerequisite",
                "ok",
                f"Framework smoke prerequisite passed in the {framework_probe_label}.",
                evidence=[f"probe_source={framework_probe_source}", *smoke_details],
            )
        )
    elif smoke_status == "failed":
        evidence = [f"probe_source={framework_probe_source}", *smoke_details]
        if smoke_error:
            evidence.append(f"smoke_error={smoke_error}")
        checks.append(
            make_check(
                "framework-smoke-prerequisite",
                "block",
                f"Framework smoke prerequisite failed in the {framework_probe_label}.",
                category_hint="framework",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["framework", "task-smoke"],
                evidence=evidence,
            )
        )

    required_runtime = runtime_layer.get("required_imports", [])
    runtime_probes = runtime_layer.get("import_probes", {})
    runtime_probe_source = runtime_layer.get("probe_source") or framework_probe_source
    runtime_probe_label = describe_probe_source(runtime_probe_source)
    runtime_probe_error = runtime_layer.get("probe_error")
    missing_runtime = [
        pkg for pkg in required_runtime
        if pkg not in required_framework and not runtime_probes.get(pkg, False)
    ]
    if not selected_python_ready and required_runtime:
        checks.append(
            make_check(
                "runtime-importability",
                "warn",
                "Runtime importability is deferred until a workspace-local Python environment is resolved. Do not install runtime packages into system Python.",
                category_hint="env",
                severity="medium",
                evidence=[f"required_runtime={','.join(required_runtime)}", "system_python_fallback=forbidden"],
            )
        )
    elif required_runtime and not missing_runtime:
        checks.append(
            make_check(
                "runtime-importability",
                "ok",
                f"Required runtime imports are available in the {runtime_probe_label}.",
                evidence=[f"probe_source={runtime_probe_source}", *required_runtime],
            )
        )
    elif missing_runtime:
        evidence = [f"probe_source={runtime_probe_source}", *missing_runtime]
        if runtime_probe_error:
            evidence.append(f"probe_error={runtime_probe_error}")
        package_names = resolve_runtime_package_names(runtime_layer, missing_runtime)
        implicit_profile = runtime_layer.get("implicit_dependency_profile") or []
        for item in implicit_profile:
            if not isinstance(item, dict):
                continue
            import_name = str(item.get("import_name") or "").strip()
            if import_name not in missing_runtime:
                continue
            package_name = str(item.get("package_name") or "").strip()
            required_for = str(item.get("required_for") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if package_name:
                evidence.append(f"package_name[{import_name}]={package_name}")
            if required_for:
                evidence.append(f"required_for[{import_name}]={required_for}")
            if reason:
                evidence.append(f"profile_reason[{import_name}]={reason}")
        checks.append(
            make_check(
                "runtime-importability",
                "block",
                f"Required runtime imports are unavailable in the {runtime_probe_label}: {', '.join(missing_runtime)}.",
                category_hint="env",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["runtime-dependencies", "framework"],
                evidence=evidence,
                package_names=package_names,
            )
        )

    remote_asset_states = remote_assets.get("assets") or {}
    if remote_asset_states:
        cache_layout = remote_assets.get("cache_layout") or {}
        endpoint = remote_assets.get("hf_endpoint")
        endpoint_source = remote_assets.get("hf_endpoint_source")
        endpoint_reachable = remote_assets.get("endpoint_reachable")
        endpoint_error = remote_assets.get("endpoint_error")
        endpoint_evidence = []
        if endpoint:
            endpoint_evidence.append(f"HF_ENDPOINT={endpoint}")
        if endpoint_source:
            endpoint_evidence.append(f"hf_endpoint_source={endpoint_source}")
        if endpoint_error:
            endpoint_evidence.append(f"endpoint_error={endpoint_error}")
        if endpoint_reachable:
            checks.append(
                make_check(
                    "remote-huggingface-endpoint",
                    "ok",
                    "Hugging Face mirror endpoint is reachable for remote asset resolution.",
                    evidence=endpoint_evidence or ["remote asset endpoint is reachable"],
                )
            )
        else:
            checks.append(
                make_check(
                    "remote-huggingface-endpoint",
                    "block",
                    "Hugging Face mirror endpoint is unavailable for remote asset resolution.",
                    category_hint="env",
                    severity="high",
                    remediable=False,
                    remediation_owner="manual-network",
                    revalidation_scope=["workspace-assets", "task-smoke"],
                    evidence=endpoint_evidence or ["remote asset endpoint is unreachable"],
                )
            )

        if "model_path" in remote_asset_states:
            hub_cache = cache_layout.get("hub_cache")
            hub_cache_writable = bool(cache_layout.get("hub_cache_writable"))
            evidence = []
            if hub_cache:
                evidence.append(f"hub_cache={hub_cache}")
            if cache_layout.get("source"):
                evidence.append(f"cache_source={cache_layout.get('source')}")
            if hub_cache_writable:
                checks.append(
                    make_check(
                        "remote-huggingface-model-cache",
                        "ok",
                        "Model cache location is writable for remote Hugging Face resolution.",
                        evidence=evidence or ["model cache is writable"],
                    )
                )
            else:
                checks.append(
                    make_check(
                        "remote-huggingface-model-cache",
                        "block",
                        "Model cache location is not writable for remote Hugging Face resolution.",
                        category_hint="env",
                        severity="high",
                        remediable=False,
                        remediation_owner="manual-workspace",
                        revalidation_scope=["workspace-assets", "task-smoke"],
                        evidence=evidence or ["model cache is not writable"],
                    )
                )

        if "dataset_path" in remote_asset_states:
            datasets_cache = cache_layout.get("datasets_cache")
            datasets_cache_writable = bool(cache_layout.get("datasets_cache_writable"))
            evidence = []
            if datasets_cache:
                evidence.append(f"datasets_cache={datasets_cache}")
            if cache_layout.get("source"):
                evidence.append(f"cache_source={cache_layout.get('source')}")
            if datasets_cache_writable:
                checks.append(
                    make_check(
                        "remote-huggingface-dataset-cache",
                        "ok",
                        "Dataset cache location is writable for remote Hugging Face resolution.",
                        evidence=evidence or ["dataset cache is writable"],
                    )
                )
            else:
                checks.append(
                    make_check(
                        "remote-huggingface-dataset-cache",
                        "block",
                        "Dataset cache location is not writable for remote Hugging Face resolution.",
                        category_hint="env",
                        severity="high",
                        remediable=False,
                        remediation_owner="manual-workspace",
                        revalidation_scope=["workspace-assets", "task-smoke"],
                        evidence=evidence or ["dataset cache is not writable"],
                    )
                )

    for key in ("entry_script", "model_path", "dataset_path", "checkpoint_path", "output_path"):
        asset = workspace.get(key, {})
        if not asset:
            continue
        required = asset.get("required", False)
        exists = asset.get("exists", False)
        satisfied = asset.get("satisfied", exists)
        resolution_mode = asset.get("resolution_mode")
        if required and not satisfied:
            if resolution_mode == "remote-huggingface":
                continue
            if key == "entry_script" and asset.get("source") == "bundled-example" and asset.get("template_path"):
                evidence = [f"{key} missing", f"template_path={asset.get('template_path')}"]
                if asset.get("example_recipe_id"):
                    evidence.append(f"example_recipe_id={asset.get('example_recipe_id')}")
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "block",
                        "Required training entry script is missing but can be scaffolded from the bundled training example.",
                        category_hint="asset",
                        severity="high",
                        remediable=True,
                        remediation_owner="readiness-agent",
                        revalidation_scope=["workspace-assets", "target", "runtime-dependencies"],
                        evidence=evidence,
                        **asset_check_fields(asset, key),
                    )
                )
            elif key == "entry_script":
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "block",
                        "Required entry script is missing.",
                        category_hint="workspace",
                        severity="high",
                        remediable=False,
                        remediation_owner="workspace",
                        revalidation_scope=["workspace-assets", "target"],
                        evidence=[f"{key} missing"],
                    )
                )
            elif key == "dataset_path" and asset.get("asset_provider") == "huggingface" and asset.get("repo_id"):
                evidence = [f"{key} missing", f"asset_provider={asset.get('asset_provider')}", f"repo_id={asset.get('repo_id')}"]
                if asset.get("dataset_split"):
                    evidence.append(f"dataset_split={asset.get('dataset_split')}")
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "block",
                        "Required dataset asset is missing locally but can be downloaded from Hugging Face.",
                        category_hint="asset",
                        severity="high",
                        remediable=True,
                        remediation_owner="readiness-agent",
                        revalidation_scope=["workspace-assets", "task-smoke"],
                        evidence=evidence,
                        **asset_check_fields(asset, key),
                    )
                )
            elif key == "dataset_path":
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "block",
                        "Required dataset path is missing for training.",
                        category_hint="workspace",
                        severity="high",
                        remediable=False,
                        remediation_owner="workspace",
                        revalidation_scope=["workspace-assets", "task-smoke"],
                        evidence=[f"{key} missing"],
                    )
                )
            else:
                evidence = [f"{key} missing"]
                if asset.get("asset_provider"):
                    evidence.append(f"asset_provider={asset.get('asset_provider')}")
                if asset.get("repo_id"):
                    evidence.append(f"repo_id={asset.get('repo_id')}")
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "block",
                        (
                            f"Required asset {key} is missing locally but can be downloaded from Hugging Face."
                            if asset.get("asset_provider") == "huggingface" and asset.get("repo_id")
                            else f"Required asset {key} is missing."
                        ),
                        category_hint="asset",
                        severity="high",
                        remediable=bool(asset.get("asset_provider") == "huggingface" and asset.get("repo_id")) or key == "model_path",
                        remediation_owner="readiness-agent",
                        revalidation_scope=["workspace-assets", "task-smoke"],
                        evidence=evidence,
                        **asset_check_fields(asset, key),
                    )
                )
        elif required:
            if exists:
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "ok",
                        f"Required asset {key} is present.",
                        evidence=[f"{key} exists"],
                    )
                )
            elif resolution_mode == "remote-huggingface":
                remote_asset = remote_asset_states.get(key, {})
                evidence = [f"resolution_mode={resolution_mode}"]
                if asset.get("repo_id"):
                    evidence.append(f"repo_id={asset.get('repo_id')}")
                if remote_assets.get("hf_endpoint"):
                    evidence.append(f"HF_ENDPOINT={remote_assets.get('hf_endpoint')}")
                if remote_asset.get("cache_path"):
                    evidence.append(f"cache_path={remote_asset.get('cache_path')}")
                if key == "dataset_path" and asset.get("dataset_split"):
                    evidence.append(f"dataset_split={asset.get('dataset_split')}")
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "ok",
                        f"Required asset {key} will resolve remotely from Hugging Face at runtime.",
                        evidence=evidence,
                        **asset_check_fields(asset, key),
                    )
                )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect deterministic readiness checks from target and closure",
        epilog=(
            "Internal helper. Use the top-level readiness workflow entrypoint instead of "
            "calling leaf helpers directly. If the selected workspace environment is unresolved, "
            "do not use system python or pip as a substitute."
        ),
    )
    parser.add_argument("--target-json", required=True, help="path to execution target JSON")
    parser.add_argument("--closure-json", required=True, help="path to dependency closure JSON")
    parser.add_argument("--task-smoke-json", help="optional path to task smoke checks JSON")
    parser.add_argument("--output-json", required=True, help="path to output checks JSON")
    args = parser.parse_args()

    target = json.loads(Path(args.target_json).read_text(encoding="utf-8"))
    closure = json.loads(Path(args.closure_json).read_text(encoding="utf-8"))
    checks = collect_checks(target, closure)
    if args.task_smoke_json:
        checks.extend(json.loads(Path(args.task_smoke_json).read_text(encoding="utf-8")))
    Path(args.output_json).write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({"checks": len(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
