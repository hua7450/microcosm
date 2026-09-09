from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.staging_v2 import validate_v2_bundle
from microcosm.build.uk_runtime.ledger_targets import UKLedgerTargetCompilation
from microcosm.calibrate import TargetRegistry, TargetSpec


def _load_driver_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "calibrate_uk_national_dataset.py"
    spec = importlib.util.spec_from_file_location("calibrate_uk_national_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _registry():
    return TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=1.0,
                source="test",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )


def _args(tmp_path: Path) -> list[str]:
    paths = {
        "input": tmp_path / "input.h5",
        "ledger": tmp_path / "ledger",
        "staging": tmp_path / "staging.h5",
        "diagnostics": tmp_path / "diagnostics.json",
        "record": tmp_path / "record.json",
    }
    paths["ledger"].mkdir(exist_ok=True)
    for key, path in paths.items():
        if key != "ledger":
            path.write_bytes(key.encode())
    return [
        "--input-h5",
        str(paths["input"]),
        "--input-sha256",
        "a" * 64,
        "--ledger-facts",
        str(paths["ledger"]),
        "--ledger-facts-sha256",
        "b" * 64,
        "--ledger-manifest-sha256",
        "c" * 64,
        "--staging-h5",
        str(paths["staging"]),
        "--diagnostics-json",
        str(paths["diagnostics"]),
        "--build-record-json",
        str(paths["record"]),
        "--release-id",
        "dev-calibration",
        "--no-staging",
    ]


def test_driver_refuses_release_candidate_outright(tmp_path: Path):
    # The seam's scoped battery covers 6 of the declared entries and must
    # never sign a shippability claim (the #757 release-cut audit); the
    # release verdict belongs to the release-cut certification producer.
    driver = _load_driver_module()
    with pytest.raises(SystemExit):
        driver._parse_args(_args(tmp_path) + ["--release-candidate"])
    # The refusal is unconditional — an otherwise doctrine-clean invocation
    # is refused too, not just ones with override flags.
    with pytest.raises(SystemExit):
        driver._parse_args(_args(tmp_path) + ["--release-candidate", "--epochs", "128"])


def test_driver_refuses_canonical_release_ids(tmp_path: Path):
    # Canonical release ids name shippable candidates; the seam runs under
    # staging or dev ids only, and redirects canonical ids to the
    # release-cut producer.
    driver = _load_driver_module()
    base = _args(tmp_path)
    release_index = base.index("--release-id")
    for canonical in (
        "populace-uk-2024-frs-k100",
        "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z",
    ):
        args = [*base]
        args[release_index + 1] = canonical
        with pytest.raises(SystemExit):
            driver._parse_args(args)


def test_driver_accepts_operator_exclusions_on_staging_posture(tmp_path: Path):
    driver = _load_driver_module()
    exclusions = tmp_path / "operator.json"
    exclusions.write_text("{}", encoding="utf-8")
    parsed = driver._parse_args(
        _args(tmp_path) + ["--measure-exclusions", str(exclusions)]
    )
    assert parsed.measure_exclusions == exclusions


def test_driver_refuses_bad_sha_and_path_alias(tmp_path: Path):
    driver = _load_driver_module()
    base_args = _args(tmp_path)
    with pytest.raises(SystemExit):
        driver._parse_args([*base_args[:3], "not-a-sha", *base_args[4:]])

    args = _args(tmp_path)
    staging_index = args.index("--staging-h5") + 1
    args[staging_index] = args[args.index("--input-h5") + 1]
    with pytest.raises(SystemExit, match="distinct paths"):
        driver._parse_args(args)


def test_driver_refuses_feed_outside_the_committed_pin_without_override():
    driver = _load_driver_module()

    with pytest.raises(SystemExit, match="committed UK feed pin"):
        driver._check_committed_ledger_feed_pin(
            "b" * 64,
            allow_unpinned_feed=False,
        )

    driver._check_committed_ledger_feed_pin(
        "b" * 64,
        allow_unpinned_feed=True,
    )


def test_driver_threads_registry_exclusions_resolver_and_overrides(
    monkeypatch, tmp_path, capsys
):
    driver = _load_driver_module()
    calls = []
    registry = _registry()
    pruned_registry = TargetRegistry([], country="uk")
    artifact = SimpleNamespace(
        path=tmp_path / "ledger",
        facts=({"fact": 1},),
        facts_sha256=driver._LEDGER_FACT_FEED_PIN["facts_sha256"],
        manifest_sha256="c" * 64,
    )
    artifact.path.mkdir()
    (artifact.path / "consumer_facts.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        driver, "load_ledger_consumer_artifact", lambda *a, **k: artifact
    )
    monkeypatch.setattr(
        driver,
        "compile_uk_target_registry",
        lambda facts, target_period: UKLedgerTargetCompilation(registry, ()),
    )
    monkeypatch.setattr(
        driver,
        "load_uk_frs_release",
        lambda: SimpleNamespace(calibration_year=2025),
    )
    monkeypatch.setattr(
        driver, "load_uk_calibration_measure_exclusions", lambda path: ()
    )
    monkeypatch.setattr(
        driver,
        "apply_uk_calibration_measure_exclusions",
        lambda reg, exclusions: (pruned_registry, {"excluded": "reviewed"}),
    )

    class FakeResolver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(driver, "UKMeasureResolver", FakeResolver)

    def fake_run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            staging_sha256="1" * 64,
            diagnostics_sha256="2" * 64,
            terminal_gate_sha256="3" * 64,
            build_record_sha256="4" * 64,
            build_record={"gate_summary": {"uk_target_fit": "passed"}},
        )

    monkeypatch.setattr(driver, "run_uk_calibration", fake_run)

    argv = _args(tmp_path)
    # The driver verifies the input pin before the resolver reads the H5, so
    # the threading test must pin the fixture file's real digest.
    argv[argv.index("--input-sha256") + 1] = driver._sha256_file(
        Path(argv[argv.index("--input-h5") + 1])
    )
    result = driver.main(argv + ["--epochs", "128"])

    assert result == 0
    call = calls[0]
    assert call["register_registry"] is pruned_registry
    assert call["band_edge_registry"] is registry
    assert call["calibration_year"] == 2025
    assert call["exclusion_receipt"] == {"excluded": "reviewed"}
    assert call["doctrine"].epochs == 128
    assert call["doctrine_overrides"] == {"epochs": {"default": 256, "effective": 128}}
    assert isinstance(call["measure_resolver"], FakeResolver)
    assert (
        call["measure_resolver"].kwargs["simulation_source"] == call["paths"].input_h5
    )
    assert call["source_pins"]["ledger_facts"] == {
        "sha256": driver._LEDGER_FACT_FEED_PIN["facts_sha256"],
        "size_bytes": 2,
    }
    assert call["run_config_extra"] == {
        "calibration_year": 2025,
        "allow_unpinned_feed": False,
    }
    assert call["progress_callback"] is None
    assert call["event_callback"] is None
    assert call["staging_delivery"] == {
        "contract_version": 2,
        "enabled": False,
        "mode": "disabled",
        "run_id": None,
        "configured_repository": None,
        "upload_attempts": 0,
        "upload_successes": 0,
        "read_back": "not_requested",
        "last_error_code": None,
        "opt_out_reason": "--no-staging",
    }
    assert "uk_target_fit" in capsys.readouterr().out


def test_driver_exposes_shared_staging_modes(tmp_path: Path) -> None:
    driver = _load_driver_module()
    base = _args(tmp_path)
    without_disabled = base[: base.index("--no-staging")]

    remote = driver._parse_args(without_disabled)
    local = driver._parse_args([*without_disabled, "--staging-local-only"])

    assert remote.staging_repo_id == "policyengine/populace-uk-staging"
    assert not remote.staging_local_only and not remote.no_staging
    assert local.staging_local_only and not local.no_staging
    with pytest.raises(SystemExit):
        driver._parse_args([*without_disabled, "--staging-repo-id", ""])
    with pytest.raises(SystemExit):
        driver._parse_args(
            [*without_disabled, "--staging-local-only", "--staging-read-back"]
        )


def test_driver_records_local_calibration_stage_coverage(
    monkeypatch, tmp_path: Path
) -> None:
    driver = _load_driver_module()
    registry = _registry()
    artifact = SimpleNamespace(
        path=tmp_path / "ledger",
        facts=({"fact": 1},),
        facts_sha256=driver._LEDGER_FACT_FEED_PIN["facts_sha256"],
        manifest_sha256="c" * 64,
    )
    artifact.path.mkdir()
    (artifact.path / "consumer_facts.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        driver, "load_ledger_consumer_artifact", lambda *args, **kwargs: artifact
    )
    monkeypatch.setattr(
        driver,
        "compile_uk_target_registry",
        lambda facts, target_period: UKLedgerTargetCompilation(registry, ()),
    )
    monkeypatch.setattr(
        driver, "load_uk_frs_release", lambda: SimpleNamespace(calibration_year=2025)
    )
    monkeypatch.setattr(
        driver, "load_uk_calibration_measure_exclusions", lambda path: ()
    )
    monkeypatch.setattr(
        driver,
        "apply_uk_calibration_measure_exclusions",
        lambda active, exclusions: (active, {}),
    )
    monkeypatch.setattr(driver, "UKMeasureResolver", lambda **kwargs: object())

    def fake_run(**kwargs):
        callback = kwargs["event_callback"]
        for stage_id in (
            "input_loading",
            "measure_resolution",
            "calibration",
            "diagnostics",
            "release_check_evaluation",
            "candidate_h5_creation",
            "build_record_creation",
        ):
            callback(stage_id, "started", {})
            callback(stage_id, "completed", {"aggregate_count": 1})
        kwargs["progress_callback"](
            {
                "kind": "calibration_epoch",
                "epoch": 1,
                "epochs": 2,
                "phase": "solve",
                "loss": 1.5,
                "iteration": 1,
            }
        )
        return SimpleNamespace(
            staging_sha256="1" * 64,
            diagnostics_sha256="2" * 64,
            terminal_gate_sha256="3" * 64,
            build_record_sha256="4" * 64,
            build_record={
                "gate_summary": {"uk_target_fit": "passed"},
                "staging_delivery": kwargs["staging_delivery"],
            },
        )

    monkeypatch.setattr(driver, "run_uk_calibration", fake_run)
    argv = _args(tmp_path)
    argv = argv[: argv.index("--no-staging")]
    input_path = Path(argv[argv.index("--input-h5") + 1])
    argv[argv.index("--input-sha256") + 1] = driver._sha256_file(input_path)
    staging_root = tmp_path / "telemetry"
    argv.extend(
        [
            "--staging-local-only",
            "--staging-dir",
            str(staging_root),
            "--staging-run-id",
            "calibration-stage-test",
        ]
    )

    assert driver.main(argv) == 0

    bundle = validate_v2_bundle(staging_root, "calibration-stage-test")
    assert bundle["calibration_progress"]["events"][0]["epoch"] == 1
    completed = {
        event["stage_id"]
        for event in bundle["events"]
        if event["status"] == "completed"
    }
    assert {
        "target_compilation",
        "input_loading",
        "measure_resolution",
        "calibration",
        "diagnostics",
        "release_check_evaluation",
        "candidate_h5_creation",
        "build_record_creation",
        "complete",
    } <= completed
    record_path = Path(argv[argv.index("--build-record-json") + 1])
    record = json.loads(record_path.read_text())
    assert record["staging_delivery"]["mode"] == "local_only"


def test_driver_refuses_the_national_release_id(tmp_path: Path):
    # The constant national id names a shippable release (ruling 2026-08-27);
    # the seam runs under staging or dev ids only.
    driver = _load_driver_module()
    base = _args(tmp_path)
    args = [*base]
    args[base.index("--release-id") + 1] = "microcosm-uk-2024-25-national"
    with pytest.raises(SystemExit):
        driver._parse_args(args)
