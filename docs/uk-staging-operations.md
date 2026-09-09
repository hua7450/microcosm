# UK staging operations

This document records the operational ownership and lifecycle decisions for
UK Microcosm staging telemetry. It does not authorize production publication.

## Maintainer responsibilities

- PolicyEngine Hugging Face organization administrators review individual
  access requests and administer `policyengine/populace-uk-staging`. The
  administrator performing initial provisioning becomes the recorded primary
  owner; another organization administrator must be recorded as backup before
  remote telemetry is enabled.
- Administrators of `PolicyEngine/microcosm` administer the GitHub `staging`
  environment. The initial required-reviewer candidates are `@anth-volk` and
  `@MaxGhenis`; environment creation must confirm both accounts still have the
  required repository access before applying that configuration.
- Calibration Diagnostics deployment maintainers administer the server-side
  read credential and deploy consumer support before UK remote output is
  enabled.

The current GitHub repository-administrator list was inspected on 2026-09-08.
It included `@anth-volk`, `@MaxGhenis`, `@nikhilwoodruff`, `@nwoodruff-co`,
`@PavelMakarchuk`, `@policyengine-auto`, and `@vahid-ahmadi`. This dated list is
evidence, not a permanent authorization list; provisioning must query current
permissions again.

The GitHub `staging` environment did not exist when inspected on 2026-09-08.
It was then created with required reviewers `@anth-volk` and `@MaxGhenis`,
self-review prevention enabled, and no environment secrets. The general
integration-test workflow does not reference this protected environment, so
these reviewers do not delay its synthetic tests. The environment is audited
or reconciled idempotently with:

```bash
uv run python tools/configure_github_staging_environment.py
uv run python tools/configure_github_staging_environment.py --apply
```

The setup permits no environment secret except `HF_STAGING_READ_TOKEN`. The
general integration-test workflow instead reads that optional name as a
repository-level Actions secret, where it may read only the private repository
card. Fork pull requests do not receive it. The integration build itself
receives no external service writer credential.

## Contract fixture ownership

Microcosm is the canonical source for staging contract fixtures because it is
the producer and validator of the repository files. Canonical fixtures live
under `packages/microcosm-build/tests/fixtures/staging/` and have a checked-in
SHA-256 manifest. Calibration Diagnostics keeps a byte-identical copy and
checks the same manifest digest. A contract change is incomplete until both
repositories accept the new fixture manifest in their respective feature
branches.

Version 1 fixtures describe current US output and remain fixed during the UK
implementation. Version 2 fixtures include successful spine, calibration,
sanitized failure, delivery-failure, and incompatible-version cases.

Regenerate and verify the canonical version 2 bytes with:

```bash
uv run python tools/generate_staging_contract_fixtures.py
uv run python tools/generate_staging_contract_fixtures.py --check
```

## Repository provisioning

The repository card template is
`docs/templates/populace-uk-staging-README.md`. An authorized Hugging Face
organization administrator provisions or reconciles the repository with:

```bash
uv run python tools/provision_uk_staging_repository.py --apply
uv run python tools/provision_uk_staging_repository.py --verify-access
uv run python tools/provision_uk_staging_repository.py --verify-access --verify-write
```

The procedure creates the dataset repository with private visibility first,
then applies the Hugging Face setting `gated="manual"`, which means each access
request requires individual approval. If the setting or card update fails, the
recovery operation selects private visibility again and never selects public
visibility. The write probe is optional, uses only
`verification/operator-write-probe.json`, and deletes that probe after a
successful check.

The procedure was run successfully on 2026-09-08 while authenticated as
`anth-volk`, a member of the `policyengine` organization. The resulting
`policyengine/populace-uk-staging` dataset reported `private: true` and
`gated: "manual"`; in concrete terms, the repository is not publicly readable
and individual access requests require explicit approval. Anonymous repository
inspection was refused, authenticated repository-card download returned the
888-byte card, and the operator write probe succeeded and was removed. This
confirms that the organization supports the required combination of private
visibility and individual manual approval.

The bootstrap credential was then replaced on 2026-09-08 by the local token
named `microcosm-uk-staging-local-writer`. Hugging Face reported its role as
`fineGrained`; authenticated repository-card download and the temporary write
probe both succeeded, and the probe was removed. The token was configured by
the operator for this dataset only; its secret value was not printed or written
to the worktree. This establishes the scoped local-writer check, but the
complete access-control task still requires a separate authenticated user that
has not been approved to demonstrate refusal. Calibration Diagnostics and the
optional GitHub repository-card check each require their own fine-grained
read-only credential.

## Command modes and files

Both UK commands support these staging modes:

- Default: local version 2 files plus best-effort delivery to
  `policyengine/populace-uk-staging`.
- `--staging-local-only`: the same validated files without constructing a
  remote client.
- `--no-staging`: no telemetry files, with a version 2 opt-out object written
  into build evidence.

Common options are `--staging-dir`, `--staging-repo-id`,
`--staging-prefix`, `--staging-run-id`, `--staging-candidate-id`,
`--staging-upload-interval-seconds`, and `--staging-read-back`. An empty
repository identifier is invalid in remote mode. Authenticated read-back is
valid only in remote mode.

Each local or remote bundle contains `runs.json`, `latest_staging.json`, and:

```text
runs/<run_id>/run_manifest.json
runs/<run_id>/progress.json
runs/<run_id>/events.ndjson
runs/<run_id>/calibration_progress.json  # calibration runs only
```

Every JSON document and event declares its schema name and version 2. Unknown
schema names or versions are incompatible data. Only reviewed aggregate JSON
artifacts are permitted; population H5 files, NumPy archives, source survey
tables, row-level extracts, archives, credentials, and environment data are
rejected before remote storage is called.

## Bounded smoke verification

The UK spine command accepts a source-family count only with the explicit
non-release posture:

```bash
uv run python tools/build_uk_frs_spine.py \
  --frs-raw-dir <licensed-frs-directory> \
  --spi-tab <put2223uk.tab> \
  --hmrc-ods <hmrc.ods> \
  --spine-h5 <output-directory>/uk-smoke.h5 \
  --sample-source-households 100 \
  --sample-seed 578 \
  --smoke \
  --staging-run-id <unique-development-run-id> \
  --staging-read-back
```

This command constructs and enriches the real spine, validates it, writes the
H5 and sidecar, validates local telemetry, optionally verifies authenticated
remote reads, and stops. It does not invoke national calibration, release
certification, release assembly, or publication. The H5 attributes, sidecar,
and telemetry identify the result as non-release. Release assembly and
publication refuse that evidence even when remote delivery succeeded.

Continuous integration uses a much smaller deterministic synthetic input and
`--staging-local-only`. Every output stays below the runner's temporary
directory, checkout credentials are not persisted, repository permission is
`contents: read`, and fork-originated pull requests run without secrets. The
job does not reference the protected GitHub `staging` environment.

The deterministic fixture contains 135 source families and exercises all 28
current UK spine transformations. Five requested families is valid for the
sampler, which retains 13 after structural additions, but it cannot reach the
capital-gains transformation's fixed minimum of 270 candidate households. For
seed 42, the smallest complete fixture run requests 63 source families. The
sampler retains 69; the support-channel and capital-gains clone transformations
then provide enough candidate households for the final donor transformation.
The fixture also contains positive source evidence for every required retained
HMRC income field.

Run the exact CI command locally with:

```bash
uv run python tools/build_uk_frs_spine.py \
  --synthetic-fixture-dir packages/microcosm-graph/tests/fixtures/parity/uk_spine/sources \
  --spine-h5 <temporary-directory>/uk-smoke.h5 \
  --sample-source-households 63 \
  --sample-seed 42 \
  --smoke \
  --staging-local-only \
  --staging-dir <temporary-directory>/staging \
  --staging-run-id ci-uk-smoke-h0063-s42
```

This fixture option refuses licensed input options, remote staging, and any
non-smoke use. The workflow
`.github/workflows/integration-tests.yml` runs on manual dispatch and every
pull request to `main`, without a path filter. Its shell commands live in
`tools/run_integration_tests.sh`. The test prints total and per-transformation
elapsed time. A local Apple-silicon run completed the command in approximately
45 seconds on 2026-09-08; dependency setup and runner variance remain within
the initial 15-minute workflow timeout.

### Local 100-source-household interface check

On 2026-09-08, the unpushed implementation completed this local-only command
against the deterministic synthetic input:

```bash
.venv/bin/python tools/build_uk_frs_spine.py \
  --synthetic-fixture-dir packages/microcosm-graph/tests/fixtures/parity/uk_spine/sources \
  --spine-h5 <temporary-directory>/uk-smoke.h5 \
  --sample-source-households 100 \
  --sample-seed 42 \
  --smoke \
  --staging-local-only \
  --staging-dir <temporary-directory>/staging \
  --staging-run-id local-uk-smoke-h0100-s42 \
  --staging-candidate-id local-uk-smoke-h0100-s42
```

The version 2 timestamps recorded 104 seconds from run creation to completion.
The sampler selected 100 of 135 eligible source families with no additions,
and the completed construction sequence contained 670 household rows and 805
person rows after support and donor-row creation. The sample receipt digest was
`41606b401732bfad71a7087f64c5885070882900f95a774f01f7b70c958f5307`;
the synthetic fixture digest was
`3e015967d7804a0724f3ae7f263453e3bb66c69c99bc9ef9baf88ecda2568c75`.
The output H5 digest was
`e515f20e45cf987995e27854e998965cebeac083915f1fea9f11791a8ec1ee85`,
and its build-sidecar digest was
`b2879b04ec4ef8b97f6045de3eb19325eff38eaa430591b7e6d10ebf3d0503cd`.
Local contract validation accepted the index, latest-run pointer, manifest,
progress document, and all 67 ordered events. Every output was written beneath
the temporary directory, and the delivery record reports `local_only`, zero
upload attempts, and no configured repository.

This is evidence for the command-line and local telemetry interface only. It
does not replace the required run against licensed inputs with authenticated
delivery and read-back from `policyengine/populace-uk-staging`.

### Authenticated 100-source-household verification

On 2026-09-09, the unpushed implementation completed the required non-release
smoke run against licensed inputs and the private
`policyengine/populace-uk-staging` repository. The exact command was:

```bash
.venv/bin/python tools/build_uk_frs_spine.py \
  --frs-raw-dir /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/extracted/frs \
  --spine-h5 /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/run/uk-smoke-h0100-s578-20260909T125305Z.h5 \
  --spi-tab /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/extracted/spi/put2223uk.tab \
  --hmrc-ods /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/public/Collated_Tables_3_1_to_3_11_2324.ods \
  --cgt-ods /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/public/Table_3_2025_Size_of_gain_by_income.ods \
  --was-tab /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/extracted/was/was_round_8_hhold_eul_may_2025_230525.tab \
  --lcfs-hh-tab /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/extracted/lcfs/dvhh_ukanon_v2_2023.tab \
  --lcfs-person-tab /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/extracted/lcfs/dvper_ukanon_202324_2023.tab \
  --etb-tab /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/extracted/etb/householdv2_1977-2024.tab \
  --sample-source-households 100 \
  --sample-seed 578 \
  --smoke \
  --staging-dir /private/tmp/microcosm-uk-smoke-inputs.yNa0nY/run/staging \
  --staging-repo-id policyengine/populace-uk-staging \
  --staging-prefix runs \
  --staging-run-id uk-smoke-h0100-s578-20260909T125305Z \
  --staging-candidate-id uk-smoke-h0100-s578-20260909T125305Z \
  --staging-upload-interval-seconds 30 \
  --staging-read-back
```

The run started at `2026-09-09T12:53:50Z` and completed at
`2026-09-09T12:58:20Z`, for 270 seconds total. The existing monotonic stage
observer recorded 226.545 seconds inside the 28 transformations; input
verification, serialization, remote writes, authenticated reads, and other
command overhead account for the remaining 43.455 seconds. The transformation
durations were:

| Transformation | Seconds |
| --- | ---: |
| `frs_spine` | 7.952 |
| `age_tail` | 0.005 |
| `frs_employment` | 4.042 |
| `frs_council_tax` | 0.929 |
| `frs_disability` | 0.751 |
| `frs_education` | 3.689 |
| `frs_legacy_proxies` | 5.261 |
| `frs_education_grant_split` | 0.937 |
| `frs_take_up` | 0.008 |
| `frs_person_draws` | 0.122 |
| `frs_household_draws` | 0.003 |
| `frs_brma` | 0.507 |
| `was_wealth` | 33.292 |
| `regional_property_uprating` | 0.014 |
| `lcfs_consumption` | 46.191 |
| `etb_vat` | 13.422 |
| `etb_services` | 24.128 |
| `frs_hmrc_spine_leaves` | 0.387 |
| `spi_support_channel` | 0.124 |
| `hmrc_spi_income_spine` | 82.990 |
| `uc_reporter_redraw` | 1.243 |
| `uc_capital_coherence` | 0.007 |
| `uc_deduction_attributes` | 0.006 |
| `cgt_incidence_clone` | 0.018 |
| `cgt_band_donors` | 0.036 |
| `hmrc_cgt_gains_spine` | 0.212 |
| `salary_sacrifice` | 0.261 |
| `student_loans` | 0.009 |

The sampler found 16,288 eligible source families, requested 100, added 24
families required for structural coverage, and retained 124 source families
and 124 initial household rows. Its receipt digest was
`097e41354d4a50ce565cb27d8bfd290fec5634fc9b09537366098bf8d5ef5fe2`.
Support and donor-row construction produced 670 household rows, 796 benefit-unit
rows, and 1,421 person rows. The final frame content identity was
`158b8f3783feeb3b3fcd893d7217dc15b0dedab6adc200ced1c482dbe7947353`,
and the stochastic contract digest was
`c0c506e18e8c0537b715a6710a72452ee79116e9f7ec9f915b9e2d101883573b`.
The pinned version 1 and version 2 telemetry fixture-manifest digests remained
`165d24caf29b82afdb0ce241b65d088da552abd59d6ddabf4b2183b9a61be75b`
and `372a1c82e4bafbe636299636f25d51a2edad7d7dc28814cadbddac6f436948f0`.

The local H5 was 4,760,660 bytes with digest
`dc5d81bdd0ce2fcd7afa23b65c68948c10f87a89456349af07de91962a68706e`.
The build sidecar digest was
`095ce3fab51562fe7e797f4ff8e18efd38e359bba162539d0b5cc41202f8211d`.
Both remained under the operator's temporary directory and were not uploaded.
The remote repository contained only these five run-contract files:

```text
latest_staging.json
runs.json
runs/uk-smoke-h0100-s578-20260909T125305Z/events.ndjson
runs/uk-smoke-h0100-s578-20260909T125305Z/progress.json
runs/uk-smoke-h0100-s578-20260909T125305Z/run_manifest.json
```

Authenticated read-back passed for the manifest, progress document,
latest-run pointer, and run index. All declared schema versions were 2, all
identifiers matched, the event sequence contained 67 contiguous records, and
the last event reported `complete` with status `completed`. The exact remote
payload also passed the version-aware parser from
`PolicyEngine/calibration-diagnostics#181`. No unapproved path existed beneath
the run directory.

The local and remote latest-run pointer, run index, and event stream were
byte-identical. The remote manifest and progress document reported 40 upload
attempts and successes, while the final local copies reported 45. This is the
expected result of the last five-file synchronization: each remote document
captures the count before its own upload, and the local bundle is persisted
again after all five uploads. All other fields were identical.

The staging manifest records `run_kind=smoke`, `non_release=true`, and a null
`release_id`. The local validation sidecar retained an internal `release_id`
named `uk-frs-spine-h0100-s578-20260909T125350Z` because that existing report
schema assigns an identifier to each build; the same report records
`release_candidate=false` and `shippable=false`. That identifier was not placed
in staging records or written to a production repository. The command did not
run calibration, certification, release assembly, or publication, and it did
not modify any tracked source file.

## Monitoring authentication

Calibration Diagnostics already loads staging files inside Next.js API routes,
and its Hugging Face authorization header is constructed only in server-side
library code. UK support will retain that request boundary and use separate
server deployment variables:

- `POPULACE_UK_STAGING_HF_REPO`
- `POPULACE_UK_STAGING_HF_REVISION`
- `POPULACE_UK_STAGING_HF_TOKEN`

The browser calls the Calibration Diagnostics API routes and never calls the
private Hugging Face repository with a credential. Responses and error text
must not contain the token. The UK token is a fine-grained read credential for
`policyengine/populace-uk-staging`; it is not shared with the local writer.

The version-aware consumer implementation was published for review on
2026-09-08 as
[`PolicyEngine/calibration-diagnostics#181`](https://github.com/PolicyEngine/calibration-diagnostics/pull/181).
Its feature-branch checks passed locally with TypeScript compilation and all
328 tests before publication; its GitHub tests, TypeScript/build check, and
Vercel preview deployment also passed. Review was requested from `@MaxGhenis`.
UK remote output remains disabled until that change is reviewed, merged,
deployed with the separate read credential, and verified against both contract
fixture versions.

## Retention

Staging run directories are retained for 90 days, with at least the ten most
recent runs retained even when they are older. PolicyEngine Hugging Face
organization administrators own the monthly cleanup. Cleanup first writes a
new `runs.json` that omits expired identifiers, verifies that the retained
entries remain readable, and only then removes the expired run directories.
`latest_staging.json` must always reference a retained run. Cleanup never
touches production repositories or local operator evidence.

## Access-control verification

Hugging Face currently documents both `private=True` and `gated="manual"` on
the repository-settings API, and documents individual approval for gated
datasets. This establishes API support, but not the effective behavior of the
PolicyEngine organization. Before remote output is enabled, the provisioning
procedure must verify the combined settings with an organization writer and
then prove anonymous or unapproved refusal, approved repository-card download,
and narrowly scoped writer access. If any check fails, the repository remains
private and UK remote output remains disabled.

References:

- <https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.update_repo_settings>
- <https://huggingface.co/docs/hub/datasets-gated>
- <https://huggingface.co/docs/hub/repositories-settings>

## Monitoring, rollback, and publication

Monitor `runs.json` for discovery, `progress.json` for current status, and
`events.ndjson` for ordered stage durations. A remote verification run also
downloads and validates the run manifest, progress document, latest-run
pointer, and run index. Upload failures are recorded with counts and reviewed
error codes while local recording continues; unrestricted remote exception
text is not serialized.

Rollback is configuration-first: use `--staging-local-only` to retain local
evidence or `--no-staging` for a deliberate opt-out, disable the GitHub
workflow, and revoke its optional read credential. Revoke the local writer
credential separately. Leave existing private run evidence intact for audit
and do not alter production repository references.

National calibration build records carry the validated version 2 delivery
object. Release assembly copies that object unchanged into
`build_manifest.json`. Normal publication refuses local-only evidence, invalid
or unknown delivery versions, and remote mode with zero successful uploads. It
retains the existing explicit missing-staging override and accepts a valid
deliberate opt-out. Production publication remains a separate operator action.
