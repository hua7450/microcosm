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
repository-level Actions secret, where it may inspect the private repository's
settings. Fork pull requests do not receive it. The integration build itself
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

An authorized Hugging Face organization administrator provisions or reconciles
the repository with:

```bash
uv run python tools/provision_uk_staging_repository.py --apply
uv run python tools/provision_uk_staging_repository.py --verify-access
uv run python tools/provision_uk_staging_repository.py --verify-access --verify-write
```

The procedure creates the dataset repository with private visibility first,
then applies the Hugging Face setting `gated="manual"`, which means each access
request requires individual approval. If the settings update fails, the
recovery operation selects private visibility again and never selects public
visibility. The write probe is optional, uses only
`verification/operator-write-probe.json`, and deletes that probe after a
successful check.

The procedure was run successfully on 2026-09-08 while authenticated as
`anth-volk`, a member of the `policyengine` organization. The resulting
`policyengine/populace-uk-staging` dataset reported `private: true` and
`gated: "manual"`; in concrete terms, the repository is not publicly readable
and individual access requests require explicit approval. Anonymous repository
inspection was refused, authenticated repository inspection succeeded, and the
operator write probe succeeded and was removed. This confirms that the
organization supports the required combination of private visibility and
individual manual approval.

The bootstrap credential was then replaced on 2026-09-08 by the local token
named `microcosm-uk-staging-local-writer`. Hugging Face reported its role as
`fineGrained`; authenticated repository inspection and the temporary write
probe both succeeded, and the probe was removed. The token was configured by
the operator for this dataset only; its secret value was not printed or written
to the worktree. This establishes the scoped local-writer check, but the
complete access-control task still requires a separate authenticated user that
has not been approved to demonstrate refusal. Calibration Diagnostics and the
optional GitHub repository access check each require their own fine-grained
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

## Smoke and exact-count verification

The UK spine command keeps fractional input sampling for scale tests. The
`--smoke` option marks its H5, sidecar, and staging records as non-release. It
does not invoke national calibration, release certification, release assembly,
or publication.

Continuous integration runs all current UK spine transformations against the
complete deterministic synthetic fixture. It does not reduce the fixture by a
source-family count. Every output stays below the runner's temporary directory,
and the workflow receives no external writer credential.

Run the integration command locally with:

```bash
uv run python tools/build_uk_frs_spine.py \
  --synthetic-fixture-dir packages/microcosm-graph/tests/fixtures/parity/uk_spine/sources \
  --spine-h5 <temporary-directory>/uk-smoke.h5 \
  --sample-fraction 1.0 \
  --sample-seed 42 \
  --smoke \
  --staging-local-only \
  --staging-dir <temporary-directory>/staging \
  --staging-run-id ci-uk-smoke-full-s42
```

The workflow `.github/workflows/integration-tests.yml` runs on manual dispatch
and every pull request to `main`, without a path filter. Its commands live in
`tools/run_integration_tests.sh`. The test reports total elapsed time and the
elapsed time for each transformation.

Exact household cardinality belongs to national calibration, after the complete
spine pool and target matrix exist. Supply all three selection options together:

```bash
uv run python tools/calibrate_uk_national_dataset.py \
  <required calibration inputs and outputs> \
  --exact-k <household-count> \
  --exact-k-pi-hi 0.95 \
  --exact-k-seed 17
```

For `K` below the input household count, calibration learns inclusion
probabilities on the complete pool, draws a seeded fixed-size Sampford support,
normalizes the selected input weights by their inclusion probabilities, and
refits ordinary calibration on exactly `K` households. For `K` equal to the
input count, it keeps the full support and still refits the weights. The build
record includes the requested count, realized count, seed, selection receipt,
and refit-baseline diagnostics.

An authenticated staging transport check completed on 2026-09-09 using the
earlier source-family-count interface. It uploaded only the five version 2 JSON
run records for `uk-smoke-h0100-s578-20260909T125305Z` to
`policyengine/populace-uk-staging`; it did not upload the H5 dataset. The
manifest, progress record, latest-run pointer, run index, and event stream all
passed authenticated read-back and version-aware parsing. That obsolete option
has since been removed because it selected source families before construction
and therefore did not guarantee a requested final household count.

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

## Access-control verification

Hugging Face currently documents both `private=True` and `gated="manual"` on
the repository-settings API, and documents individual approval for gated
datasets. This establishes API support, but not the effective behavior of the
PolicyEngine organization. Before remote output is enabled, the provisioning
procedure must verify the combined settings with an organization writer and
then prove anonymous or unapproved refusal, approved authenticated access, and
narrowly scoped writer access. If any check fails, the repository remains
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
