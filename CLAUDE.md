# Microcosm — agent guide

Agent-operational notes only. The [README](README.md) covers usage and release
operations; [DESIGN.md](DESIGN.md) is the architectural authority. On any
conflict, executable configuration (pyproject.toml, `.github/workflows/`,
`tools/`) and DESIGN.md win over this file.

## Layout

uv workspace monorepo. Shards live in `packages/microcosm-<x>/` and import as
the PEP 420 namespace `microcosm.<x>`: `frame`, `fit`, `calibrate`, `build`,
`data`. There is no top-level `microcosm` package directory — never add an
`__init__.py` to the namespace root.

## Commands

```bash
uv sync --all-packages   # set up the whole workspace
uv sync --all-packages --locked --extra us --extra uk  # CI engine env
uv run pytest            # behavioral contract suite + unit tests (all shards)
uv run ruff check .      # lint
```

PR CI (`.github/workflows/test.yml`) has four lanes — `lint`, `fast`,
`engine` (three jobs), and `wheels` — fed by a `changes` job that classifies
the diff into `shared`/`us`/`uk`. `lint` verifies
`tools/ci_test_groups.py --verify`, syncs with `--locked`, and runs ruff.
`fast` runs the full tracked test-file inventory without engine extras in
three groups (`trade`, `spine-uk`, `rest`); engine-gated tests skip there
through whichever guard they carry — the `requires_*` markers, or the
`importorskip` calls that remain the norm on the US side. `engine-shared` always syncs
`--extra us --extra uk` and runs the shared/spec group. `engine-us` and
`engine-uk` use statically named matrix jobs and job-level `if` conditions
based only on the `changes` job outputs: country jobs run on main pushes or
when that country or shared paths changed. A country PR that merges over a
fresh change to the other country is certified by main's push run; watch main
after merging. The `wheels` lane remains the packaging gate: build every
shard's real wheel, install into a clean uv-export-constrained venv, assert
the wheel/import boundary and spec digests, and run the suite against installed
wheels.

`requires_us` and `requires_uk` are registered pytest markers. Mark new tests
that need a live PolicyEngine engine with the appropriate marker; the root
collection hook skips them when that engine is absent, and the marker also
makes `-m requires_uk` a real selector. Do not add new module-local skip
aliases. Existing `importorskip` guards (still the norm across the US files)
keep working and were deliberately left in place — convert one only when you
are already editing that test for another reason.

**Adding a test file.** It must sit directly in `packages/<shard>/tests/` — flat,
no subdirectories; `fixtures/` and `golden/` hold data only — and be named
`test_*.py`. The lanes run explicit file lists built from a flat pathspec, while
local `uv run pytest` and the wheels lane discover recursively, so a test parked
next to its fixtures would run locally and stay green in CI without ever
executing against an engine. `--verify` fails on such a file rather than letting
it hide. Build tests that exercise a country engine must be named `test_us_*` or
`test_uk_*` so they land in that country's lane; an engine-dependent file named
anything else falls into the always-on `shared-spec` group and runs on every PR.
After adding one, check `tools/ci_test_groups.py --verify`: your file should
appear in the group you expect and never under `[defaulted]`. `tools/ci_test_groups.py` is the partition
authority for CI file groups; update it and keep `--verify` green whenever
test files move or new grouped lanes are added. Spec identities
(`spec_sha256` pins, seed digests) attest kernel source and locked
RNG-library versions, so they legitimately move when main changes an attested
module or dependency. CI tests the merge ref, so merge main and re-pin rather
than hunting for an environment leak. Editable installs hide packaging breaks;
if you touch packaging, build wheels locally before pushing.

The separate `.github/workflows/integration-tests.yml` workflow runs integration
coverage on every pull request to `main` and on manual dispatch. Its current UK
job runs the real spine command against the committed synthetic fixture. It
requests 63 source families with seed 42, uses `--smoke --staging-local-only`,
writes only under the runner's temporary directory, and is not part of
`ci-ok`. The job has `contents: read`, does not persist checkout credentials,
does not reference a protected GitHub environment, and receives no external
writer credential. Fork pull requests run the synthetic test without secrets.
An optional repository-level `HF_STAGING_READ_TOKEN` permits a separate
repository-card read. The workflow invokes `tools/run_integration_tests.sh` so
its shell logic remains locally executable. Run it locally without the optional
repository read with:

```bash
HF_STAGING_READ_TOKEN= bash tools/run_integration_tests.sh
```

## The PR-CI / certification boundary

PR CI is secrets-free and never touches restricted microdata. Green PR checks
mean the code contracts hold — they do **not** certify data artifacts.
Builds, calibrations, and releases run outside PR CI, need gated Hugging Face
data and credentials, and cannot run from forks. Release publication is a
deliberate human step (`tools/publish_release.sh` →
`microcosm-publish-release`), gated by `tools/preflight_us_release_gates.py`;
see README "Releasing & alerts". Publication also refuses a release whose
build recorded staging telemetry that never reached its repo
(`--allow-missing-staging` overrides); a build that declared `--no-staging`
publishes without the flag. Never publish or promote artifacts as a side
effect of another task.

A US release or release-gate preflight that receives a multispine pool through
`--base-h5` must authenticate its sibling terminal manifest. A current stacked
pool whose terminal battery is red remains fail-closed unless the operator
passes `--allow-gate-failed-base-pool`; that opt-in carries the full red verdict
into `release_manifest.json` for a separate human publication decision. It does
not weaken the exact-k manifest arm or authorize publication by itself.
A sealed deny-list in `microcosm.build.us_runtime.h5_io` overrides this opt-in
for known-excluded publications while preserving their scoring-only diagnostic
path.

## Root journals are history, not state

The root `PROGRESS*.md`, `FINAL_REPORT.md`, `*_COVERAGE_PROGRESS.md`, and
similar files are session-handoff journals: accurate when written, historical
afterward. Do not treat their "State"/"Next" sections as current truth —
check git/GitHub instead. When a branch carrying such a journal merges,
historicize any currency claims in it ("nothing was pushed", "do not merge",
"in progress") in place with a dated note, so the file cannot mislead later
readers. Adjudicated verdicts belong in `experiments/` or the tracking issue,
with the journal pointing to them.

## Review this file

Update this guide in the same PR whenever the workspace layout, test
commands, or release flow change. If you find it contradicting the repo,
trust the repo and fix this file.
