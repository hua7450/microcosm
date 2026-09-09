# SPI comparison on the UK publication baseline

Historical measurement baseline. The [subsequent matched comparison](879-corrected-calibration-comparison.md) corrects the national VOA geography and UC family measurements and supersedes the headline fit interpretation below. This report and its original receipt preserve the earlier runs.

This experiment ports the reviewed SPI changes in [#879](https://github.com/PolicyEngine/microcosm/pull/879)
onto the UK publication stack identified in [Max's 7 September handoff](https://github.com/PolicyEngine/microcosm/issues/665#issuecomment-5562794129).
The treatment preserves observed FRS inputs below age 16, rebases selected SPI
income flows to the build period, and conditions the FRS-only imputation on a
pension-receipt proxy. Source review also identified an omitted OTHERINV index
and corrected the interpretation of the age boundary.

**Matched result: national loss falls 35.8%, from 0.02346446 to 0.01506890,
and targets within 10% rise from 334/371 to 338/371. The candidate remains a
draft with release-blocking failures.** The ONS interest row accounts for
98.8% of the net objective gain. Aggregate income tax and state pension worsen;
three new target-fit failures appear. These results support the source
corrections but do not resolve the joint tax/pension/UC problem or establish an
overall release improvement. This is a combined SPI treatment, including the
reviewed OTHERINV correction; its effects are not attributed to the pension
bridge alone.

The [paired aggregate receipt](receipts/uk-publication-spi-comparison.json)
contains every one of the 371 target estimates, fixed bindings and loss weights,
contributions by target and family, source-quality summaries and gate verdicts.
It supersedes the historical experiment for judging this publication-stack port.
The [receipt generator](build_879_publication_receipt.py) rechecks those
identities against retained local runs and licensed input hashes. Its explicit
contract/manifest arguments and local paths are documented in the script;
regeneration reproduces the paired receipt byte-for-byte.

## Publication and runtime provenance

| Item | Pin or status |
|---|---|
| Control publication commit | `d43b4203c6ebe10e062cb3ef3034e66731ea055d` on `uk-publication-stack-834` |
| Earlier [#874](https://github.com/PolicyEngine/microcosm/pull/874) head | `8de6bf44d7bd4a5b6ce5c8c24add4501c0944824` |
| Original SPI delta | `bc11803f915fe72afd724515a27a3863035f4efc` → `194111af32b265b226dcbefe33cc735f362461dc` |
| Treatment source commit | `fc49b48200e1e15fe35bb4e15b948992bf03c28d`, directly atop the publication control; later report-only commits do not identify the executed source |
| PolicyEngine-UK / Core | `2.94.0` / `3.31.0` |
| NumPy / pandas / scikit-learn | `2.4.6` / `3.0.3` / `1.8.0` |
| Both environments' `uv.lock` SHA-256 | `ca9305fcebb5854d8d8369eb10fd134c918da7a79ab90f21c932c946b35165e5` |
| SPI source period / FRS build period / calibration year | `2022` / `2024` / `2025` |
| Control H5 SHA-256 | `2e68a13fc066c4b0b8b240361922596f9c268275e91223a49f66e3c8442ab157` |
| Treatment H5 SHA-256 | `a674409ea6c27919f02c267c06f5af24ee0628cbf2213ad8b7e8e37032eb0180` |

The port applies the declared-base SPI delta selectively. The publication stack
is not a descendant of the old SPI base, so inheriting the entire old branch
would change more than the SPI treatment. The earlier local candidate
`0feec085888790ea6387a0085878dc681f2e8445` described in the handoff was not
available in this checkout; it is not this treatment's execution receipt.

The selected publication tree includes #874's Chronicle loader/target repairs,
childcare inputs and England bus targets, plus the later publication deferral
register and gate-clock wiring. [#874's committed receipts](https://github.com/PolicyEngine/microcosm/blob/d43b4203c6ebe10e062cb3ef3034e66731ea055d/experiments/834-childcare-tfc-receipts.md)
identify the earlier composition evidence. Those results do not certify the
later publication tip. The [historical SPI report](840-spi-income-coherence.md)
retains the separate `bc11803f` / UK 2.92.1 experiment and its unchanged paired JSON.

## Reviewed source behavior

### Recipient age

[SPI Annex A, physical page 12](https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/9422_put_2223_full_documentation.pdf#page=12)
labels its first published age band “Under 25.” The pipeline's existing
`_SPI_AGE_RANGES` constructs ages from 16 upward. The treatment therefore uses
16 as an explicit recipient boundary to avoid applying that constructed support
to younger people. It does not interpret 16 as verified SPI survey eligibility.
A 15-year-old retains the observed FRS inputs protected by this stage; a
16-year-old remains eligible for its SPI draws.

The [FRS 2024–25 methodology](https://www.gov.uk/government/statistics/family-resources-survey-financial-year-2024-to-2025/family-resources-survey-background-information-and-methodology)
includes qualifying 16–19-year-olds in its dependent-child definition. This
change preserves under-16 inputs, not every FRS child's inputs. The treatment
leaves the imputation of qualifying older children as an explicit limitation.

Stage 1 consumes the full existing query shape and discards under-16 draws
before assignment. That preserves the random stream shared with the subsequent
base-channel dividend redraw. Both SPI stages preserve the protected under-16
inputs, and the base dividend redraw preserves under-16 FRS dividends.

### Income periods and mappings

The source-year label `2022` represents SPI 2022–23. The FRS loader stores the
2024 period for its [April 2024–March 2025 collection](https://www.gov.uk/government/statistics/family-resources-survey-financial-year-2024-to-2025/family-resources-survey-financial-year-2024-to-2025).
The treatment applies the selected model index at `2024-01-01` divided by that
index at `2022-01-01`, before conditioning stage 2 on the resulting incomes.
This annual convention does not harmonize individual interview months or
establish exact fiscal/calendar alignment. The later 2024-to-2025 projection
is a separate model step.

The installed [PolicyEngine-UK 2.94.0 distribution](https://pypi.org/project/policyengine-uk/2.94.0/)
resolves the following indices. These are modeling growth assumptions, not
HMRC-prescribed growth rates or factors fitted to this experiment's residuals.

| SPI flows | Installed model index | 2022-to-2024 factor |
|---|---|---:|
| Employment components | OBR average earnings | 1.116368286445 |
| Self-employment | OBR per-capita mixed income | 1.021134421134 |
| Private pension | OBR private-pension index | 1.102502017756 |
| Dividends, property, miscellaneous income and OTHERINV | OBR per-capita GDP | 1.092376803703 |
| Taxable savings interest | ONS household interest income | 2.269155046634 |

The old treatment held OTHERINV nominal. [Annex A, physical page 19, note 1](https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/9422_put_2223_full_documentation.pdf#page=19)
and the installed `variables/input/other_investment_income.py` describe the
corresponding residual investment-income concept. The existing stage-1
crosswalk already maps OTHERINV directly to `other_investment_income`; the
publication treatment now uses that variable's declared OBR per-capita GDP
index. GDP remains a proxy for this taxable component's growth.

Six SPI flows remain explicitly nominal because this review did not establish
a same-scope indexed mapping:

| Flow | Retained limitation |
|---|---|
| GIFTAID and GIFTINV | The corresponding charity inputs have no declared uprating index. |
| INCPBEN | The SPI field is taxable-only. The broader model's reported incapacity benefit has a CPI index, but its scope does not establish the required taxable-income mapping. |
| OSSBEN | The field aggregates other taxable social-security payments. |
| UBISJA | The field combines unemployment benefit, income support and Jobseeker's Allowance. |
| SRP | The SPI pension concept includes widow's pensions and certain retirement lump sums. |

[HMRC's PAYE manual](https://www.gov.uk/hmrc-internal-manuals/paye-manual/paye77001)
distinguishes taxable and exempt incapacity-benefit categories. [SPI Annex A,
physical pages 17](https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/9422_put_2223_full_documentation.pdf#page=17)
and [19](https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/9422_put_2223_full_documentation.pdf#page=19)
define the pension field and qualify the treatment of identifiable lump sums.
These scope differences remain visible in the treatment's uprating receipt.

Only an explicit null mapping can produce a `held_nominal` receipt. A declared
mapped variable with a missing, blank or unsupported index, a missing parameter,
or a non-finite/non-positive index value fails rather than silently becoming
nominal. Tests exercise that contract with the actual installed UK engine and
with deliberately incomplete index definitions.

SPI taxable interest is rebased before the build-year FRS tax-free interest is
added once. The treatment also rebases base-channel dividends for recipients
aged 16 and older and reconstructs employment and HMRC accounting aggregates
from the changed leaves. The treatment does not resolve the remaining
interest/ONS scope question in #866 or all period-selection questions in #862.

### Pension-receipt conditioning

The FRS-only forest receives `state_pension_reported > 0` for its training rows
and `hmrc_spi_state_pension_income > 0` for SPI recipients. Given the wider SPI
pension scope, this is a predictive proxy rather than an exact receipt or
amount identity. The forest can still draw a different receipt status. Reduced
discordance alone would not establish entitlement accuracy or consistent
pension amounts. The comparison reports the two pension-income bands and
aggregate pension/tax separately.

The reviewed SPI documentation has SHA-256
`6b39771e6122391c4f0b262dcfb33ec6454264a55638da0f59677696029d2897`;
the source review visually inspected physical pages 12, 17 and 19.

## Matched genuine-build contract

The control and treatment use the same raw FRS inputs, input sample, raw design
weights, source-selection algorithms, generation seeds, build period and
solver settings. Only the reviewed SPI code/specification differs. Both runs
rebuild every downstream stage. Changed SPI incomes can alter downstream
predictors, CGT support and cloning; final row counts and generated prior
weights must therefore be measured, not assumed identical.

The licensed source set is FRS 2024–25, SPI 2022–23, WAS round 8, LCFS 2023–24
and ETB 1977–2024, together with the pinned HMRC income and CGT workbooks.
The SPI donor `put2223uk.tab` has SHA-256
`5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66`.
The exact source-file hashes and build arguments belong in each genuine run's
receipts; no licensed person or household rows are committed. The separate
[#878 CGT treatment](https://github.com/PolicyEngine/microcosm/pull/878) remains
outside this experiment.

| Calibration setting | Shared value |
|---|---|
| Geography and scale | Full national spine; national targets only |
| Year | 2025 |
| Optimizer | Adam, 1,500 epochs, learning rate 0.02 |
| Target weighting | `family_equal` |
| Solver seed | 0 |
| Maximum weight ratio | 10 |
| Gate evaluation date | 7 September 2026 in both actual run receipts |
| Target, measure-exclusion, gate, take-up and CGT-source definitions | Preserve the publication control's definitions |

Both runs use the canonical Chronicle consumer artifact produced from source
`6fb700e`, with 128,717 facts and schema
`policyengine_ledger.consumer_artifact.v2`:

- Facts SHA-256: `6ae49d7d7ab297df25a0b9bfe2d6776827c672d284fbb360957fe8337089549f`.
- Manifest SHA-256: `dcda51d6496aea67f768a284e7955c7520e7c8b91e2bed3569f247567b7153f0`.

The publication loader verifies both hashes. Its runtime compiles the national
reference declarations and then applies the reviewed measure exclusions.
Both genuine runs independently compile 415 active references, exclude the
same 44 measures and calibrate the same 371 targets under registry
`794e11262011`. The paired receipt verifies the complete target names, values,
periods, entities, measure/filter bindings, source metadata and loss weights,
plus runtime, lock, canonical feed and manifest, actual raw-file hashes,
resource pins, seeds, fit-weight algorithms and gate definitions. Names and
values have SHA-256 `a5a76f92ddc56b267505301ba7933f50ef7ce2ba0724f898b4f13d7896c8ffb8`
and `9b96ff695a6b75f66d966cab452aedd06043f56eac4d692261405b571195c16a`.
Changed generated support changes the matrix values; that measured treatment
outcome is not a change to the target contract.

### Reproducing the aggregate receipt

The generator accepts the two retained run directories, this repository and
two explicit local input manifests. Reconstruct their reviewed metadata from
the committed receipt as follows, choosing an output directory outside the
repository and filling in local paths to the licensed inputs and Chronicle
artifact. This extraction uses input metadata only; the generator independently
pins its canonical hashes and never reads a previous numeric output.

```python
import json
from pathlib import Path

receipt = json.loads(Path(
    "experiments/receipts/uk-publication-spi-comparison.json"
).read_text())
local = Path("/path/outside/repository/reproduction")
local.mkdir(parents=True, exist_ok=True)
contract = dict(receipt["comparison_contract"])
contract["raw_source_locations_file"] = str(local / "source-paths.json")
contract["chronicle_path"] = "/path/to/chronicle-uk-6fb700e"
(local / "comparison-contract.json").write_text(json.dumps(contract, indent=2))
(local / "protected-manifest.json").write_text(json.dumps(
    receipt["protected_publication_files"], indent=2
))
raw = receipt["raw_source_verification"]
print(raw["raw_acquisition_repository"], raw["raw_acquisition_revision"])
print(*raw["verified_files"], sep="\n")  # All 21 required input basenames.
```

Create `source-paths.json` with shape
`{"repo": "<raw_acquisition_repository>", "revision": "<raw_acquisition_revision>", "files": {"inputs": ["/absolute/path/to/input1", "/absolute/path/to/input2"]}}`.
Use the repository/revision printed above and list every required basename
under `verified_files`, retaining each actual filename. Groups can have any
name because the reader flattens their path lists. Include the HMRC ODS paths
in `files`, or place those ODS files beside `source-paths.json`. The Chronicle
directory must contain `consumer_facts.jsonl` and `manifest.json` with the
recorded hashes. The generator verifies the actual raw-file sizes/hashes and
both Chronicle files; paths alone do not establish provenance.

Then run the documented CLI in the pinned treatment environment, for example:

```sh
uv run --no-sync python experiments/build_879_publication_receipt.py \
  --repo . --control /path/to/pub879-control --treatment /path/to/pub879-spi \
  --contract /path/outside/repository/reproduction/comparison-contract.json \
  --protected-manifest /path/outside/repository/reproduction/protected-manifest.json \
  --output /path/outside/repository/reproduction/paired-receipt.json
```

Later evidence-only commits may be checked out: the generator binds the
recorded executed source commits `d43b4203` and `fc49b482`, not the current
HEAD. Keep the runtime/source files and locked environment at those reviewed
versions. Both the repository invocation and a standalone generator copy
without an existing receipt directory reproduced the committed receipt bytes.

## Protected publication decisions

Ten of the 11 files carrying the publication stack's additional changes remain
byte-identical: the UK country package, gates and target-fit exclusion register;
the battery bindings, calibration runner, terminal gates and weighted integrity
implementation; the terminal-gate tests; and the data contract and its test.
The remaining file, `test_spec_engine_country_bundles.py`, changes only its
expected UK specification identity from
`0accc39d40d8c8106f4c9bc5562ccca50f1212cd2a8cdf72f416978ca12570ff` to
`e0dd9175920f6d1368a74957367182dc4d6357ffbdb1f728457ead91dd16dd9b`.

Native validation exposed this inherited identity pin after the reviewed SPI
specification changed. The unchanged publication control passed its original
expectation; the shared compiler produced the new treatment identity. The
repair updates that exact expected-hash literal and preserves every assertion
and publication-added test. Reversing the one replacement reproduces the
publication file's recorded bytes. This is a documented correction to the
overnight isolated harness's claim that all 11 files could remain byte-identical.
Source-owned coverage and graph regeneration update their derived evidence;
the publication's gate, register and take-up decisions remain unchanged.

The [existing childcare receipts](https://github.com/PolicyEngine/microcosm/blob/d43b4203c6ebe10e062cb3ef3034e66731ea055d/experiments/834-childcare-tfc-receipts.md)
retain the following choices for both runs:

| Decision | Status retained in this comparison |
|---|---|
| A22/A23 TFC and targeted-childcare rates | Hold at 0.88 and 0.597 pending adjudication. |
| Extended and universal-childcare rates | Retain the landed 0.6054 and 0.4539 values. |
| UC with-children target-fit deferrals | Retain existing reasons and expiry of 30 September 2026. |
| Private-pension £100–150k count deferral | Retain its existing entry during the matched comparison; assess staleness from the resulting receipts. |
| CGT target-fit deferral | Retain #875's existing timing/translation reason and expiry of 5 October 2026. |

The historical expected-count childcare fit pushed TFC toward 0.997 and targeted
childcare to 1.0. Its expected TFC spending and child-count ratios were 0.804 and
1.138; targeted childcare reached 0.852 of its target at the fitted vector.
Those ceiling results explain the recorded holds, not a decision to adopt the
ceiling rates. The committed fit receipt pin is
`c56ca568ef446f07f773ea03253bc2f9d878aeea6bc781ee38ed268d1fedde96`.

No new binding, exclusion, expiry extension, take-up change or gate suppression
is part of the SPI comparison. A release gate becoming stale remains a reported
failure until its existing decision is separately reviewed.

## Native validation

Normal package import and `CountryTaxBenefitSystem` initialization succeed in
both locked environments. The native matrix exercises the shared compiler,
source/schema/coverage contracts, full invented graph against the live legacy
oracle, and actual installed UK-engine uprating. Invented fixtures test execution
and contracts; the genuine builds below test population fit.

| Invocation | Passed | Failed | Skipped | Classification |
|---|---:|---:|---:|---|
| Publication focused contracts, 370 cases | 363 | 1 | 6 | Historical cached-evidence engine pin 2.89.0 versus installed 2.94.0. |
| Treatment focused contracts before pin repair, 386 cases | 378 | 2 | 6 | Same historical failure plus the omitted UK specification hash expectation. |
| Affected country-bundle file after exact pin repair, 18 cases | 18 | 0 | 0 | Resolves the introduced expectation failure; the original invocation remains recorded. |
| Treatment SPI file after formatting, 38 cases | 37 | 0 | 1 | Includes actual engine mapping/failure tests and explicit age 15/16 boundaries. |
| Publication full graph/legacy oracle, 336 cases | 335 | 0 | 1 | Optional US-engine case skips. |
| Treatment full graph/legacy oracle, 336 cases | 335 | 0 | 1 | Same scope and optional skip. |
| Treatment data contract, 229 cases | 229 | 0 | 0 | Checks the downstream specification identity dependency. |

After later targeted reruns supersede the same logical cases, the treatment has
943 passing, one inherited failing and seven skipped distinct cases. The 18- and
38-case reruns are not additional distinct tests. This is not a clean whole-suite
rerun or an all-tests-pass claim. The historical failure is
`test_cached_candidate_regeneration_matches_committed_evidence`; both untouched
publication and treatment reproduce the same cached 2.89.0/installed 2.94.0
engine mismatch. The comparison retains that golden evidence. Source-owned
coverage regeneration/check, changed-file Ruff formatting/lint and test inventory
checks pass. The [native receipt](receipts/uk-publication-spi-native-validation.json)
records invocations, counts and the exact failure classification.

## Genuine build and calibration results

Both full national spines complete all 28 production stages and pass all 15
spine gates. Both 1,500-epoch calibration solves finish. Terminal gate failure
returns exit code 1 and prevents export of `calibrated.h5` in both runs.

Each selected run emits a signed HMAC-SHA256 attestation using an ephemeral
local developer key, with no signing error. These producer signatures are not
production certificates or independently reproducible verification after key
disposal. An earlier control invocation used the wrong key encoding and is
retained separately as an excluded unsigned attempt. Every diagnostics field
outside build provenance matches the selected control rerun exactly. Calibration
and diagnostic readers verify unchanged H5 hashes before/after; an observed
mtime change did not change the control's bytes.

### Fit and concentration

| Metric | Publication control | SPI treatment |
|---|---:|---:|
| Final loss | 0.023464462 | 0.015068903 |
| Targets within 10% | 334 / 371 | 338 / 371 |
| Effective sample size | 5,868.4 | 5,758.9 |
| Top 1% share of calibrated weight | 18.867% | 19.032% |
| Maximum calibrated/prior weight ratio | 10.0 | 10.0 |
| Positive-weight households | 52,846 | 52,846 |
| Total calibrated household weight | 29,848,124.77 | 29,811,148.67 |

Absolute relative error improves for 181 targets, worsens for 189 and is
unchanged for one (equality tolerance 1e-12). Nine targets enter the 10% band
while five leave it. The fixed `family_equal` objective weights families
equally and targets within each family equally; one interest row is the entire
ONS national-accounts family. Its contribution falls from 0.008294907490 to
0.000003178594, accounting for 0.008291728896 of the net 0.008395558522 loss
reduction (98.763%). The remaining 370 targets contribute a net reduction of
0.000103829626. The aggregate objective therefore cannot stand in for a joint
tax/pension/UC success criterion.

Signed target error is `(estimate - target) / target`. The receipt also contains
the target amounts and every before/after estimate.

| Target | Control error | Treatment error |
|---|---:|---:|
| OBR income tax | -12.151% | -12.562% |
| OBR total state pension | -8.136% | -11.695% |
| HMRC pension amount, £20–30k income band | +25.323% | +25.211% |
| HMRC pension amount, £50–70k income band | +34.563% | +16.867% |
| HMRC private-pension count, £100–150k income band | +24.716% | +14.256% |
| ONS savings interest | -18.249% | +0.007% |
| UC households | -13.400% | -12.975% |
| UC single with children | -35.309% | -34.525% |
| UC one child | -28.924% | -31.551% |
| UC two children | -38.872% | -30.671% |
| UC five or more children | -28.600% | -28.618% |
| OBR CGT | +44.180% | +43.744% |
| HMRC gains total | -0.248% | -0.520% |
| HMRC CGT taxpayer count | -0.349% | -0.312% |
| TFC government top-up | -0.136% | +0.299% |
| TFC children with used accounts | -0.148% | +0.381% |
| Targeted childcare, two-year-olds | -0.335% | +0.099% |
| Working-parent childcare, ages 2–4 | +0.033% | -0.072% |
| Universal-only childcare | -0.451% | -0.079% |

Income tax falls from £291.165bn to £289.802bn against a £331.438bn target;
state pension falls from £134.293bn to £129.090bn against £146.186bn. The
£50–70k pension band improves enough to pass the unchanged 25% fit bound,
while the £20–30k band remains just outside it. UC with-children residuals are
mixed and remain substantial. Better source coherence has not resolved the
calibration tension.

### Objective contributions and regressions

The following table includes every target family. Positive reduction means
less contribution to final loss; negative reduction means deterioration.

| Family | Targets | Control contribution | Treatment contribution | Reduction |
|---|---:|---:|---:|---:|
| `obr` | 21 | 0.003864297 | 0.003806251 | +0.000058046 |
| `isc` | 1 | 0.000048858 | 0.000001314 | +0.000047544 |
| `hmrc_salary_sacrifice` | 3 | 0.000204716 | 0.000212828 | -0.000008112 |
| `hmrc_spi` | 129 | 0.001372467 | 0.001048345 | +0.000324122 |
| `hmrc_cgt` | 2 | 0.000135737 | 0.000189119 | -0.000053382 |
| `dwp_benefit_cap` | 1 | 0.000072313 | 0.000025936 | +0.000046377 |
| `dwp_legacy_benefits` | 4 | 0.002312393 | 0.002360476 | -0.000048084 |
| `dwp_universal_credit` | 95 | 0.000884966 | 0.001291308 | -0.000406342 |
| `dwp_two_child_limit` | 15 | 0.000075536 | 0.000043369 | +0.000032167 |
| `ons_population` | 50 | 0.000404484 | 0.000368509 | +0.000035975 |
| `ons_household_composition` | 7 | 0.000025831 | 0.000037483 | -0.000011652 |
| `council_tax_stock` | 18 | 0.002347491 | 0.002300426 | +0.000047065 |
| `scotgov_social_security` | 1 | 0.000085076 | 0.000090198 | -0.000005122 |
| `ons_national_accounts` | 1 | 0.008294907 | 0.000003179 | +0.008291729 |
| `ons_employment` | 1 | 0.000037085 | 0.000020812 | +0.000016273 |
| `ons_land` | 3 | 0.000975697 | 0.000974123 | +0.000001574 |
| `slc_repayments` | 3 | 0.000036065 | 0.000028184 | +0.000007881 |
| `slc_borrowers` | 3 | 0.000021739 | 0.000039943 | -0.000018204 |
| `slc_student_support` | 6 | 0.002018843 | 0.001995919 | +0.000022924 |
| `hmrc_tfc` | 2 | 0.000064663 | 0.000154545 | -0.000089882 |
| `dfe_funded_childcare` | 3 | 0.000124053 | 0.000037850 | +0.000086203 |
| `dft_local_bus` | 2 | 0.000057244 | 0.000038786 | +0.000018458 |

The five largest target-level increases in loss contribution are:

| Target | Control error | Treatment error | Added loss |
|---|---:|---:|---:|
| `dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_27_600_to_28_800@2025` | -0.014% | -100.000% | 0.000478404 |
| `voa.council_tax_stock.band_a@2025` | +22.804% | +28.182% | 0.000135810 |
| `obr.state_pension@2025` | -8.136% | -11.695% | 0.000077043 |
| `hmrc.cgt.gains_total@2025` | -0.248% | -0.520% | 0.000061837 |
| `hmrc/self_employment_income_income_band_50_000_to_70_000@2025` | +19.850% | +35.413% | 0.000054837 |

The UC couple-without-children £27.6–28.8k annual-payment cell has a target of
746.333 benefit units. Its control estimate moves from 586.0 before calibration
to 746.232 afterward; treatment is zero both before and after. All 52,846 final
household weights are positive. For a nonnegative count this implies no positive
realized treatment support in that cell, rather than a zero-weight collapse.
This is an inference from aggregate diagnostics; the prepared matrix was not
retained. It does not identify which SPI or downstream support change removed
the cell. Reweighting its current empty support cannot recover the target.

### Generated support and source quality

| Final generated spine | Publication control | SPI treatment |
|---|---:|---:|
| Households | 52,846 | 52,846 |
| People | 113,626 | 113,590 |
| Benefit units | 61,234 | 61,213 |
| Total household prior weight | 29,247,433 | 29,247,433 |

Both use importance weights after combining source support. Household counts
and total prior mass agree, but the final household ID sets and prior-weight
vectors ordered by household ID differ. The same input sample and design-weight
algorithms therefore do not imply identical final generated support. The full
rebuild propagates the SPI changes into 36 fewer people and 21 fewer benefit
units, including downstream CGT cloning effects.

At final source-spine prior weights, employment assigned to SPI-channel
under-16s falls from £66.176bn to zero; their private pension falls from
£0.515bn to zero. These are imputation-quality diagnostics, not estimates of
actual children's income. Among SPI recipients aged 16 and older, the pension
receipt proxy mismatch affects 3.797% of prior-weighted person mass in the
control. The treatment has fewer than three mismatching records, so its small
cell and weighted amount are suppressed. This does not establish pension
entitlement accuracy, and qualifying FRS children aged 16–19 remain exposed to
the SPI draw as described above.

The UC income screen is `max(0, uc_maximum_amount - uc_income_reduction) > 0`
at 2025 policies and source-spine prior weights. It precedes take-up and other
award conditions; passing it is not complete UC eligibility. For SPI support:

| Family | Control passing mass | Treatment passing mass | Control share | Treatment share |
|---|---:|---:|---:|---:|
| Couple with children | 211,143 | 477,527 | 9.42% | 21.32% |
| Single with children | 385,679 | 709,599 | 27.17% | 49.40% |

The separate CGT treatment remains excluded. Nevertheless, the changed SPI
spine propagates through the unchanged CGT algorithm: prior-weighted total
gains change from £97.351bn to £97.591bn and £5m-plus gains from £24.603bn to
£22.182bn. That is an induced-support result, not evidence for #878's tail
correction. After calibration, OBR CGT remains +43.744% above target, while
HMRC gains and taxpayer counts remain within 1%; #875's vintage/timing
reconciliation remains relevant.

### Gate and decision verdicts

All 15 spine gates pass on both runs. For terminal gates:

| Gate | Publication control | SPI treatment |
|---|---|---|
| Aggregate administration | Passed | Passed |
| Calibration reference coverage | Passed | Passed |
| Target fit | Failed | Failed |
| Weight ESS | Passed | Passed |
| Weight ratio | Passed | Passed |
| Zero-weight strata | Passed | Passed |

The control fails the £20–30k and £50–70k state-pension amount bands and the
stale private-pension exclusion. The treatment removes the £50–70k failure,
retains the £20–30k failure (+25.211%) and adds three failures:

- UC couples without children, £27.6–28.8k annual payment: −100.000%.
- Self-employment income, £50–70k income band: +35.413%.
- Council-tax stock, band A: +28.182%.

The private-pension £100–150k count exclusion is stale in both runs: its
residual is already within the 25% bound in the control (+24.716%) and improves
to +14.256%. The retained entry therefore continues to fail the native stale
decision check. It should be adjudicated separately; deleting it would not
resolve the other target-fit failures. Neither run has expired, premature or
dormant exclusions. UC and CGT reviewed deferrals retain their original reasons
and dates.

A22/A23 remain holds at TFC 0.88 and targeted childcare 0.597; the landed
extended/universal values stay 0.6054/0.4539. The new childcare target errors
are small at these fixed rates, but calibration fit does not re-estimate
take-up or rerun the historical eligible-base ceiling analysis. The experiment
does not justify replacing those decisions. No target, exclusion, gate, expiry,
take-up setting or solver tuning was changed to obtain this comparison.

The shared quality schema/exporter described in the handoff is not present on
the selected publication tree. Its existing `tools/emit_lineage_dashboard.py`
exports US imputation lineage. This aggregate experiment receipt does not
create a shared dashboard contract, and the invented UK dashboard prototype
is not genuine dataset evidence. Both native runs refuse calibrated export;
release publication remains blocked.

Refs [#665](https://github.com/PolicyEngine/microcosm/issues/665),
[#736](https://github.com/PolicyEngine/microcosm/issues/736),
[#796](https://github.com/PolicyEngine/microcosm/issues/796),
[#840](https://github.com/PolicyEngine/microcosm/issues/840),
[#866](https://github.com/PolicyEngine/microcosm/issues/866) and
[#862](https://github.com/PolicyEngine/microcosm/issues/862); independent CGT
work remains in [#878](https://github.com/PolicyEngine/microcosm/pull/878) and
[#875](https://github.com/PolicyEngine/microcosm/issues/875).
