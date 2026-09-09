# SPI recipient age, income year and pension receipt

> Historical experiment, 6 September 2026. These results use the
> `bc11803f915fe72afd724515a27a3863035f4efc` baseline, PolicyEngine-UK 2.92.1
> and the older Chronicle feed. The [original report at source head
> `194111af`](https://github.com/PolicyEngine/microcosm/blob/194111af32b265b226dcbefe33cc735f362461dc/experiments/840-spi-income-coherence.md)
> and the paired JSON identify that experiment. Its numeric results remain
> unchanged below; they do not measure the later publication baseline.
>
> Source correction, 7 September 2026: [SPI Annex A, physical page 12](https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/9422_put_2223_full_documentation.pdf#page=12)
> labels the first age band “Under 25”; it does not establish a survey minimum
> of 16. The treatment's boundary of 16 follows its constructed donor-age
> support. It preserves under-16 FRS inputs, while qualifying FRS children
> aged 16–19 remain exposed to imputation.
>
> The [publication comparison](879-publication-spi-comparison.md) uses
> PolicyEngine-UK 2.94.0, the newer Chronicle feed and a reviewed OTHERINV
> mapping to the model's other-investment-income index. OTHERINV remained
> nominal in this historical treatment. The historical paired JSON is unchanged.

The rebuilt SPI support assigned donor incomes to under-16 records, mixed
2022–23 SPI flows with 2024 FRS flows, and imputed a second pension receipt status
without conditioning on the SPI pension leaf. This historical treatment reduced
national calibration loss by 36.6% and brought both failing HMRC pension-income
bands inside 25%. Aggregate income tax and state pension worsened, however, and
the single-parent UC shortfall remains large. This is an incremental improvement,
not a resolution of the full calibration tension in #840, #796 and #736.

This experiment starts from `bc11803f915fe72afd724515a27a3863035f4efc`
(`uk-spine-rebuild-842-850`). The separate CGT experiment, #878, starts from the
same commit; neither treatment contains the other. PR #877's size-selection
work is not added to either treatment.

## Plan and implementation

1. Reproduce the unchanged full national spine and the reported v19 calibration.
2. Apply the recipient guard at 16, matching the constructed SPI donor-age
   support. Whole-household support clones include under-16s. Preserve their
   observed FRS inputs in both SPI stages and their base-channel dividends
   during the SPI dividend redraw. This is a modeling boundary; the original
   attribution of a 16 survey minimum to Annex A was incorrect.
3. Rebase stage-1 monetary draws from the donor's 2022 period to the spine's 2024
   period before conditioning the FRS fill on those incomes. Declare each column's
   corresponding PolicyEngine variable and record its index and realized factor.
4. Add a binary pension-receipt predictor to the FRS-only fill: observed
   `state_pension_reported > 0` in training, and drawn
   `hmrc_spi_state_pension_income > 0` for recipients. This bridges receipt status;
   it does not equate FRS regular pension with the full taxable SPI pension amount.
5. Rebuild every downstream stage and calibrate to the identical national surface.
   Report all targets and gate failures, including regressions.

The stage-1 forest consumes its legacy full query shape and discards under-age
draws before assignment. This preserves the adult draw pool and the subsequent
base dividend random stream before the explicit income rebasing. Stage 2 queries
recipients aged 16 and older. Base dividends for those recipients receive the same rebasing as SPI dividends.
Tax-free savings interest drawn from the build-year FRS is added *after* taxable
SPI interest is rebased, so it is not uprated twice. Employment and HMRC income
identities are reconstructed from the changed leaves.

The authoritative YAML, JSON compatibility projection, operation schema, stage
evidence, release-input coverage and graph fixtures describe the same treatment.

## Income-year assumptions

The pinned PolicyEngine-UK 2.92.1 indices give these 2022-to-2024 factors:

| SPI flows | Model index | Factor |
|---|---|---:|
| Employment components | OBR average earnings | 1.116368 |
| Self-employment | OBR per-capita mixed income | 1.021134 |
| Private pension | OBR private-pension index | 1.102502 |
| Dividends, property and miscellaneous income | OBR per-capita GDP | 1.092377 |
| Taxable savings interest | ONS household interest income | 2.269155 |

The large interest factor reflects the selected model index; it is not fitted
to this experiment's calibration residual. Other investment income, charitable
deductions and the broader SPI taxable-benefit/pension leaves are explicitly
held nominal where no model-variable mapping is chosen. Those choices are
recorded, not silently treated as complete monetary alignment. In particular,
this does not settle the taxable-interest/ONS scope mismatch in #866 or all
period-selection questions in #862.

## Comparison contract

Both runs use PolicyEngine-UK 2.92.1, PolicyEngine-Core 3.31.0, Python 3.13.14,
the locked environment, the full FRS spine and 10,000 SPI support households.
Calibration uses year 2025, 364 targets, 1,500 Adam epochs, learning rate 0.02,
`family_equal`, seed 0, free total mass and maximum weight ratio 10. Targets,
measure exclusions, target-fit exclusions, tolerances and take-up are unchanged.

Chronicle commit `1cab80987a462e00055f259cc56dc6b311c030bf` supplies the frozen
facts, SHA-256 `226358e73e7c449e71a3f6dc91a72e8e3941e0f14265943d81c990edb21c2a6c`.
The control reproduces v19's loss (0.025315), income tax (£290.991bn) and
pension-band errors (+26.8%, +33.0%). The newer Chronicle feed was rejected by
this branch's series-selection contract and is not mixed into the comparison.

Source inputs were resolved from UK-data revision
`25af520a6651b8812fef56964a42a79a3f9f515a` and checked against the existing pins.
No licensed rows are committed. The CGT URL now redirects to different workbook
bytes; the exact pinned 2025 workbook was recovered from the
[National Archives snapshot](https://webarchive.nationalarchives.gov.uk/ukgwa/20250730164651id_/https://assets.publishing.service.gov.uk/media/6878ac62760bf6cedaf5bd93/Table_3_2025_Size_of_gain_by_income.ods),
SHA-256 `8e75c00bab949348a7238fea6d995f626c85e5d02813b46606dd7fea85e9d0c3`.
Its vintage is unchanged.

## Source quality

Pre-calibration employment income on SPI under-16s falls from £66.176bn to zero.
These are synthetic-support totals, including downstream copies and their prior
household weights, not an estimate of actual UK children's earnings. The fix
retains observed under-16 inputs rather than imposing zero income on every child.
Adult SPI pension-receipt disagreement between the two source leaves falls from
3.797% of weighted adult support to less than 0.01%.

The UC screen below is `max(0, uc_maximum_amount - uc_income_reduction) > 0`,
evaluated at 2025. It is an income screen before take-up, not a full eligibility
adjudication. Family composition uses the engine's UC child/qualifying-young-person
definition and `is_married`; the separate DWP child-count concept remains open.

| SPI family | Control screened share | Treatment screened share |
|---|---:|---:|
| Single with children | 27.17% | 49.40% |
| Couple with children | 9.42% | 21.32% |

The CGT algorithm and source are unchanged in this branch, but its downstream
assignments can change when incomes and family support change. For example,
pre-calibration £5m+ gains change from £24.603bn to £22.182bn. That movement is
not evidence that this branch implements #878's tail correction.

## National calibration results

The [paired receipt](receipts/uk-spi-income-coherence.json) includes all 364
before/after target estimates, source and diagnostics hashes, implementation
file hashes, source-quality aggregates and terminal failures. Experimental runs
used the parent commit plus the recorded implementation files; the parent code
pin alone does not identify the uncommitted treatment used in those runs.

| Metric | Control | SPI treatment |
|---|---:|---:|
| Final loss | 0.0253150 | 0.0160433 |
| Targets within 10% | 90.934% | 92.033% |
| Effective sample size | 5,722.6 | 5,687.6 |
| Top 1% share of weight | 18.901% | 19.050% |
| Income tax | £290.991bn | £289.180bn |
| Income-tax error | −12.203% | −12.750% |
| Total state pension | £134.387bn | £128.912bn |
| Total state-pension error | −8.071% | −11.816% |
| HMRC pension amount, £20–30k income band | +26.795% | +19.890% |
| HMRC pension amount, £50–70k income band | +32.999% | +16.643% |
| ONS savings-interest error | −16.876% | +0.013% |
| UC single-with-children count | 1.469691m | 1.484661m |
| UC single-with-children error | −33.611% | −32.935% |
| OBR CGT error | −24.975% | −24.739% |

Loss improves by 36.6%, with a substantial improvement in interest matching and
the two pension-band errors. Under `family_equal`, the single ONS interest target
accounts for about 95.7% of the net loss reduction. This concentration matters
when interpreting the headline objective. The aggregate tax and pension regressions mean
that better overall loss does not establish a complete solution to their joint
fit with UC. The single-parent UC count rises by only about 15,000 despite the
larger increase in pre-calibration income-screen support.

The solve completes, but the terminal battery still blocks release export. New
target-fit failures are the couples-without-children UC payment cell
£27,600–28,800 (−100%, no realized model support), self-employment income in the
£50–70k band (+34.604%), and council-tax band A (+28.884%). The private-pension
count exclusion for £100–150k is now stale. Existing exclusions still include
other substantial residuals, visible in the full receipt. None were added,
extended or removed to obtain an apparently passing run. No calibrated release
artifact was exported or certified.

## Tested alternatives

The rank-preserving proposal in #840 was implemented and rebuilt before being
removed. Within demographic and equal-weight cells, it permuted intact SPI
adult vectors onto FRS income ranks. It increased UC-support overlap but did
not improve the national objective:

| Variant | Final loss | Income tax |
|---|---:|---:|
| Child guard plus rank matching | 0.0262020 | £287.008bn |
| Same, with pension-receipt bridge | 0.0260768 | £287.546bn |
| Child guard and pension bridge, without rank matching or rebasing | 0.0256650 | £285.701bn |
| Final treatment, with rebasing and base-child dividend guard | 0.0160433 | £289.180bn |

These comparisons motivate the chosen treatment but are not a complete factorial
decomposition: the last step also adds the base-child dividend guard. The final
branch contains no rank permutation. The generic weighted-QRF bootstrap issue
in #481 and SPI's incomplete representation of non-taxpayers remain separate.

## Validation and reproduction

The focused SPI, manifest, source-stage, release-coverage and graph-parity suites
pass: 158 tests passed and 3 optional input/engine cases skipped. New regressions
cover child-input preservation, the shared dividend random stream, pension-receipt
predictors, income rebasing and avoiding double uprating of tax-free interest.
Repository Ruff lint, formatting of changed Python files, and test-inventory
checks pass. Repository-wide formatting still reports 121 inherited files,
also present on the unchanged control. A broader coverage-regeneration
test has a pre-existing golden mismatch: the committed evidence records
PolicyEngine-UK 2.89.0 while the locked runtime is 2.92.1. The same failure was
reproduced on the untouched control; its historical golden was not rewritten.
The remaining schema, reemission, field-usage, coverage and UK graph checks pass:
60 passed and 1 skipped, with that known failing case explicitly deselected.

Use separate control and treatment directories with the same licensed input
root, running `uv sync --all-packages --locked --extra uk` and
`tools/build_uk_frs_spine.py` with the pinned FRS, SPI, HMRC, CGT, WAS, LCFS and
ETB files. The exact build command is also documented in #878's
[reproduction section](https://github.com/PolicyEngine/microcosm/blob/uk-cgt-spine-quality-725/experiments/uk-cgt-open-tail-quality.md#reproduction).
Set `POPULACE_FIT_N_JOBS=2`, `POPULACE_FIT_PREDICT_WORKERS=2` and the numerical
library thread limits to 2, as in the recorded runs.

Build the Chronicle UK consumer artifact at the stated Chronicle commit and
pass it, its facts/manifest hashes and the spine hash to
`tools/calibrate_uk_national_dataset.py` with `--epochs 1500`,
`--target-weight-rule family_equal`, a `dev-` release ID and a local developer
signing key. Preserve diagnostics and gates even when the terminal command
exits nonzero. This is national-only evidence at the declared seeds; it does
not establish local-area accuracy or robustness across every seed.

Refs #840, #866, #862, #481, #736, #796, #665. The remaining CGT questions in
#725 and #875 are described separately in #878.
