# PR 879: matched national comparison after measurement corrections

> Integration update, 7 September 2026: the second rebase is onto `main`
> `5ab1b056f4d3e8a71873b13fe566f0bb899899b6`, which includes merged PR #874.
> Ten publication prerequisite commits dropped because they are already merged;
> four earlier fit-deferral/clock commits remain carried by this branch.
> The review-fix patch replayed unchanged at code tip
> `de606c218cb8446dc2b81477cbae56aae3cf897c` (previously `9ac8d9c1`).
> This integration brings upstream `659adb1a` fiscal-year target selection,
> DfT geography and take-up-contract fixes. The historical calibrations below
> were **not rerun for this head**: their original measurement/spine code pins,
> artifact identities and numerical results remain the scope of the evidence.
> Post-rebase code validation is separate from those historical calibration runs.

> Historical first-rebase note, 7 September 2026: at this stage PR #879 targeted
> `main` and was rebased onto
> `396df96fe7889777f3a7dd737e3305298219e6fa`. The four SPI/measurement and evidence
> commits replayed with identical patches; their rebased tip was `4f000458`.
> At that stage the branch also carried the 14 unmerged UK publication-stack prerequisite
> commits on which these experiments depend. That first rebase changed no executable
> Python behavior relative to the previously tested PR head `4ccaabba`.
> That first rebase left calibration receipts, target/exclusion registers and the
> lockfile unchanged. This note does not describe the base of later integrations.
> The calibrations below were not rerun for the rebase: their original source
> commits and artifact identities remain the provenance of the reported results.
> Historical rebase validation (7 September, UK 2.94.0 / Core 3.31.0 installed):
> 458 focused tests passed with four skips; repository Ruff
> lint and CI test-inventory verification passed, and all 51 changed Python files
> passed the formatting check. This is not a full-suite or certification claim.

The England-scope and FRS-composition corrections are implemented and the matched national calibrations are complete. The UC published-band mismatch and native entitlement limitations below remain unresolved, so this is a conditional comparison, not complete administrative measurement validation. The corrected SPI comparison reduces loss by **39.81%**, but remains release-blocked. England council-tax measurements now fit closely. The two high-payment childless-couple bands are empty in both corrected spines, establishing a support gap rather than a treatment-only regression.

## What changed

These are the changes measured in the retained comparison. The later upstream integration described above has separate code-validation results and no new calibration results.

- National VOA bands A–H and total explicitly require `country == ENGLAND`, matching their `E92000001` facts. The selected `E92000001` facts cover England within the broader England-and-Wales [VOA publication](https://www.gov.uk/government/statistics/council-tax-stock-of-properties-2025/council-tax-stock-of-properties-statistical-commentary). Local VOA and Scottish bindings are unchanged. The existing use of household counts as a dwelling-stock proxy remains separate from this geography correction.
- The UC payment-distribution and family-composition measurements use retained FRS claimant/parent roles rather than a generic adult count. This preserves cohabiting couples and prevents an older child becoming a partner. UC child counts include reported child members under 20 and native UC child/qualifying-young-person status, excluding the claimants themselves. The five child-count targets use the same child definition.
- SPI price-year corrections, income draws, take-up, target values, all 44 measure exclusions, reviewed fit-exclusion decisions, solver settings and gates are unchanged.

This is a measurement correction. Every UC award in each rerun was checked bit-for-bit against its original full-model award before solving. The underlying UK model still has a separate standard-allowance relationship error; relabelling calibration families does not repair entitlement.

## Controlled experiment and matrix verification

Both runs reuse byte-identical copies of the complete 2024 spines already built on the publication control and SPI treatment. The common measurement code is `117860efe96b4fc1e336e0c9bfab10917cd3aa59`. The control spine originated at `d43b4203c6ebe10e062cb3ef3034e66731ea055d`; the SPI spine at `fc49b48200e1e15fe35bb4e15b948992bf03c28d`. Each receipt run now links its explicit `spine_origin_code_pin` to the authenticated earlier comparison receipt and the matching original/copied spine hashes; `code_pin` remains the executed measurement commit. Both calibrate 2025 using UK 2.94.0 / Core 3.31.0, Chronicle `6fb700e`, 371 identical active targets, 1,500 epochs, family-equal loss weights, learning rate 0.02, maximum weight ratio 10 and seed 0.

Before either optimizer ran, all nine actual VOA matrix rows were asserted equal to the appropriate England indicator derived independently from the nine English regions. The historical run script also asserted all 84 active UC payment-band rows against the corrected family masks, actual band bounds and benefit-unit-to-household links. Those families came from the runtime resolver, so that check was not an independent relationship-classification oracle. The original preflight recorded all 100 band inputs/supports but omitted individual UC exact-match flags. Full-register edges were retained so existing exclusions do not widen adjacent bands. The comparison receipt verifies unchanged initial weights and bit-identical matrix rows outside the authorized VOA and UC-composition changes. Original and copied spine hashes remain equal after both runs.

## Reproducing the retained-evidence audit

The [evidence builder](build_879_corrected_measurement_receipt.py) reads the original receipts from reachable immutable ancestor `a20d8c31936d5d0d4d193891d2c11d74217a038f` and verifies their fixed SHA-256 pins. It authenticates every retained diagnostics, gate, preflight, support, solve and matrix artifact against those committed hashes, and each spine against the earlier source receipt. It then writes a fresh [comparison receipt](receipts/uk-corrected-measurement-comparison.json) containing 84 explicit exact saved-matrix row comparisons per run, with family, bounds, positive-payment predicate, benefit-unit support and nonzero-household counts. It preserves every earlier result and gate decision.

```bash
uv run --no-sync python experiments/build_879_corrected_measurement_receipt.py \
  --repo . --control /local/measurement_comparison/pub879-control \
  --treatment /local/measurement_comparison/pub879-spi \
  --output /local/corrected-receipt.json
```

The retained restricted artifacts must be available locally. The builder authenticates the private composition snapshots before loading them, but their hashes were first recorded during this review repair. They contain saved runtime-resolver families and awards: their new hash pins do not establish independent historical correctness. The new row audit is distinct from the original pre-solve assertions and does not rerun a model or optimizer. Public output contains allowlisted aggregates and hashes, without person/household identifiers or restricted tables.

The engine-free `test_uk_uc_family.py` regression supplies independent labelled relationship scenarios, tests every included lower edge and adjacent floating-point values, retains the 16 excluded edges while materializing all 84 active rows, and checks that two eligible benefit units in one household contribute two. It covers young claimants, cohabiting partners and children aged 18, 19 and 20. Its high-award case explicitly records the known open-last-band mismatch; passing tests and saved-matrix equality do not certify the published administrative bounds.

## Results

| Corrected run | Final loss | Targets within 10% | Effective sample size |
|---|---:|---:|---:|
| Control | 0.021724747 | 337 / 371 | 6,127.6 |
| SPI treatment | 0.013076866 | 341 / 371 | 6,110.6 |

The previous and corrected runs use different measurement definitions, so the within-pair SPI contrast is the controlled comparison. The four columns below show how correcting those definitions also changes the apparent baseline problems.

| Target | Previous control error | Previous SPI error | Corrected control error | Corrected SPI error |
|---|---:|---:|---:|---:|
| VOA England Band A | +22.804% | +28.182% | +0.049% | -0.027% |
| VOA England total dwellings | +15.930% | +15.787% | +0.036% | +0.010% |
| Income tax | -12.151% | -12.562% | -12.067% | -12.709% |
| Total state pension | -8.136% | -11.695% | -7.909% | -11.430% |
| UC single with children | -35.309% | -34.525% | -31.365% | -27.923% |
| UC total benefit units | -13.400% | -12.975% | -11.189% | -10.006% |
| UC childless couples, £27.6–28.8k | -0.014% | -100.000% | -100.000% | -100.000% |
| UC childless couples, highest band | +0.270% | +0.228% | -100.000% | -100.000% |
| OBR capital gains tax | +44.180% | +43.744% | +44.919% | +44.819% |

England Band A now has weighted estimates of 6,106,286 and 6,101,679, against 6,103,320. These are actual recalibrated estimates, superseding the earlier fixed-weight England-only diagnostic. Other targets still compete through the common household weights. Interest targets account for 91.5% of the net objective gain, so lower total loss should not be treated as uniform improvement in joint tax/benefit quality.

## Reassessing the high-payment childless-couple bands

The formerly decisive two duplicated records are classified as lone-parent benefit units in both runs. Their awards remain £27,646.60 in the control and £27,489.43 in the SPI treatment. Neither provides support for the corrected childless-couple distribution. The correction changes family classification for 769 / 768 benefit-unit rows, of which 144 / 154 have positive UC in control / treatment.

The support columns below show **benefit-unit rows / unique source benefit units**. Duplicated or SPI-cloned rows are not independent source cases.

| Annual band (rounded bounds) | Status | Control support | SPI support | Control error | SPI error |
|---|---|---:|---:|---:|---:|
| £20,400–£21,600 | Active | 8 / 4 | 6 / 3 | +0.06% | +0.52% |
| £21,600–£22,800 | Active | 2 / 1 | 2 / 1 | +0.03% | -0.03% |
| £22,800–£24,000 | Active | 2 / 1 | 2 / 1 | +0.68% | +0.59% |
| £24,000–£25,200 | Active | 6 / 3 | 6 / 3 | +0.45% | +0.06% |
| £25,200–£26,400 | Active | 4 / 2 | 4 / 2 | -0.24% | -0.31% |
| £26,400–£27,600 | Excluded | 0 / 0 | 0 / 0 | -100.00% | -100.00% |
| £27,600–£28,800 | Active | 0 / 0 | 0 / 0 | -100.00% | -100.00% |
| £28,800+ (highest labelled band) | Active | 0 / 0 | 0 / 0 | -100.00% | -100.00% |

There are 14 / 13 active payment-band targets with fewer than five unique source benefit units in control / treatment. All 100 bands, including excluded bands, have row counts, unique-source counts and weighted results in the companion receipt. The old two-row fit is not the success baseline. These tails need genuine support; a good weighted fit on one duplicated source case would still be fragile.

## Gate verdict

Both reruns remain blocked by the unchanged `uk_target_fit` gate and refuse calibrated H5 export. The corrected SPI run's unreviewed target-fit failures are:

| Target | Corrected SPI error |
|---|---:|
| `dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_27_600_to_28_800@2025` | -100.000% |
| `dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_28_800_to_30_000@2025` | -100.000% |
| `hmrc/self_employment_income_income_band_50_000_to_70_000@2025` | +34.485% |
| `hmrc/state_pension_income_band_20_000_to_30_000@2025` | +27.941% |

The existing reviewed private-pension count exclusion for the £100–150k income band is stale in both runs because the result is back inside its bound. The exclusion for UC households with exactly two children is also stale in the corrected treatment. Both remain unchanged in this controlled comparison. The receipt records the full gate statuses. This is not a certified dataset or a release-readiness claim.

### Historical UC fit-deferral rationales

All four UC entries in the [reviewed target-fit register](../packages/microcosm-build/src/microcosm/build/uk/target_fit_reviewed_exclusions.json) rely on the older #813 composition measurement. Their support, take-up and wealth-draw causal explanations remain historical hypotheses requiring fresh adjudication under the corrected composition; this report does not renew them.

| Existing entry | Corrected control error | Corrected SPI error | Review status |
|---|---:|---:|---|
| `dwp.uc.households_single_with_children@2025` | -31.365% | -27.923% | Claimant/partner membership changed; #813 support and take-up attribution needs fresh adjudication |
| `dwp.uc.households_children_1@2025` | -26.692% | -29.053% | Revised reported-child membership changes this cell; historical wealth/take-up attribution needs fresh adjudication |
| `dwp.uc.households_children_2@2025` | -34.568% | -21.525% | Revised child membership; now inside 25% in SPI, making the entry stale; fresh adjudication required |
| `dwp.uc.households_children_5_or_more@2025` | -28.504% | -28.532% | Revised child definition also governs this cell; similar residuals do not validate the historical causal attribution; fresh adjudication required |

Each of the four UC register reasons now retains its original wording verbatim followed by a dated historical-basis annotation, including the two-child entry’s stale treatment result. Every approver, adjudication reference, approval/expiry date and entry remains unchanged. The annotations require fresh adjudication and grant no renewed approval. The receipt field `protected_policy_resource_sha256["target_fit_reviewed_exclusions.json"]` retains `93cd8b1d1e45a4ed0a1d8e96da36072d5b201a7f9ad6d88342b21a716a5108d2`, the checksum of the historical pre-annotation register used by those runs. It is not a checksum of today’s annotated register with commit-pinned report links. The current resource differs only in the four explanatory reason strings; the historical hash is not refreshed. The 25% fit bound and all gate/exclusion decisions remain fixed for this comparison. Neither the #840 source review nor this evidence repair supplies a new input-mass parity waiver, known-gap approval or release authorization. Gate-override source observability is a separate unresolved suggestion; this repair neither adds an override nor changes its reporting.

## UC target inventory and remaining measurement limitations

There are **100 payment-band references: 25 × four family types; 84 active and 16 excluded**. There are also 11 active general UC targets and 15 active two-child-limit targets, giving **110 active UC-related targets** out of 371 overall. Two OBR UC expenditure references are compiled but excluded; including these and the 16 excluded bands gives 128 compiled UC-related references. The separate inventory lists every band, target value and activation status.

[DWP payment-band metadata](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Monthly%20Award%20Amount%20%28bands%29.html) defines 26 positive-payment categories for the relevant period, including a separate £2,500.01-or-over monthly tail. This register includes only the first 25 for each family. The current interpreter multiplies source lower edges by 12 and takes the next lower edge as its exclusive upper bound: the first interval is [£0.12, £1,200.12), not an exact transcription of the published inclusive-currency endpoints. The highest **included** band becomes [£28,800.12, infinity), despite its bounded £2,400.01–£2,500 monthly source label. For example, a £36,000 annual award enters this bounded-source row even though the source has a separate open-tail category. Full-register edges preserve excluded-cell boundaries; exclusion pruning does not cause this mismatch.

The catalog omission and finite-band endpoint semantics remain release-blocking measurement debt. [#736](https://github.com/PolicyEngine/microcosm/issues/736) already tracks omitted no-payment/or-over source rows and is the existing follow-up location. A repair must define currency endpoint handling, bind the omitted tail to its own facts and rerun the matched comparison. Renaming or pooling the current row would not establish the missing source totals. This report preserves the existing 100-row catalogue and all numerical results.

DWP family types use the awarded standard-allowance single/couple rate and verified reported children under 20. Couples with an ineligible partner can appear as single in the administrative classification; relationship counts alone cannot reproduce that rule. The new FRS relationship measure is a better composition proxy than generic age-only categories, but it does not resolve the pinned model's couple-allowance treatment of some parents with older children, nor partner-ineligibility exceptions or all administrative child-reporting differences. DWP child reporting can also include multigenerational arrangements beyond child-element eligibility. The source review traced FRS head/parent/dependent-child construction, but did not independently adjudicate the vintage FRS role dictionary. In particular, the `famtypb2` mapping and downstream native-model relationship behavior need a separate source-dictionary and entitlement review; no completed model repair is claimed here. These limitations prevent interpreting the corrected family cells as exact administrative matches. [DWP Family Type metadata](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Family%20Type.html), [DWP child-count metadata](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Number%20of%20Children.html).

The exact-vintage [FRS 2024–25 glossary](https://www.gov.uk/government/statistics/family-resources-survey-financial-year-2024-to-2025/family-resources-survey-background-information-and-methodology#benefit-unit-or-family) supports recognising cohabiting couples and dependent children aged 16–19: FRS adults are people aged 16+ who are not dependent children; older dependent children must meet partnership, residence and education/training conditions. A benefit unit comprises a single adult or married/cohabiting couple and dependent children. Its survey head is the household reference person where present, otherwise the first interview-listed member; that designation does not establish UC claimant eligibility. The remaining dictionary gap is narrower: the numeric `uperson`, `depchldb` and `famtypb2` codes, missing-code conventions and exact adult/child file routing were not independently validated. `is_parent` is a generated adult-file role proxy, not a verified raw biological-parent field.

### Redraw and nominal-income limitations

UC reporter redraw still uses the legacy `is_married`/qualifying-young-person predictors, `has_non_child_member` screen and transition categories. Calibration now uses the FRS relationship composition proxy. For example, the legacy redraw can call a lone young claimant `child_only` while calibration calls that claimant `SINGLE`; cohabiting-couple categories can also differ. The redraw implementation/settings and the already generated support are held fixed when remeasuring the two retained spines. This preserves the conditional experiment but leaves a support-generation and downstream classification limitation. Aligning redraw, adjudicating FRS source roles, fixing the native standard-allowance relationship problem and handling partner eligibility are separate substantive work; none is fixed by relabelling the calibration rows.

The SPI `hmrc_spi_state_pension_income` leaf remains nominal from 2022 even though the pinned engine exposes a state-pension uprating index. That index was not selected for this treatment; there is no evidence here that holding the leaf nominal is preferable. The retained leaf can affect HMRC assessable-income bands and component amounts. A separate uprating sensitivity remains unmeasured, so the treatment does not establish complete monetary alignment. The original [#840 experiment](840-spi-income-coherence.md) records the selected scope.

## Validation

Historical measurement validation on 7 September 2026, with UK 2.94.0 / Core 3.31.0 installed, covered 305 distinct passing tests and two existing skips after updating two stale country-resource-roster assertions to include the already-registered reviewed target-fit exclusion resource. The initial measurement suite had 111 passes and two skips; the additional country-package/calibration/gate suite had 192 passes and two inherited roster failures, both then repaired and rerun successfully. This is not a full-repository test claim. Repository Ruff, CI test-inventory verification and the pinned-feed target-reference regeneration test passed in that historical scope. The separate 458-pass/four-skip rebase validation above covered a later focused set; neither count describes engine-free CI. No release gate, exclusion policy, take-up input or original spine was altered.

Review-repair validation on 7 September, before the second rebase, is separate: the two SPI test files pass without engines on Python 3.13 and 3.14 (39 passes, 14 skips each), and with the existing UK 2.94.0 / Core 3.31.0 environment (52 passes, one skip). Real-engine tests retain `requires_uk` coverage; parsed/path-equivalence coverage stubs deterministic uprating data and runs without an engine. The independent UC file passes five tests; the evidence-builder file passes 15 tests, including tamper refusal for all retained inputs and exact-row/privacy checks. The retained-evidence builder completed both 84-row audits without rerunning calibration. These are focused checks, not data certification.

Post-rebase validation on 7 September covers the integrated code based on `main` at `5ab1b056f4d3e8a71873b13fe566f0bb899899b6`, including the upstream fiscal-year selection, DfT geography and take-up-contract changes:

| Scoped check | Environment | Passed | Skipped |
|---|---|---:|---:|
| 13 affected test files | Python 3.13, no country engines | 588 | 19 |
| Same 13 affected test files | Python 3.14, no country engines | 588 | 19 |
| Five affected test files | Existing UK 2.94.0 / Core 3.31.0 environment | 81 | 2 |

These scopes overlap and are reported separately. The earlier 113-file fast group was interrupted for the user-requested `main` update; it did not complete and is not counted as a full-suite pass. The completed checks validate code contracts, not a recalibration or certified dataset.
