# CGT observation year and individual scope

The UK release retains FRS FY2024–25 observations with `survey_year=2024`,
`base_year=2024` and exported `time_period=2024`. Calibration uses a 2025 registry
index. CGT now declares a separate 2024 measurement period, matching the HMRC
FY2024–25 observations. This release does not require a CGT forecast for 2025.

| Date | Meaning |
| --- | --- |
| 2024 survey/base/output | FRS FY2024–25 source values and exported dataset |
| 2025 calibration index | Registry identity and default measurement year for other targets |
| 2024 CGT observation/measurement | HMRC FY2024–25 individual gains, taxpayers and liability; model gains, tax and annual exempt amount use 2024 |
| 2025 OBR cash period | FY2025–26 cash forecast, retained as diagnostic provenance outside fitting |

The previous gains and taxpayer targets were already observed FY2024–25 totals,
but included trusts (£127.316bn and 584,000). Their measurements could mix stored
base-year gains with model tax in the default 2025 year. The revised contract
selects the provisional Table 1 **individuals** observations from the pinned
Chronicle feed: gains £119.258bn, 551,000 taxpayers and liability £22.503bn. All
three are fitted under registry period 2025 using model measurements in 2024.

Each selector pins its aggregate fact key, source concept, Table 1 dimension,
source entity label, `tax_year` period type and period value 2024. Chronicle's
legacy `tax_unit` label is a fact-selector field; the UK engine measures people
and maps their amounts to households. It does not create a UK tax-unit entity.
Missing or mismatched observations cannot replace these facts. Generated
`uprating_from_period`/`uprating_to_period` fields record the registry identity
hold; they apply no numerical uprating to these observed values.

`cgt_2024_gains` and `cgt_2024_tax` are transient resolver aliases. They force
engine calculation at the declared year even when raw `capital_gains` already
exists in the source. Persisted aliases are rejected to prevent stale values
from bypassing the engine. The taxpayer proxy and gain aggregate use the 2024
annual exempt amount and the same gains array. Gains are the model's net-gains
input; this binding makes no additional deduction for losses or the allowance.
The taxpayer proxy remains gains above the allowance, so it does not identify
every administrative case below that threshold.

The fitted OBR cash row is replaced by the HMRC individual liability row. The
unchanged March 2026 OBR FY2025–26 cash forecast (£21.801546197bn) is required by
the liability compilation as diagnostic metadata, with its exact fact key,
period, source-projection assertion and provenance. A missing or substituted
cash fact fails compilation of the liability target. Cash receipts and disposal
year liabilities have different timing and population scope; this metadata
does not assert a reconciliation between them.

With the committed default exclusions, the roster still has 366 active targets
and 21 families. Under `family_equal`, OBR has 19 rows instead of 20 and
`hmrc_cgt` has three instead of two. Each retained OBR row's objective coefficient
changes from 1/420 to 1/399; CGT gains and count change from 1/42 to 1/63; the
replacement liability row changes from the old cash row's 1/420 to 1/63. These
are consequences of family membership, not optimizer tuning. The historical
OBR exemption remains dormant; the new HMRC liability target has no exemption.

Both national and local calibration resolve the dated CGT arrays before
fitting, then restore the base-year data columns and export the fitted weights.
Restoring source values does not reverse the weight fit. Resolver receipts
record the base period, default calibration period and dated CGT measurements;
the UC claimant validation and receipts introduced by #883 remain intact. The
#881 source `year_rule` resolver governs source acquisition separately and is
unchanged by this CGT contract.

This correction preserves source Tables 2/3, donor selection, carrier counts,
gain amounts, age/region assignment, exclusions and release gates. The existing
Table 3 distribution still represents FY2023–24 and is mapped into the 2024
source build. Historical source-stage descriptions referring to trust-inclusive
or two aggregate calibration targets describe the older target surface; the
current `uk_population_targets.json` is authoritative for the three fitted
individual observations. Distribution target fences remain in force. Source
vintage, thin support, allowance-threshold priors and the model's generic CGT
rate/relief representation can still produce residuals after this year and
scope correction. Neither unit tests nor a national fit certify a dataset.

The implementation uses the current main lock (PolicyEngine-UK 2.97.0, Core
3.31.0) and the existing Chronicle 6fb700e feed. A separate prerequisite adds
the two FRS person-role inputs consumed by the #883 UC capital stage to its
graph projection. A clean source build demonstrated their omission. The same
prerequisite is applied to control and candidate; it changes no CGT imputation.
