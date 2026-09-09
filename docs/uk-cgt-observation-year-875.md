# CGT observation year and individual scope

The FY2024–25 dataset uses HMRC's three individual observations for that year:
551,000 taxpayers, £119.258bn gains and £22.503bn liability. Calibration remains
indexed at 2025 and exported source data remains dated 2024. The contract pins
exact Chronicle keys, individual scope and `tax_year: 2024`; trust-inclusive
national totals cannot satisfy these references.

`measurement_period: 2024` forces the CGT outcome calculation to use 2024 rather
than the default calibration period, and disables the stored-column shortcut.
The dated aliases bind a variable and year, refuse a mismatched binding, and may
not be persisted in the source H5. Gains and the taxpayer proxy share the 2024
annual exempt amount (£3,000) and gains array. The proxy selects gains above the
allowance; it does not identify every administrative taxpayer below that
threshold. The gains input is net of losses before the allowance; the aggregate
binding makes no additional loss or allowance deduction.

## Observed-year fit and cash diagnostic

For this accepted FY2024–25 deliverable, the three published individual
observations replace the former trust-inclusive gains/count and OBR cash fit.
This explicitly supersedes the former FY2025–26 solve disposition described in
[#875](https://github.com/PolicyEngine/microcosm/issues/875) **for this observed-year
acceptance unit**. It does not amend or close that issue, invalidate its earlier
deliverable, or settle the general self-assessment liability-to-cash translation.
The forward-year reconciliation remains separate work.

The unchanged March 2026 OBR FY2025–26 cash forecast (£21.801546197bn) remains a
separately pinned diagnostic outside fitting. Its exact key, fiscal year,
source-projection assertion, value and source survive when the fact is available.
Absent, incompatible or duplicate diagnostic data produces explicit `unavailable`
metadata, the expected identity and a reason, with no numeric substitute. It does
not remove any independently valid HMRC observation. A malformed committed
cash declaration remains a compilation error; so do missing, duplicate or
incompatible fitted HMRC observations. No cash row returns to the matrix.

The former active `obr.capital_gains_tax@2025` fit exemption is retired because
that row is no longer fitted. Its exact historical reason, approver,
adjudication and dates are preserved in the
[aggregate evidence](evidence/uk-cgt-889/calibration-comparison.json).
This retirement creates no new human signature, renewal or expiry. All other
exemptions and gate thresholds are unchanged; HMRC liability has no exemption.

The default roster remains 366 targets across 21 families. With `family_equal`,
OBR has 19 rows instead of 20 and `hmrc_cgt` has three instead of two. Each retained
OBR row changes from weight 1/420 to 1/399; gains/count change from 1/42 to 1/63;
the replacement tax row changes from 1/420 to 1/63. These 22 changes follow family
membership. The comparison therefore combines dating, individual scope, liability
comparator and objective weighting; it is not an isolated causal year effect.

## Dating, revision and materialization

The #881 `year_rule` mechanism resolves release-relative survey/calibration years
for source construction, policy lookups and predictor materialization. A fixed
administrative observation must keep its date when the release advances, so its
`measurement_period` is a separate target contract. A future shared generic
variable/period resolver may simplify the implementation; automatically replacing
2024 with the survey year would change the present observation's meaning.

The provisional HMRC observations use exact aggregate keys. A different-key
revision alongside the old observation leaves the old key selected. Removing the
pinned key or duplicating it fails compilation. The authoring field
`matched_fact_count_at_or_before_period` is descriptive metadata, not a runtime
revision guard. A lone altered value under the same key is accepted by the
compiler if artifact verification is bypassed; immutable facts and manifest
hashes prevent that alteration in the authenticated build path.

To adopt a revision, authenticate a new Chronicle artifact, review its year,
population and values, deliberately update changed keys and feed hashes, and
regenerate references, membership, relevant public fixtures and parity receipts.
Rerun strict-selection, real-feed and dated-export checks; refresh numerical
evidence if fitted values change. Never loosen selectors to choose the latest
publication silently.

Both national and local paths resolve the dated arrays before materialization.
The rowwise builder resolves each block, aligns by entity IDs, injects the alias
columns and then builds the target matrix. Scoring uses the same resolve/inject
route. The adapter's refusal of an unresolved alias is intentional. The national
and local parametrized integration tests exercise the real resolver with distinct
2024/2025 outcomes, fit weights and restore the original base-year HDF columns;
a separate UK2.97 engine case verifies the calculation date. The source-column
restoration does not reverse the fitted weights. These tests do not certify a
full local calibration.

## Source, dependency and evidence boundary

The lock remains PolicyEngine-UK 2.97.0 / Core 3.31.0. National targets use Chronicle
`ec7169b5db40b9f54117c80f70f14efc1dd0fedd`; local targets retain the separate `6fb700e`
artifact. The four CGT/OBR facts are identical across those feeds. Current #891
paid UC source windows and family classification are preserved.

Source Tables 2/3 remain the 2025 publication's FY2023–24 distribution, mapped into
the 2024 build. Donor selection, carrier count, gain amounts, ages and geography
are unchanged. Older source-stage prose describes the preceding two-target or
trust-inclusive surface; the current target contract controls the three fitted
individual observations. Distribution fences remain active. No source-vintage,
tail, age or liability-model repair is implemented here.

The required #883 graph prerequisite declares the two FRS claimant-role inputs
and updates the graph fixture. While [#892](https://github.com/PolicyEngine/microcosm/pull/892)
remains open, those inputs stay in this PR; its exact graph regression replaces
the overlapping test and resolves that duplicated regression. The separate
exemption-retirement changes still require composition if both PRs land.
Once upstream includes the prerequisite, rebase and remove the duplicated
prerequisite commits while retaining the upstream coverage.

The [current matched comparison](../experiments/889-cgt-observation-year.md)
records the 9b5/ec7 candidate and cc9/ec7 control on the identical source. Its
£108.032bn gains remain 9.41% below HMRC, despite liability and taxpayer totals
being close. The older 30479731/6fb comparison is historical. Neither national
fit, diagnostic export nor unit tests certify a releasable dataset.
