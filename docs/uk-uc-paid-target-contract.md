# UK UC paid-claim target contract

The ten national UC count targets now compare positive model UC with
administrative **paid claims**. Previously their source rows covered all claims,
including open claims with no payment. This repairs the comparison universe;
it does not establish an exact monthly claim history or guarantee a smaller
calibration residual. Related work remains in [#882](https://github.com/PolicyEngine/microcosm/issues/882).

## Source and measurement bindings

[Chronicle #252](https://github.com/PolicyEngine/chronicle/pull/252) supplies
payment indicator × child entitlement crossed separately with family type and
reported child count. The pinned complete UK artifact comes from
`ec7169b5db40b9f54117c80f70f14efc1dd0fedd`: all 100 source suites pass and its
131,450 facts validate. The packaged `uk/national_chronicle_feed.json` records
the facts, manifest and schema hashes. National calibration checks both artifact
digests before compiling targets; local surfaces retain their separate pins.

| Target group | Source operation | Model counterpart |
|---|---|---|
| Paid headline | Within each month, sum five family categories × both child-entitlement states, with payment indicator Yes; average the 12 monthly sums | GB benefit units with annual `universal_credit > 0` |
| Four paid family counts | Within each month, sum entitlement No and Yes for the named family and payment Yes; average 12 months | Positive-UC benefit units classified by the model standard allowance and the reported-child proxy |
| Five paid child-count categories | Average each publisher `total_benefit_units` series with payment Yes and `child_entitlement: all` | Positive-UC benefit units by the existing declared-under-20 child-count proxy |

Unknown family records contribute to the headline. A publisher zero must be
present; it cannot be replaced by an absent cell. The family cube has no
published grand Total, so its headline is explicitly derived. The child-count
cube has publisher Totals over entitlement; these are used directly rather
than reconstructed from independently perturbed detail cells.

The administrative-family measurement follows DWP's standard-allowance rule:
an allowance above the maximum single rate implies a couple. Zero allowance
returns UNKNOWN. It preserves source claimant roles and the structural family
measurement. It does not recover ineligible-partner cases that the model cannot
identify. The four existing family payment-band bindings use the same new family
measurement. [DWP family definition](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Family%20Type.html)

`uc_child_element > 0` is a separate child-entitlement diagnostic. It does not
turn reported children into own qualifying children. No family × child-count
joint is inferred from the two published marginal crosses. Thirty-nine
diagnostic masks expose paid, entitlement, family and child-definition support
without adding objective rows or changing target-family coefficients.

## Years and source coverage

The shipped count targets use January–December 2025. Model/calibration year 2025
is distinct from the raw FRS FY2024/25 vintage: source assembly uses period 2024,
configured monetary inputs extend to 2025, and retained ages are not universally
advanced. UK government model parameters for 2025 generally represent FY2025/26.
An explicit April 2025–March 2026 sensitivity therefore tests a different source
window against the same annual model state. It is not a model-year change.

`monthly_window_average` and `monthly_window_sum_average` require the paired
`source_window` period policy, explicit ordered unique months, and one fact for
every declared month × operand. They reject duplicate, missing, mixed-series or
mixed-publication cells. A sum divides by the **month count**, not the number of
cells. Receipts retain member identities, publication identity and the actual
denominator. Existing period policies retain their calendar/future-period guards.

Only these ten count references acquire 12 months. The other 101 monthly UC
references retain their declared nine-month coverage. TCL remains an April
observation, and the Scotland infant target keeps its separate observation
contract. FY2025/26 includes provisional March 2026 in the pinned source; the
calendar-2025 observations are revised.

## Reconciliation and complete target diff

All 2,584 new joint facts match the archived publisher cubes cell by cell.
All 128,672 shared old/new feed records are byte-identical; the 45 replaced
April–December family records retain their old values under expanded-history
identities. Holding old references fixed, the source update changes **zero**
compiled target values. The repaired contract changes exactly ten values,
with 415 compiled references and the same 366 active rows after existing
exclusions. No non-UC value, target roster or coefficient changes.
The [source-only diff receipt](evidence/uk-uc-882/source-target-diff.json)
records all ten old/CY/FY values, complete compilation counts, source hashes
and monthly reconciliation.

In calendar 2025 the family-derived paid headline is 6,197,311, including
1,168.92 unknown-family claims. Summed child-count publisher Totals differ from
that headline by 1.17 claims on average. The largest monthly within-category
publisher Total minus entitlement-detail sum is eight claims. These observed
disclosure residuals are reported without rewriting source values; they are
not optimizer tolerances.

## Remaining comparison limits

Positive annual UC is a proxy for a mean monthly paid population, not a count
of distinct annual claimants. A zero model award, a reported receipt flag or
take-up probability does not identify an administrative open nil-payment claim.
Those source totals remain separate diagnostics. [DWP payment indicator](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Payment%20Indicator.html)

Reported children, own qualifying children/young people, and a positive child
element remain distinct. The retained age inputs and source date precision do
not establish exact birthdays, April claim status or TCL exception histories.
The existing TCL receipt states its annual-positive-award and birth-year proxies.

Annual recurring model UC divided by twelve does not reconstruct DWP monthly
cash payments, including advances in their issue month. Empty upper payment
bands therefore require an amount-concept and component audit before any
proposal to generate support. [DWP amount definition](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Monthly%20Award%20Amount%20%28bands%29.html)

Public synthetic tests cover grid completeness, literal zeros, mixed identities,
cross-year and direct-key guards, allowance precision, claimant/family roles,
own-unit child definitions and diagnostic partitions. Hash-pinned full-feed
regeneration checks cover national references and unchanged local membership.
Population fit and original-source concentration are separate development
diagnostics; this contract change does not close #882.
