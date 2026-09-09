# Every UC calibration target in the PR 879 national comparison

The compiled catalogue contains **100 UC payment-distribution targets: 25 included monthly award bands for each of four family types**. **84 are active; 16 have existing reviewed exclusions.** The measurement corrections do not change these targets, target values, exclusions or band edges.

| Family | Catalogue bands | Active bands | Excluded bands |
|---|---:|---:|---:|
| Single, no children | 25 | 19 | 6 |
| Single, with children | 25 | 16 | 9 |
| Couple, no children | 25 | 24 | 1 |
| Couple, with children | 25 | 25 | 0 |
| Total | 100 | 84 | 16 |

There are also **26 active UC-related targets outside the payment distribution**: 11 DWP caseload/composition targets and 15 two-child-limit targets. That makes **110 active UC-related targets overall**, out of the 371 active national targets. The two OBR UC expenditure targets are compiled but excluded. Counting those and the 16 excluded payment bands gives **128 compiled UC-related references**. The general `dwp_universal_credit` loss family alone has 95 active rows; the two-child-limit targets belong to a separate loss family.

## All 100 included payment-band target values

Values below are the target number of UC benefit units, averaged across the pinned April–December 2025 observations; they are not payment amounts. **† means excluded from the solve under the existing register.** The annual column reproduces the readable target labels.

| Published monthly award band | Annual target label | Single, no children | Single, with children | Couple, no children | Couple, with children |
|---|---|---:|---:|---:|---:|
| £0.01–£100.00 | £0–£1,200 | 89,479.3 | 26,670.2 | 7,506.3 | 19,073.6 |
| £100.01–£200.00 | £1,200–£2,400 | 127,403.2 | 36,208.1 | 8,797.0 | 22,775.8 |
| £200.01–£300.00 | £2,400–£3,600 | 197,136.7 | 46,312.8 | 12,256.0 | 26,502.7 |
| £300.01–£400.00 | £3,600–£4,800 | 377,063.1 | 60,310.1 | 10,945.4 | 29,027.0 |
| £400.01–£500.00 | £4,800–£6,000 | 195,983.2 | 67,858.7 | 12,492.4 | 32,427.0 |
| £500.01–£600.00 | £6,000–£7,200 | 114,033.3 | 89,577.4 | 12,979.7 | 34,956.9 |
| £600.01–£700.00 | £7,200–£8,400 | 149,043.8 | 116,076.0 | 16,484.4 | 37,410.8 |
| £700.01–£800.00 | £8,400–£9,600 | 399,750.2 † | 97,511.0 | 15,272.4 | 39,910.3 |
| £800.01–£900.00 | £9,600–£10,800 | 386,411.7 † | 102,343.0 | 19,142.3 | 39,368.0 |
| £900.01–£1,000.00 | £10,800–£12,000 | 104,891.1 | 123,718.1 † | 13,313.3 | 40,805.4 |
| £1,000.01–£1,100.00 | £12,000–£13,200 | 122,035.6 | 125,798.4 † | 16,121.4 | 40,009.6 |
| £1,100.01–£1,200.00 | £13,200–£14,400 | 197,163.2 † | 129,133.7 † | 13,317.3 | 39,972.6 |
| £1,200.01–£1,300.00 | £14,400–£15,600 | 209,497.3 † | 115,856.7 † | 19,657.3 | 40,518.8 |
| £1,300.01–£1,400.00 | £15,600–£16,800 | 149,963.2 | 125,660.2 † | 16,443.2 | 37,118.0 |
| £1,400.01–£1,500.00 | £16,800–£18,000 | 108,689.7 | 135,836.8 † | 13,441.1 | 37,177.8 |
| £1,500.01–£1,600.00 | £18,000–£19,200 | 61,718.7 | 135,444.9 † | 10,728.6 | 37,164.6 |
| £1,600.01–£1,700.00 | £19,200–£20,400 | 34,050.4 | 106,341.9 † | 9,653.2 | 33,309.6 |
| £1,700.01–£1,800.00 | £20,400–£21,600 | 22,338.3 | 85,490.4 | 6,486.9 | 29,102.6 |
| £1,800.01–£1,900.00 | £21,600–£22,800 | 14,685.8 † | 80,226.9 † | 3,981.9 | 27,534.7 |
| £1,900.01–£2,000.00 | £22,800–£24,000 | 11,133.7 | 71,259.6 | 2,761.4 | 25,584.8 |
| £2,000.01–£2,100.00 | £24,000–£25,200 | 6,951.2 | 51,301.7 | 2,131.2 | 22,927.8 |
| £2,100.01–£2,200.00 | £25,200–£26,400 | 6,508.2 | 43,305.2 | 1,475.3 | 20,334.7 |
| £2,200.01–£2,300.00 | £26,400–£27,600 | 5,913.4 | 34,910.6 | 1,073.8 † | 17,535.0 |
| £2,300.01–£2,400.00 | £27,600–£28,800 | 1,884.1 † | 28,513.0 | 746.3 | 15,399.9 |
| £2,400.01–£2,500.00 | £28,800–£30,000 | 1,653.8 | 22,593.0 | 605.8 | 12,836.2 |

The ordinary bins are £100 a month wide, annualised for the model as £1,200. The underlying source labels begin at £0.01, £100.01, etc.; the materializer multiplies the lower edge by 12 and uses the next band's lower edge as an exclusive upper bound. Thus the first effective interval is [£0.12, £1,200.12), and zero UC is excluded. Removing a target through the exclusion register does not widen neighbouring bands because edges come from the full pre-exclusion catalogue.

**Upper-tail caveat:** the highest included bin is £2,400.01–£2,500.00 a month, with the readable annual label £28,800–£30,000. The current materializer instead gives that last compiled band an infinite upper bound: [£28,800.12, infinity). This is an existing source/measurement mismatch, held constant in the requested comparison; the JSON records the actual upper bound as null. The table must not be read as proof that the solver stops counting UC at £30,000. [DWP publishes](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Monthly%20Award%20Amount%20%28bands%29.html) 26 positive-payment categories for months from September 2022, including a separate £2,500.01-or-over tail omitted from this 25-bin register. A £36,000 annual award therefore enters a row whose source label is bounded. The next-lower-edge convention also differs from the published inclusive-currency endpoints by the annualised penny offsets. The catalog and boundary repair belongs with [#736](https://github.com/PolicyEngine/microcosm/issues/736), which already tracks omitted no-payment/or-over source rows, and requires new matched measurements; these historical targets and results remain unchanged.

These narrow cells reflect the source catalogue, not sample-size-driven bin selection. A cell can have hundreds of target benefit units but only one unique source case in the spine. The corrected comparison report therefore includes unweighted and unique-source support, alongside weighted fit, for every band. The calibration recipe does not automatically pool thin bands.

## All other UC-related compiled targets

Counts are benefit units, children or affected households as named. OBR expenditure targets use their native monetary units rather than counts.

| Target | Target value | Status |
|---|---:|---|
| `obr.universal_credit_in_cap` | 66,410,800,011.5 | Excluded |
| `obr.universal_credit_outside_cap` | 12,875,549,128.1 | Excluded |
| `dwp.uc.households_children_1` | 1,278,537.0 | Active |
| `dwp.uc.households_children_2` | 1,100,617.6 | Active |
| `dwp.uc.households_children_3` | 494,553.2 | Active |
| `dwp.uc.households_children_4` | 174,668.9 | Active |
| `dwp.uc.households_children_5_or_more` | 77,376.8 | Active |
| `dwp.uc.households_single_no_children` | 3,446,962.0 | Active |
| `dwp.uc.households_single_with_children` | 2,226,220.0 | Active |
| `dwp.uc.households_couple_no_children` | 284,345.6 | Active |
| `dwp.uc.households_couple_with_children` | 899,534.7 | Active |
| `dwp.uc.scotland_households_child_under_1` | 14,333.2 | Active |
| `dwp.uc.two_child_limit.households_affected` | 469,780.0 | Active |
| `dwp.uc.two_child_limit.children_affected` | 597,920.0 | Active |
| `dwp.uc.two_child_limit.children_in_affected_households` | 1,665,540.0 | Active |
| `dwp.uc.two_child_limit.households_3_children` | 297,310.0 | Active |
| `dwp.uc.two_child_limit.children_in_3_children_households` | 891,930.0 | Active |
| `dwp.uc.two_child_limit.households_4_children` | 117,190.0 | Active |
| `dwp.uc.two_child_limit.children_in_4_children_households` | 468,760.0 | Active |
| `dwp.uc.two_child_limit.households_5_children` | 37,020.0 | Active |
| `dwp.uc.two_child_limit.children_in_5_children_households` | 185,120.0 | Active |
| `dwp.uc.two_child_limit.households_6_plus_children` | 18,260.0 | Active |
| `dwp.uc.two_child_limit.children_in_6_plus_children_households` | 119,720.0 | Active |
| `dwp.uc.two_child_limit.households_claimant_pip` | 65,280.0 | Active |
| `dwp.uc.two_child_limit.children_claimant_pip` | 235,270.0 | Active |
| `dwp.uc.two_child_limit.households_disabled_child_element` | 129,630.0 | Active |
| `dwp.uc.two_child_limit.children_disabled_child_element` | 479,460.0 | Active |
| `dwp.uc.households` | 6,758,888.9 | Active |

## Definitions and provenance

DWP's family-type statistics use single/couple standard-allowance classification and reported child presence. Since April 2019 the child-presence category includes reported children or young people under 20, beyond those eligible for the UC child element. [DWP Family Type metadata](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Family%20Type.html), [DWP Number of Children metadata](https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Number%20of%20Children.html).

The corrected Microcosm composition measure uses retained FRS claimant/parent roles and reported under-20 child members, together with the native UC child/qualifying-young-person flag. It is a proxy for the administrative categories: the native model's standard-allowance relationship error and partner-eligibility exceptions are not repaired by relabelling the calibration groups. Those limitations are distinct from the target count and bin widths.

Source: the full compiled register and unchanged exclusion register for the pinned Chronicle `6fb700e` feed, calibration year 2025, PR 879 measurement comparison. No licensed source-record identifiers are included.
