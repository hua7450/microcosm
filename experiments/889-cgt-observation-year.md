# Observed FY2024–25 CGT calibration comparison

The observed-year contract brings taxpayer count and liability close to the
individual HMRC observations, while gains remain 9.41% low. These are national
fits on one unchanged source H5. They combine the measurement-year correction,
individual scope, liability comparator and automatic family weighting changes;
they do not isolate the effect of any one change.

| Common FY2024–25 individual measure | HMRC target | Under control weights | Under candidate weights | Control error | Candidate error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Taxpayers | 551,000 | 583,538.709 | 551,037.296 | +5.9054% | +0.0068% |
| Gains (£bn) | 119.258 | 127.484860 | 108.031570 | +6.8984% | −9.4136% |
| Liability (£bn) | 22.503 | 26.986389 | 22.653870 | +19.9235% | +0.6704% |

Both columns evaluate the same captured 2024 outcomes. The control's native 2025
cash-target error is a different comparison and is not substituted into this
table. Cash receipts have different timing and population scope from disposal-year
individual liability.

The [JSON receipt](../docs/evidence/uk-cgt-889/calibration-comparison.json) contains
full-precision values, exact HMRC and OBR identities, options, hashes, original
gate results, concentration metrics and all 22 changed objective weights. The
[363-row CSV](../docs/evidence/uk-cgt-889/target-comparison.csv) contains every
unchanged non-CGT target, both estimates and both errors. Every corresponding
non-CGT target value and household coefficient row was checked identical.

## Trade-offs and remaining gates

Gains' absolute error worsens from 6.90% to 9.41%; being just inside the 10%
diagnostic threshold does not establish an adequate gains model. Household ESS
falls from 14,049.762 to 13,725.441. Among the same 707 gain-above-allowance carriers,
ESS falls from 490.704 to 384.586. The 692 FRS origin groups do not imply 692
independent CGT donors. These fits remain sensitive to their limited support.

Among 363 non-CGT rows, the number within 10% rises from 347 to 348, and both fits
have 361 within 25%; neither threshold gains a new miss. The two empty childless
couple UC payment bands remain at −100%. Both original runs exit with a failed
terminal gate and no staged calibrated H5. Five stale UC/pension exemptions also
appear in both original failure receipts. Their disposition is separate from
this PR's retirement of the dormant OBR cash exemption. Original failed receipts
are preserved, including the historical dormant OBR entry in the candidate.

The diagnostic HDF audit verifies all 218 original person/benefit-unit/household
columns and exact captured fitted weights at base period 2024. That audit is not
a passing release export. The local integration tests cover the resolver and
HDF route on synthetic cases; no full local calibration is certified here.

## Scoped supersession and historical record

For the accepted FY2024–25 release, three individual observations are the fitted
CGT acceptance unit. This supersedes #875's previous FY2025–26 fit disposition
only for that observed-year deliverable. The OBR March 2026 FY2025–26 forecast
remains unchanged as optional diagnostic provenance, with explicit unavailable
status if its exact fact cannot be used. No general payment-lag translation,
forward-year gains level or replacement distribution prior is inferred.

The JSON receipt archives the exact former `obr.capital_gains_tax@2025` signed
exemption, including its original reason, approver and approval/expiry dates.
The active register removes only that entry and supplies no exemption for HMRC
liability. This records a scoped implementation disposition, not a new human
signature, an amendment to issue #875, or closure of its reconciliation work.

The frozen incumbent parity fixture remains historical evidence. Current signed
differences identify the new individual values, the added HMRC liability row and
the cash row absent from the fit; they do not overwrite the incumbent's values.
The original national calibration battery excludes release-cut-owned compile
parity. The new real-feed 2023/2025 compile-parity checks are separate validation;
the recorded national fits do not establish parity or current release readiness.

## Provenance and reproduction

Numerical candidate: `9b5ac7c52f8a3742940674dd2b58a7595ea24db6`.
Control: `cc9c953c72b003406f4aeec8ae5e82bd211099bc`.
Source H5 SHA256:
`03a63ecc1d0b3064d42e8b412fa4a28c587a79bd153ff2ffebc2715f4b316e33`.
Both use UK 2.97.0, Core 3.31.0 and the same locked environment; the receipt records
the full lock and installed-version identities. National Chronicle is `ec7169b5`:
facts `4a50ee9568a01bbb57f73d927084ed6b4b9e52249b51a2338455874ae6e382b5`,
manifest `a95d0ee9f87f36947eaecdb3de29cf81a91e47ccaa822fed42da677eedca877f`.
Local Chronicle remains independently pinned to `6fb700e`.

Both solves use 1,500 Adam updates, learning rate 0.02, seed 0, `family_equal`,
free household mass, cap 10, no sparsity penalty and `best_feasible_loss` iterate
selection (control epoch 1333; candidate 1422). The defaults are not overridden
in the library. The effective fitted roster has 366 targets and 21 families.
Under this treatment, 19 retained OBR weights change from 1/420 to 1/399;
gains/count change from 1/42 to 1/63; cash-to-liability changes from 1/420 to 1/63.

A reviewer with authorized access to the exact source can use
`tools/calibrate_uk_national_dataset.py` in the two pinned checkouts, with the
recorded H5 and Chronicle SHA arguments, `--epochs 1500 --learning-rate 0.02
--target-weight-rule family_equal --target-loss-cap 10`, separate output paths
and non-release identifiers. The two runs must start from identical household
IDs and initial weights. Preserve the original gate failures; do not bypass
them to obtain an apparent passing release. Private saved matrices/weights are
identified by SHA in the receipt and are not published here.

For public contract verification, authenticate the national artifact against its
committed manifest, run `tools/build_uk_ledger_compile_parity_signed_differences.py
--surface national --ledger-facts <national-facts.jsonl> --output-dir <scratch>`,
and compare the two national receipts. The full-feed test also applies the real
2023 and 2025 compile-parity gates. National and local reference regeneration
must each use their own authenticated feed. The committed CSV and JSON expose
full-precision estimates, targets, errors and concentration statistics for audit.
Recomputing weighted outcomes or ESS from microdata requires the authenticated
private captures; those records are not included in this public receipt.

Post-run review repairs alter optional diagnostic availability, the active
exemption register, parity explanations, tests and this evidence. They do not
change the captured source, fitted target values or numerical result arrays.
The original 30479731/6fb run remains historical; current claims use 9b5/cc9.
Source-vintage, age support, tail allocation and the generic liability mechanism
remain separate, unimplemented follow-ups.
