"""UK weighted-integrity gates: input-mass parity and QRF tail concentration.

Increment 4 of the UK parity plan (#609, under the #578 acceptance rule)
ports the two US integrity gates that were each purchased with a named
incident:

- **Input-mass parity** (#278): a rebuilt or selected artifact can zero an
  untargeted input column while every calibrated target still fits. The gate
  compares weighted per-column totals against a frozen reference; total loss
  fails at any tolerance.
- **QRF tail concentration** (#462): a weighted QRF can broadcast a
  donor-tail point mass so a handful of records carry most of a column's
  shipped weighted mass while the paired count target is hit exactly. The
  tell is weighted-mass concentration, which neither support clipping nor
  count targets can see.

Both gates reuse the shared implementations in :mod:`microcosm.build.gates`
verbatim — the UK wrappers only add the evidence plumbing for the national
table layout, the reference identity record, and the universal
reviewed-exclusion discipline (mandatory reason, dormant entries reported,
stale entries fail).

Thresholds carry **no committed defaults**: the US numbers are calibrated to
US incidents and the #609 measurement pass has not yet adjudicated UK
boundaries. Arming either gate requires explicit policy values; once the
measurement numbers are adjudicated, the schema-4 manifest parameters should
carry the same derivation-comment discipline as the weight-ratio threshold.

What the first measurement pass against the pinned enhanced-FRS incumbent
(sha ``584ae33d…``, 2026-08-04) established, so that no later reader
reintroduces the US constants by default:

* **The US 0.75 top-share threshold is wrong for this surface.** At the US
  ``top_k=100`` / ``min_nonzero_records=500`` settings the incumbent itself
  exceeds 0.75 on 16 of the 28 checked columns. UK reported-benefit columns
  sit on small, high-intensity subpopulations, so concentrated weighted mass
  is their normal state rather than a defect.
* **No single global top-share threshold is both incumbent-compatible and
  incident-catching.** The lowest no-headroom threshold that passes the
  incumbent on a broad surface is ~0.996, which would not have caught the
  #462 incident that motivated the gate (89% of mass in 100 of 2,295
  carriers). Tightening it instead requires raising ``min_nonzero_records``
  far enough to drop half the surface.
* **The incumbent cannot supply the QRF baseline at all.** Eleven stage-1
  outputs (every ``hmrc_spi_*`` column plus ``other_investment_income``) and
  five stage-2 disability columns are absent from it, and the seven stage-1
  columns it does carry hold *survey-reported FRS values, not donor draws*.
  Concentration measured there is a different distribution from the one the
  gate exists to police, so the QRF threshold must come from a staged
  candidate where the columns are actually QRF draws.
* **A magnitude floor is not a coherent materiality filter here.** The
  reference surface mixes units: currency totals into the trillions,
  weighted person counts in the millions, and flag counts in the tens of
  thousands. Inheriting the US ``1e9`` floor would stop checking 50 of 131
  columns, including ``gift_aid`` and ``charitable_investment_gifts`` — the
  two the release input coverage manifest specifically requires
  distributional effective mass for.
  A floor of ``0.0`` still skips the three exact-zero reference columns,
  because the shared gate compares ``<=``. So ``0.0`` is the adjudicated
  floor; it is not committed as a constant yet only because a policy arms
  both thresholds together and the tolerance is still unmeasurable.
* **The certified compact is not a valid candidate against this reference,
  confirming the #609 reading of option 2.** Measured against the pinned
  incumbent it carries 8.1% less household mass (28,840,551 versus
  31,389,678), a different surface (150 columns against the reference's 131,
  22 of them candidate-only), and per-column drift with a 22.3% median and a
  578% maximum — it is a differently calibrated build, not a mass-preserving
  derivative. It also zeroes ``gift_aid`` and
  ``charitable_investment_gifts`` and lacks all eleven ``hmrc_spi_*``
  columns, because it is the *input* to the HMRC/SPI stage that fills them:
  those are stage-1 QRF outputs. Comparing the two therefore reproduces the
  #327 failure mode on the restoration surface, exactly as #609 predicted.
  Both thresholds must come from a staged candidate, whose licensed UKDS
  inputs (SPI donor SN 9422, the HMRC surface, and a raw FRS directory) are
  the remaining blocker.

The comparison production actually runs — the shipped release against the
next shipping of the same kind, the US ``_export_input_mass_gate`` pairing
whose reference #327 adjudicated to a certified release — has a measured
baseline. The only consecutive certified UK pair
(``populace-uk-2023-72aeefc-20260611``, sha ``f489b7ef…``, against
``…dd68c73-4aa4b14-20260619T023711Z``, sha ``f17306cc…``; both verified
before reading) shows:

* **The incident class this gate exists for passed cleanly.** Zero columns
  dropped, zero columns zeroed across the shipping; four columns were added
  (reported, never failing). A #278-style silent loss between certified
  releases would have failed at *any* tolerance, with no threshold judgment
  involved.
* **A drift tolerance cannot be tight yet.** The 143 shared nonzero columns
  moved with median |drift| 10.52% and maximum +2,973.71%
  (``adult_ema``), dominated by intentional reported-benefit
  repopulation. Any tolerance between 5% and 50% would have failed 22–89
  columns of a reviewed, correct shipping. The no-headroom boundary for the
  release arm is therefore the pair's exact maximum, ``29.737060`` — wide
  enough to pass every certified intentional movement, and a hard fence
  against anything larger, while the real teeth (zero/absence) are
  tolerance-independent. As consecutive same-code shippings stabilise
  toward calibration-scale drift, the boundary ratchets *down* in reviewed
  steps, never up, re-measured from each newly certified pair with the
  measurement tooling.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.gates import (
    GateResult,
    input_mass_parity_gate,
    tail_concentration_gate,
)
from microcosm.build.input_mass import input_mass_totals
from microcosm.build.uk_runtime.national_frame import UK_EXPORTED_WEIGHT_COLUMNS
from microcosm.frame import Frame

__all__ = [
    "UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE",
    "UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE",
    "UK_INPUT_MASS_PARITY_GATE_NAME",
    "UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256",
    "UK_QRF_TAIL_CONCENTRATION_GATE_NAME",
    "UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE",
    "UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE",
    "UKInputMassParityPolicy",
    "UKInputMassReference",
    "UKInputMassReferenceDescriptor",
    "UKQRFTailConcentrationPolicy",
    "UKReviewedExclusion",
    "UK_INPUT_MASS_REFERENCE_REGISTRY",
    "coerce_input_mass_reference_registry",
    "coerce_reviewed_exclusions",
    "exclusion_evaluation_date",
    "load_uk_input_mass_reference",
    "load_uk_reference_scoped_exclusion_register",
    "load_uk_reviewed_exclusion_register",
    "uk_default_input_mass_reviewed_exclusions",
    "uk_default_qrf_tail_reviewed_exclusions",
    "uk_input_mass_parity_gate",
    "uk_input_mass_totals",
    "uk_qrf_tail_concentration_columns",
    "uk_qrf_tail_concentration_gate",
]

UK_INPUT_MASS_PARITY_GATE_NAME = "input_mass_parity"
UK_QRF_TAIL_CONCENTRATION_GATE_NAME = "qrf_tail_concentration"
UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE = "input_mass_reviewed_exclusions.json"
UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE = "qrf_tail_reviewed_exclusions.json"
UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE = "degenerate_reviewed_exclusions.json"
UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE = "target_fit_reviewed_exclusions.json"

# Canonical sha256 of {"reference": {"identity": ..., "totals": ...}} for
# the weighted input surface emitted from the pinned enhanced-FRS artifact by
# build_uk_efrs_parity_reference.py. The totals remain uncommitted under the
# UKDS EUL; this reviewed digest lets the gate and publication contract bind
# them without disclosing them.
UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256 = (
    "fd41cb5f6cf6c4ef812320f21d1942173d49ce6f8725b21fbc9d9ca5423d298c"
)


_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class UKInputMassReferenceDescriptor:
    """Named pinned reference for the UK input-mass parity comparison."""

    name: str
    filename: str
    revision: str
    sha256: str
    vintage: str
    totals_sha256: str
    scope_note: str

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "filename",
            "revision",
            "sha256",
            "vintage",
            "totals_sha256",
            "scope_note",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"UK input-mass reference descriptor {field_name} must be "
                    "a non-empty string."
                )
            if value != value.strip():
                raise ValueError(
                    f"UK input-mass reference descriptor {field_name} must not "
                    f"carry surrounding whitespace; got {value!r}."
                )
        for field_name in ("sha256", "totals_sha256"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(
                character not in _SHA256_HEX for character in value
            ):
                raise ValueError(
                    f"UK input-mass reference descriptor {field_name} must be "
                    "a lowercase sha256."
                )

    @property
    def identity(self) -> dict[str, str]:
        """The schema-1 sidecar identity mapping shape."""

        return {
            "filename": self.filename,
            "revision": self.revision,
            "sha256": self.sha256,
            "vintage": self.vintage,
        }

    def spec_payload(self) -> dict[str, object]:
        """The gates.json registry entry shape."""

        return {
            "identity": self.identity,
            "totals_sha256": self.totals_sha256,
            "scope_note": self.scope_note,
        }


_UK_INPUT_MASS_REFERENCE_DESCRIPTOR = UKInputMassReferenceDescriptor(
    name="efrs-post-calibration",
    filename="enhanced_frs_2024_25.h5",
    revision="a9e52499b6a6cca100a5ce4f36ca27b2e8a213df",
    sha256="e433e532b17bd8ce76030156285816e33d44e93edabd2204adbef71d19a68712",
    vintage="2024_25",
    totals_sha256=UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256,
    scope_note=(
        "Channel-blind post-calibration enhanced-FRS production incumbent, "
        "pinned to the 2024-25 line; its artifact carries the SPI-synthetic "
        "rows structurally but no admin-restored mass in the "
        "SPI-channel-exclusive columns, so those columns are comparable "
        "only through per-reference reviewed exclusions."
    ),
)

UK_INPUT_MASS_REFERENCE_REGISTRY = MappingProxyType(
    {_UK_INPUT_MASS_REFERENCE_DESCRIPTOR.name: _UK_INPUT_MASS_REFERENCE_DESCRIPTOR}
)

_REVIEWED_EXCLUSION_FIELDS = frozenset(
    {"reason", "approved_by", "adjudication", "approved_on", "expires_on"}
)


@dataclass(frozen=True)
class UKReviewedExclusion:
    """One admitted exclusion and its complete approval receipt (#610).

    The schema is exclusion-generic: every register entry — whichever gate
    it belongs to — records the reasoning, who approved it, the adjudication
    it descends from, when it was approved, and when it expires. The whole
    record is sealed into the policy digest, so editing an approver or
    extending an expiry moves the pinned literal. The in-force window is
    enforced at gate evaluation (never at load — the policy of record stays
    loadable on its expiry date): an entry suppresses from ``approved_on``
    through ``expires_on`` and outside that window the gate fails with an
    explicit correct-or-renew message, so neither a typo'd future approval
    nor a lapsed one can suppress anything silently.
    """

    reason: str
    approved_by: str
    adjudication: str
    approved_on: str
    expires_on: str

    def __post_init__(self) -> None:
        for name in ("reason", "approved_by", "adjudication"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"UK reviewed exclusion {name} must be a non-empty string."
                )
            if value != value.strip():
                raise ValueError(
                    f"UK reviewed exclusion {name} must not carry surrounding "
                    f"whitespace (the raw value is sealed into the policy "
                    f"digest); got {value!r}."
                )
        parsed: dict[str, date] = {}
        for name in ("approved_on", "expires_on"):
            value = getattr(self, name)
            try:
                parsed[name] = date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"UK reviewed exclusion {name} must be an ISO date "
                    f"(YYYY-MM-DD), got {value!r}."
                ) from exc
            # fromisoformat also accepts compact (20270210) and week-date
            # forms; the receipt pins the canonical rendering because the
            # raw string is sealed into the policy digest — two spellings
            # of one date must not produce two digests.
            if parsed[name].isoformat() != value:
                raise ValueError(
                    f"UK reviewed exclusion {name} must be an ISO date "
                    f"(YYYY-MM-DD), got {value!r}."
                )
        if parsed["expires_on"] <= parsed["approved_on"]:
            raise ValueError(
                "UK reviewed exclusion expires_on must be after approved_on; "
                f"got approved_on={self.approved_on!r}, "
                f"expires_on={self.expires_on!r}."
            )

    def expired(self, now: date) -> bool:
        """Honored through ``expires_on``; expired strictly after it."""

        return now > date.fromisoformat(self.expires_on)

    def premature(self, now: date) -> bool:
        """In force from ``approved_on``; premature strictly before it."""

        return now < date.fromisoformat(self.approved_on)

    def policy_payload(self) -> dict[str, str]:
        """The complete record, sealed into the policy digest."""

        return {
            "reason": self.reason,
            "approved_by": self.approved_by,
            "adjudication": self.adjudication,
            "approved_on": self.approved_on,
            "expires_on": self.expires_on,
        }


def coerce_reviewed_exclusions(
    values: object, *, label: str
) -> dict[str, UKReviewedExclusion]:
    """Validate a schema-2 exclusion mapping into typed records.

    Accepts ``None`` (no exclusions), already-typed records, or the raw
    register entry mappings. Raw entries are fully validated here;
    already-typed records pass through on ``isinstance`` because the frozen
    dataclass validated their fields at construction — so a register cannot
    bypass the discipline by construction order.
    """

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} reviewed exclusions must be a mapping.")
    invalid_names = sorted(
        repr(name)
        for name in values
        if not isinstance(name, str) or not name.strip() or name != name.strip()
    )
    if invalid_names:
        raise ValueError(
            f"{label} reviewed exclusions need non-empty, trimmed column names: "
            f"{invalid_names}."
        )
    records: dict[str, UKReviewedExclusion] = {}
    for name, entry in values.items():
        if isinstance(entry, UKReviewedExclusion):
            records[name] = entry
            continue
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"{label} reviewed exclusion {name!r} must be an object with "
                f"fields {sorted(_REVIEWED_EXCLUSION_FIELDS)} (schema 2); "
                f"got {type(entry).__name__}."
            )
        if set(entry) != _REVIEWED_EXCLUSION_FIELDS:
            raise ValueError(
                f"{label} reviewed exclusion {name!r} fields must be exactly "
                f"{sorted(_REVIEWED_EXCLUSION_FIELDS)}, got {sorted(entry)}."
            )
        try:
            records[name] = UKReviewedExclusion(
                **{
                    field_name: entry[field_name]
                    for field_name in sorted(_REVIEWED_EXCLUSION_FIELDS)
                }
            )
        except ValueError as exc:
            raise ValueError(f"{label}: exclusion {name!r}: {exc}") from exc
    return dict(sorted(records.items()))


def coerce_input_mass_reference_registry(
    value: object, *, label: str
) -> dict[str, UKInputMassReferenceDescriptor]:
    """Validate the closed-world input-mass reference registry."""

    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{label} reference_registry must be a non-empty mapping.")
    records: dict[str, UKInputMassReferenceDescriptor] = {}
    for name, entry in value.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(
                f"{label} reference_registry names must be non-empty trimmed "
                f"strings; got {name!r}."
            )
        if isinstance(entry, UKInputMassReferenceDescriptor):
            descriptor = entry
            if descriptor.name != name:
                raise ValueError(
                    f"{label} reference_registry entry {name!r} carries "
                    f"descriptor name {descriptor.name!r}."
                )
            records[name] = descriptor
            continue
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"{label} reference_registry entry {name!r} must be an object."
            )
        expected_entry_fields = {"identity", "totals_sha256", "scope_note"}
        if set(entry) != expected_entry_fields:
            raise ValueError(
                f"{label} reference_registry entry {name!r} fields must be "
                f"exactly {sorted(expected_entry_fields)}, got {sorted(entry)}."
            )
        identity = entry["identity"]
        if not isinstance(identity, Mapping):
            raise TypeError(
                f"{label} reference_registry entry {name!r} identity must be an object."
            )
        expected_identity_fields = {"filename", "revision", "sha256", "vintage"}
        if set(identity) != expected_identity_fields:
            raise ValueError(
                f"{label} reference_registry entry {name!r} identity fields "
                f"must be exactly {sorted(expected_identity_fields)}, got "
                f"{sorted(identity)}."
            )
        records[name] = UKInputMassReferenceDescriptor(
            name=name,
            filename=identity["filename"],
            revision=identity["revision"],
            sha256=identity["sha256"],
            vintage=identity["vintage"],
            totals_sha256=entry["totals_sha256"],
            scope_note=entry["scope_note"],
        )
    return dict(sorted(records.items()))


def exclusion_evaluation_date(now: date | None) -> date:
    """Resolve the injected exclusion clock, refusing datetimes.

    ``datetime`` is a ``date`` subclass, so without this guard a caller
    passing one would compare timestamps against dates (a ``TypeError`` deep
    inside a gate when exclusions exist, or a silently timestamp-shaped
    ``exclusions_evaluated_on`` detail when none do).
    """

    if now is None:
        return datetime.now(UTC).date()
    if isinstance(now, datetime) or not isinstance(now, date):
        raise TypeError(
            f"exclusion evaluation date must be a datetime.date, got "
            f"{type(now).__name__}."
        )
    return now


def _reviewed_exclusion_reasons(
    records: Mapping[str, UKReviewedExclusion],
    *,
    now: date,
) -> tuple[dict[str, str], list[str], list[str]]:
    """Project records to the shared gates' flat reasons, in-force only.

    Returns the in-force ``{column: reason}`` projection plus the sorted
    expired and premature names. Out-of-force entries are withheld so the
    underlying failure fires; the UK wrappers append the receipt context
    beside it.
    """

    projected: dict[str, str] = {}
    expired: list[str] = []
    premature: list[str] = []
    for name, record in records.items():
        if record.expired(now):
            expired.append(name)
        elif record.premature(now):
            premature.append(name)
        else:
            projected[name] = record.reason
    return projected, sorted(expired), sorted(premature)


def _expired_exclusion_failure(
    records: Mapping[str, UKReviewedExclusion],
    expired: list[str],
    *,
    family: str,
) -> str:
    described = "; ".join(
        f"{name} expired {records[name].expires_on} "
        f"(approved_by {records[name].approved_by}, "
        f"{records[name].adjudication})"
        for name in expired
    )
    return (
        f"Reviewed {family} exclusions expired — renew the adjudication or "
        f"remove the entries: {described}."
    )


def _premature_exclusion_failure(
    records: Mapping[str, UKReviewedExclusion],
    premature: list[str],
    *,
    family: str,
) -> str:
    described = "; ".join(
        f"{name} takes force {records[name].approved_on} "
        f"(approved_by {records[name].approved_by}, "
        f"{records[name].adjudication})"
        for name in premature
    )
    return (
        f"Reviewed {family} exclusions are not yet in force — correct the "
        f"receipt's approved_on or wait for it: {described}."
    )


def _read_register_payload(
    source: str | Path | None, *, resource: str
) -> tuple[Mapping[str, object], str]:
    if source is None:
        raw = files("microcosm.build.uk").joinpath(resource).read_text("utf-8")
        label = resource
    else:
        raw = Path(source).read_text(encoding="utf-8")
        label = str(source)

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        duplicates: set[str] = set()
        for name, value in pairs:
            if name in result:
                duplicates.add(name)
            result[name] = value
        if duplicates:
            raise ValueError(f"{label}: duplicate JSON key(s): {sorted(duplicates)}.")
        return result

    def reject_nonfinite_json(value: str) -> object:
        raise ValueError(f"{label}: non-finite JSON value {value!r} is invalid.")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: malformed JSON: {exc.msg}.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}: exclusion register must be a JSON object.")
    return payload, label


def load_uk_reviewed_exclusion_register(
    source: str | Path | None,
    *,
    resource: str,
) -> dict[str, UKReviewedExclusion]:
    """Load one committed reviewed-exclusion register (schema 2, #610).

    ``source`` overrides the committed default (``resource``, a JSON file
    under ``microcosm.build.uk``). The register schema is
    ``{"schema_version": 2, "description": ..., "exclusions": {column:
    {reason, approved_by, adjudication, approved_on, expires_on}}}`` — every
    entry is a complete approval receipt, re-validated by the gates so a
    register cannot bypass the discipline by construction order. Expiry is
    enforced at gate evaluation, never here.
    """

    payload, label = _read_register_payload(source, resource=resource)
    expected_fields = {"schema_version", "description", "exclusions"}
    if set(payload) != expected_fields:
        raise ValueError(
            f"{label}: exclusion register fields must be exactly "
            f"{sorted(expected_fields)}, got {sorted(payload)}."
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
    ):
        raise ValueError(
            f"{label}: exclusion register schema_version must be 2 (the #610 "
            "approval-receipt schema), got "
            f"{payload.get('schema_version')!r}."
        )
    description = payload.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"{label}: exclusion register description must be a non-empty string."
        )
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, Mapping):
        raise ValueError(
            f"{label}: exclusion register must carry an 'exclusions' object."
        )
    return coerce_reviewed_exclusions(exclusions, label=label)


def load_uk_reference_scoped_exclusion_register(
    source: str | Path | None,
    *,
    resource: str,
) -> dict[str, Mapping[str, UKReviewedExclusion]]:
    """Load the schema-3 per-reference input-mass exclusion register."""

    payload, label = _read_register_payload(source, resource=resource)
    expected_fields = {"schema_version", "description", "references"}
    if set(payload) != expected_fields:
        raise ValueError(
            f"{label}: exclusion register fields must be exactly "
            f"{sorted(expected_fields)}, got {sorted(payload)}."
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 3
    ):
        raise ValueError(
            f"{label}: exclusion register schema_version must be 3 "
            "(the per-reference approval-receipt schema), got "
            f"{payload.get('schema_version')!r}."
        )
    description = payload.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"{label}: exclusion register description must be a non-empty string."
        )
    references = payload.get("references")
    if not isinstance(references, Mapping):
        raise ValueError(
            f"{label}: exclusion register must carry a 'references' object."
        )
    result: dict[str, Mapping[str, UKReviewedExclusion]] = {}
    for reference, exclusions in references.items():
        if (
            not isinstance(reference, str)
            or not reference.strip()
            or reference != reference.strip()
        ):
            raise ValueError(
                f"{label}: reference names must be non-empty trimmed strings; "
                f"got {reference!r}."
            )
        result[reference] = MappingProxyType(
            coerce_reviewed_exclusions(
                exclusions, label=f"{label} reference {reference!r}"
            )
        )
    return dict(sorted(result.items()))


@functools.cache
def uk_default_input_mass_reviewed_exclusions() -> Mapping[
    str, Mapping[str, UKReviewedExclusion]
]:
    """Committed schema-3 per-reference input-mass exclusions."""

    return MappingProxyType(
        load_uk_reference_scoped_exclusion_register(
            None, resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE
        )
    )


@functools.cache
def uk_default_qrf_tail_reviewed_exclusions() -> Mapping[str, UKReviewedExclusion]:
    """Committed schema-2 QRF tail exclusions."""

    return MappingProxyType(
        load_uk_reviewed_exclusion_register(
            None, resource=UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE
        )
    )


def load_uk_input_mass_reference(source: str | Path) -> UKInputMassReference:
    """Load frozen reference totals emitted by the #609 measurement tooling.

    Schema: ``{"schema_version": 1, "identity": {filename, revision, sha256,
    vintage}, "totals": {column: weighted_total}}`` — totals keys are flat
    frame column names, matching :func:`uk_input_mass_totals`. The identity
    names the pinned artifact the totals were measured from and is recorded
    verbatim in the gate details and the attestation evidence digest.
    """

    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: input-mass reference must be a JSON object.")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"{path}: input-mass reference schema_version must be 1, got "
            f"{payload.get('schema_version')!r}."
        )
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{path}: input-mass reference needs an 'identity' object.")
    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError(f"{path}: input-mass reference needs a 'totals' object.")
    reference = UKInputMassReference(
        totals=dict(totals),
        filename=str(identity.get("filename", "")),
        revision=str(identity.get("revision", "")),
        sha256=str(identity.get("sha256", "")),
        vintage=str(identity.get("vintage", "")),
    )
    observed_digest = _input_mass_reference_evidence_sha256(reference)
    for descriptor in UK_INPUT_MASS_REFERENCE_REGISTRY.values():
        if (
            reference.identity == descriptor.identity
            and observed_digest == descriptor.totals_sha256
        ):
            return reference
    known = sorted(UK_INPUT_MASS_REFERENCE_REGISTRY)
    raise ValueError(
        f"{path}: input-mass reference sidecar did not match any reviewed "
        f"reference {known}; observed identity {reference.identity} with "
        f"canonical evidence sha256 {observed_digest}."
    )


@dataclass(frozen=True)
class UKInputMassReference:
    """Frozen reference totals and pinned identity for input-mass parity.

    The reference choice is load-bearing (#327): comparing a calibrated
    artifact against a raw base flags correct target-aligned drift as
    failure, so callers must name exactly which frozen artifact the totals
    were measured from. The identity is recorded in the gate details and
    bound into the terminal report's evidence digest.
    """

    totals: Mapping[str, float]
    filename: str
    revision: str
    sha256: str
    vintage: str

    def __post_init__(self) -> None:
        if not isinstance(self.totals, Mapping) or not self.totals:
            raise ValueError(
                "UK input-mass reference totals must be a non-empty mapping."
            )
        normalized: dict[str, float] = {}
        for name, total in self.totals.items():
            column = str(name)
            value = float(total)
            if not column or not math.isfinite(value):
                raise ValueError(
                    "UK input-mass reference totals must map non-empty column "
                    f"names to finite totals; got {name!r} -> {total!r}."
                )
            normalized[column] = value
        object.__setattr__(
            self, "totals", MappingProxyType(dict(sorted(normalized.items())))
        )
        for field_name in ("filename", "revision", "vintage"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"UK input-mass reference {field_name} must be non-empty."
                )
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError(
                "UK input-mass reference sha256 must be a lowercase sha256."
            )

    @property
    def identity(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "revision": self.revision,
            "sha256": self.sha256,
            "vintage": self.vintage,
        }


def _input_mass_reference_evidence_sha256(
    reference: UKInputMassReference,
) -> str:
    payload = {
        "reference": {
            "identity": reference.identity,
            "totals": {name: float(total) for name, total in reference.totals.items()},
        }
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_input_mass_reference(reference: UKInputMassReference) -> None:
    _validate_input_mass_reference_for_descriptor(
        reference, _UK_INPUT_MASS_REFERENCE_DESCRIPTOR
    )


def _validate_input_mass_reference_for_descriptor(
    reference: UKInputMassReference,
    descriptor: UKInputMassReferenceDescriptor,
) -> None:
    if not isinstance(descriptor, UKInputMassReferenceDescriptor):
        raise TypeError("descriptor must be UKInputMassReferenceDescriptor.")
    expected_identity = descriptor.identity
    if reference.identity != expected_identity:
        raise ValueError(
            "UK input-mass reference identity must match the reviewed "
            f"{descriptor.name}; expected {expected_identity}, got "
            f"{reference.identity}. Known references: "
            f"{sorted(UK_INPUT_MASS_REFERENCE_REGISTRY)}."
        )
    observed_digest = _input_mass_reference_evidence_sha256(reference)
    if observed_digest != descriptor.totals_sha256:
        raise ValueError(
            "UK input-mass reference totals must match the reviewed "
            f"{descriptor.name}; expected canonical evidence sha256 "
            f"{descriptor.totals_sha256}, got {observed_digest}."
        )


@dataclass(frozen=True)
class UKInputMassParityPolicy:
    """Reviewed thresholds and exclusion register for input-mass parity.

    There are deliberately no defaults: the US tolerances (0.5 relative, a
    $1e9 USD floor over a 337k-record pool) are calibrated to US incidents
    and dollar scales. UK values must come from the #609 measurement pass
    over the certified compact, the pinned eFRS incumbent, and the staging
    candidate, and be recorded with their derivation when committed.
    """

    relative_tolerance: float
    minimum_reference_total: float
    reviewed_exclusions: Mapping[str, UKReviewedExclusion] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tolerance = float(self.relative_tolerance)
        floor = float(self.minimum_reference_total)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "UK input-mass relative_tolerance must be finite and "
                f"non-negative, got {self.relative_tolerance!r}."
            )
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError(
                "UK input-mass minimum_reference_total must be finite and "
                f"non-negative, got {self.minimum_reference_total!r}."
            )
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "minimum_reference_total", floor)
        object.__setattr__(
            self,
            "reviewed_exclusions",
            MappingProxyType(
                coerce_reviewed_exclusions(
                    self.reviewed_exclusions, label="UK input-mass"
                )
            ),
        )

    def policy_payload(self) -> dict[str, object]:
        return {
            "relative_tolerance": self.relative_tolerance,
            "minimum_reference_total": self.minimum_reference_total,
            "reviewed_exclusions": {
                name: record.policy_payload()
                for name, record in sorted(self.reviewed_exclusions.items())
            },
        }


@dataclass(frozen=True)
class UKQRFTailConcentrationPolicy:
    """Reviewed thresholds and exclusion register for QRF tail concentration.

    No committed defaults, for the same reason as
    :class:`UKInputMassParityPolicy`: the US ``top_k=100`` /
    ``max_top_share=0.75`` / ``min_nonzero_records=500`` numbers are
    calibrated to the #462 incident on the CPS spine. The UK boundaries must
    be measured on the staging pool before they are written down.
    """

    top_k: int
    max_top_share: float
    min_nonzero_records: int
    reviewed_exclusions: Mapping[str, UKReviewedExclusion] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError(
                f"UK QRF tail top_k must be an integer, got {self.top_k!r}."
            )
        if self.top_k < 1:
            raise ValueError(
                f"UK QRF tail top_k must be at least 1, got {self.top_k!r}."
            )
        share = float(self.max_top_share)
        if not math.isfinite(share) or not 0.0 < share < 1.0:
            raise ValueError(
                f"UK QRF tail max_top_share must be in (0, 1), got "
                f"{self.max_top_share!r}."
            )
        if isinstance(self.min_nonzero_records, bool) or not isinstance(
            self.min_nonzero_records, int
        ):
            raise ValueError(
                "UK QRF tail min_nonzero_records must be an integer, got "
                f"{self.min_nonzero_records!r}."
            )
        if self.min_nonzero_records <= self.top_k:
            raise ValueError(
                "UK QRF tail min_nonzero_records must exceed top_k so the tail "
                f"is a strict subset of the carriers, got "
                f"min_nonzero_records={self.min_nonzero_records!r} with "
                f"top_k={self.top_k!r}."
            )
        object.__setattr__(self, "max_top_share", share)
        object.__setattr__(
            self,
            "reviewed_exclusions",
            MappingProxyType(
                coerce_reviewed_exclusions(
                    self.reviewed_exclusions, label="UK QRF tail"
                )
            ),
        )

    def policy_payload(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "max_top_share": self.max_top_share,
            "min_nonzero_records": self.min_nonzero_records,
            "reviewed_exclusions": {
                name: record.policy_payload()
                for name, record in sorted(self.reviewed_exclusions.items())
            },
        }


def uk_input_mass_totals(
    frame: Frame,
    *,
    columns: Iterable[str] | None = None,
) -> dict[str, float]:
    """Weighted totals of the national frame's numeric and boolean columns.

    A thin UK wrapper over the shared
    :func:`microcosm.build.input_mass.input_mass_totals`: keys are flat frame
    column names (the frame enforces global uniqueness across entity tables),
    and the exported weight columns are removed after the shared helper runs —
    the weight vector is plumbing, not mass, but it is a real engine-known
    column on the UK frame, so neither the schema-derived structural set nor
    a caller's ``columns`` allowlist can be trusted to exclude it.
    """

    totals = input_mass_totals(frame, columns=columns)
    for column in UK_EXPORTED_WEIGHT_COLUMNS:
        totals.pop(column, None)
    return totals


def uk_input_mass_parity_gate(
    candidate_totals: Mapping[str, float],
    reference: UKInputMassReference,
    *,
    descriptor: UKInputMassReferenceDescriptor,
    policy: UKInputMassParityPolicy,
    candidate_name: str = "uk_release_candidate",
    now: date | None = None,
) -> GateResult:
    """Require persisted UK input mass to survive against a frozen reference.

    Wraps :func:`microcosm.build.gates.input_mass_parity_gate` verbatim — a
    zero candidate total fails at any tolerance (the #278 signature),
    candidate-only columns are reported and never fail, and near-zero
    reference columns are skipped — and adds the universal exclusion
    discipline the shared gate lacks: an exclusion whose column is now
    within tolerance is stale and **fails**; an exclusion outside the audited
    surface (absent from the reference, or below the reference floor) is
    dormant and reported.
    """

    if not isinstance(reference, UKInputMassReference):
        raise TypeError("reference must be UKInputMassReference.")
    if not isinstance(descriptor, UKInputMassReferenceDescriptor):
        raise TypeError("descriptor must be UKInputMassReferenceDescriptor.")
    if not isinstance(policy, UKInputMassParityPolicy):
        raise TypeError("policy must be UKInputMassParityPolicy.")
    _validate_input_mass_reference_for_descriptor(reference, descriptor)
    evaluated_on = exclusion_evaluation_date(now)
    records = dict(policy.reviewed_exclusions)
    exclusions, expired, premature = _reviewed_exclusion_reasons(
        records, now=evaluated_on
    )
    base = input_mass_parity_gate(
        candidate_totals,
        reference.totals,
        candidate_name=candidate_name,
        reference_name=reference.filename,
        relative_tolerance=policy.relative_tolerance,
        minimum_reference_total=policy.minimum_reference_total,
        reviewed_exclusions=exclusions,
    )

    stale: list[str] = []
    dormant: list[str] = []
    for column in sorted(exclusions):
        if column not in reference.totals:
            dormant.append(column)
            continue
        if abs(reference.totals[column]) <= policy.minimum_reference_total:
            dormant.append(column)
            continue
        # Re-check the single excluded column without its exclusion, reusing
        # the shared gate's exact semantics (zero-fail, drift, absence).
        probe = input_mass_parity_gate(
            ({column: candidate_totals[column]} if column in candidate_totals else {}),
            {column: reference.totals[column]},
            candidate_name=candidate_name,
            reference_name=reference.filename,
            relative_tolerance=policy.relative_tolerance,
            minimum_reference_total=policy.minimum_reference_total,
        )
        if probe.passed:
            stale.append(column)

    failures = list(base.failures)
    if stale:
        failures.append(
            "Stale reviewed input-mass exclusions — the column is within "
            f"tolerance now, remove the exclusion: {stale}."
        )
    if expired:
        failures.append(
            _expired_exclusion_failure(records, expired, family="input-mass")
        )
    if premature:
        failures.append(
            _premature_exclusion_failure(records, premature, family="input-mass")
        )
    return GateResult(
        name=UK_INPUT_MASS_PARITY_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={
            **dict(base.details),
            "stale_exclusions": stale,
            "dormant_exclusions": dormant,
            "expired_exclusions": expired,
            "premature_exclusions": premature,
            "exclusions_evaluated_on": evaluated_on.isoformat(),
            "reference": descriptor.name,
            "reference_scope_note": descriptor.scope_note,
            "reference_identity": reference.identity,
        },
    )


def uk_qrf_tail_concentration_columns(
    frame: Frame,
    *,
    output_columns: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Assemble the tail-concentration evidence for the declared QRF surface.

    The surface defaults to every output declared by a
    ``fit_weighted_qrf_stage*`` operation in the HMRC source manifest.
    Unlike the US wrapper, there is **no sparsity filter**: on the UK staging
    pool the SPI channel is a large fraction of all persons, so a
    nonzero-share cutoff would silently skip the gate exactly where the risk
    lives (#609). Every declared, present, numeric output is checked;
    ``min_nonzero_records`` in the policy remains the sole thinness guard.
    """

    if output_columns is None:
        # Lazy import: the source-contract module owns manifest knowledge and
        # pulls in the full HMRC runtime, which this module must not require
        # at import time.
        from microcosm.build.uk_runtime.hmrc_source_contract import (
            uk_hmrc_weighted_qrf_output_columns,
        )

        declared = uk_hmrc_weighted_qrf_output_columns()
    else:
        declared = tuple(str(name) for name in output_columns)
    if not declared:
        raise ValueError("UK QRF tail-concentration surface must be non-empty.")

    person = frame.table("person")
    person_weights = np.asarray(
        frame.resolve_weights("person").values, dtype=np.float64
    )
    values: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    checked: list[str] = []
    absent: list[str] = []
    non_numeric: list[str] = []
    for column in declared:
        if column not in person.columns:
            absent.append(column)
            values[column] = np.array([], dtype=np.float64)
            weights[column] = np.array([], dtype=np.float64)
            continue
        series = person[column]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(
            series
        ):
            non_numeric.append(column)
            values[column] = np.array([], dtype=np.float64)
            weights[column] = np.array([], dtype=np.float64)
            continue
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        values[column] = numeric.to_numpy(dtype=np.float64)
        weights[column] = person_weights
        checked.append(column)
    surface: dict[str, object] = {
        "declared_qrf_outputs": len(declared),
        "checked_columns": sorted(checked),
        "absent_columns": absent,
        "non_numeric_columns": non_numeric,
        "density_filter": "none: every declared output is checked (#609)",
    }
    return values, weights, surface


def uk_qrf_tail_concentration_gate(
    column_values: Mapping[str, Iterable[float]],
    column_weights: Mapping[str, Iterable[float]],
    *,
    policy: UKQRFTailConcentrationPolicy,
    surface: Mapping[str, object] | None = None,
    now: date | None = None,
) -> GateResult:
    """No declared UK QRF output hides its mass in a handful of records.

    Wraps :func:`microcosm.build.gates.tail_concentration_gate` — which
    already enforces the universal exclusion discipline (stale entries fail,
    dormant entries are reported) — under the UK gate name, and records the
    manifest-derived surface metadata alongside the shared details.
    """

    if not isinstance(policy, UKQRFTailConcentrationPolicy):
        raise TypeError("policy must be UKQRFTailConcentrationPolicy.")
    evaluated_on = exclusion_evaluation_date(now)
    records = dict(policy.reviewed_exclusions)
    projected, expired, premature = _reviewed_exclusion_reasons(
        records, now=evaluated_on
    )
    base = tail_concentration_gate(
        column_values,
        column_weights,
        top_k=policy.top_k,
        max_top_share=policy.max_top_share,
        min_nonzero_records=policy.min_nonzero_records,
        reviewed_exclusions=projected,
    )
    details = dict(base.details)
    details["expired_exclusions"] = expired
    details["premature_exclusions"] = premature
    details["exclusions_evaluated_on"] = evaluated_on.isoformat()
    failures = list(base.failures)
    if expired:
        failures.append(_expired_exclusion_failure(records, expired, family="QRF-tail"))
    if premature:
        failures.append(
            _premature_exclusion_failure(records, premature, family="QRF-tail")
        )
    if surface is not None:
        details["surface"] = dict(surface)
        failures.extend(
            f"{column}: declared QRF output is absent from the person table."
            for column in surface.get("absent_columns", ())
        )
        failures.extend(
            f"{column}: declared QRF output is not numeric in the person table."
            for column in surface.get("non_numeric_columns", ())
        )
        declared_count = surface.get("declared_qrf_outputs")
        classified = {
            field: surface.get(field)
            for field in (
                "checked_columns",
                "absent_columns",
                "non_numeric_columns",
            )
        }
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count <= 0
            or any(
                not isinstance(names, list)
                or any(not isinstance(name, str) or not name for name in names)
                for names in classified.values()
            )
        ):
            failures.append(
                "QRF surface must declare a positive output count and three "
                "lists of non-empty column names."
            )
        else:
            checked = set(classified["checked_columns"])
            absent = set(classified["absent_columns"])
            non_numeric = set(classified["non_numeric_columns"])
            all_lists = [
                *classified["checked_columns"],
                *classified["absent_columns"],
                *classified["non_numeric_columns"],
            ]
            accounted = checked | absent | non_numeric
            gate_accounted = set(details["top_share"]) | set(details["thin_columns"])
            if (
                len(all_lists) != len(accounted)
                or declared_count != len(accounted)
                or set(details["top_share"]) & set(details["thin_columns"])
                or accounted != gate_accounted
                or checked != gate_accounted - absent - non_numeric
            ):
                failures.append(
                    "QRF surface declarations must reconcile exactly across "
                    "declared, checked, absent, nonnumeric, checked-tail, and "
                    "thin outputs."
                )
    if details["columns_checked"] == 0:
        failures.append(
            "No declared QRF output had enough weighted carriers for the "
            "tail-concentration check."
        )
    return GateResult(
        name=UK_QRF_TAIL_CONCENTRATION_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
