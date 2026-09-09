"""The committed signed-differences register and its loader.

The register is what makes "anything differing that is not signed is a defect"
enforceable, so the loader has to be strict about the vocabulary and the
scoping, and the committed file has to stay internally coherent.
"""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from importlib.resources import files
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.signed_differences import (
    SIGNED_DIFFERENCE_CLASSES,
    SIGNED_DIFFERENCE_EXPECTATIONS,
    SIGNED_DIFFERENCE_SURFACES,
    UKSignedDifference,
    UKSignedDifferenceRegister,
    load_uk_spine_swap_signed_differences,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _valid_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": "example-difference",
        "class": "mechanism_change",
        "scope": {
            "surface": "nonzero_shares",
            "columns": ["some_column"],
            "entities": ["household"],
        },
        "expectation": "column_differs",
        "magnitude_evidence": "A disclosure-safe statement of the magnitude.",
        "evidence": "experiments/686-uk-spine-swap-receipts.md#r0",
        "adjudicator": "juaristi22",
        "adjudicated_on": "2026-08-22",
    }
    entry.update(overrides)
    return entry


def _write(tmp_path: Path, payload: object) -> str:
    path = tmp_path / "register.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _github_anchors(path: Path) -> set[str]:
    """The fragment ids GitHub derives from a Markdown file's headings.

    Lower-case, punctuation dropped, whitespace to hyphens — so an em-dash
    surrounded by spaces yields a doubled hyphen, which is the detail that
    makes hand-written anchors get this wrong.
    """

    anchors: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        matched = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if matched is None:
            continue
        text = re.sub(r"[^\w\s-]", "", matched.group(1).lower())
        anchors.add(re.sub(r"\s", "-", text))
    return anchors


_DECIMAL = r"(?P<value>\d+\.\d{4,})"


def _incumbent_share_quotes(text: str, column: str) -> list[str]:
    """Incumbent share figures quoted beside a scoped column name."""

    column_pattern = re.escape(column)
    patterns = [
        rf"{column_pattern}[^;]*?incumbent(?:'s)?(?:\s+\w+)?\s+{_DECIMAL}",
        rf"{column_pattern}[^;]*?donor\s+\d+\.\d{{4,}}\s+against\s+{_DECIMAL}\s+and",
        rf"{column_pattern}[^;]*?incumbent\s+carries[^;]*?share\s+of\s+{_DECIMAL}",
        rf"{column_pattern}[\s\S]{{0,220}}?Unweighted\s+share\s+{_DECIMAL}\s+for\s+the\s+incumbent",
    ]
    quotes: list[str] = []
    for pattern in patterns:
        for matched in re.finditer(pattern, text):
            quotes.append(matched.group("value"))
    return quotes


class TestCommittedRegister:
    def test_committed_register_loads(self) -> None:
        register = load_uk_spine_swap_signed_differences()
        assert register.schema_version == 1
        assert register.differences
        assert register.scope_note

    def test_uc_capital_entries_pin_new_exports_and_retired_claim_lift(self) -> None:
        register = load_uk_spine_swap_signed_differences()

        for column, identifier in {
            "frs_benunit_capital": "frs-benunit-capital-net-new-column",
            "uc_reported_capital": "uc-reported-capital-net-new-column",
        }.items():
            entry = register.matching(
                surface="nonzero_shares",
                column=column,
                expectation="column_missing_in_reference",
                entity="benunit",
            )
            assert entry is not None
            assert entry.id == identifier

        # Retired 2026-09-02 on the spine-m actuals (Part F of the #832
        # receipts): the reporter redraw's demotions outweigh #828's OR-refresh
        # lift, leaving would_claim_uc 0.001951 below the incumbent — inside the
        # whole-spine acceptance band, where the register's doctrine signs
        # nothing. An unsigned would_claim_uc divergence is a defect again.
        assert (
            register.matching(
                surface="nonzero_shares",
                column="would_claim_uc",
                expectation="column_differs",
                entity="benunit",
            )
            is None
        )

        reporter = register.matching(
            surface="nonzero_shares",
            column="universal_credit_reported",
            expectation="column_differs",
            entity="person",
        )
        assert reporter is not None
        assert reporter.id == "uc-reporter-benefit-unit-redraw-incidence"
        assert reporter.quantitative == {
            "shares": {
                "universal_credit_reported": {
                    "incumbent_share": 0.057359,
                    "direction": "candidate_below",
                    "max_abs_delta": 0.0184,
                }
            },
            "magnitude_provenance": (
                "I1 dry-run bound from "
                "experiments/832-uc-reporter-receipts.md Part C: 4,428 "
                "unchanged base-channel reporter persons plus between zero "
                "and 3,299 one-person SPI reporter landings among 113,649 "
                "people implies share in [0.038962, 0.067990]. I5 actual "
                "(Part F, spine-m): 5,730 nonzero reporter records, share "
                "0.050418, delta 0.006941 below the incumbent, inside the "
                "signed 0.0184 floor-distance; the above side of the structural "
                "range is unreached by the built artifact."
            ),
        }

    def test_dsa_entry_pins_calibration_year_seed(self) -> None:
        register = load_uk_spine_swap_signed_differences()

        entry = register.matching(
            surface="nonzero_shares",
            column="disabled_students_allowance_eligible_expenses",
            expectation="column_missing_in_reference",
            entity="person",
        )
        assert entry is not None
        assert entry.id == "dsa-eligible-expenses-seeded-at-calibration-year"
        assert entry.difference_class == "defect_fix"
        assert entry.quantitative == {
            "structural": {
                "expected_columns": ["disabled_students_allowance_eligible_expenses"]
            }
        }

    def test_committed_entries_are_precisely_scoped(self) -> None:
        # A surface-wide entry (empty columns) signs every column on that
        # surface. That is a real capability for entity_counts, but on a
        # column surface it would sign away defects wholesale.
        register = load_uk_spine_swap_signed_differences()
        for difference in register.differences:
            if difference.surface in {"nonzero_shares", "weighted_totals"}:
                assert difference.columns, (
                    f"{difference.id} signs a column surface without naming "
                    "columns; it would absorb unrelated divergences."
                )

    def test_committed_evidence_anchors_point_at_a_real_file(self) -> None:
        register = load_uk_spine_swap_signed_differences()
        for difference in register.differences:
            relative = difference.evidence.split("#", 1)[0]
            assert (REPO_ROOT / relative).is_file(), (
                f"{difference.id} cites missing evidence file {relative}"
            )

    def test_committed_evidence_anchors_resolve_to_a_real_heading(self) -> None:
        # The evidence pointer is what makes an adjudication auditable. A
        # fragment that no longer resolves fails silently in a browser, so the
        # rot only shows up when a reviewer clicks and lands nowhere.
        register = load_uk_spine_swap_signed_differences()
        headings: dict[str, set[str]] = {}
        for difference in register.differences:
            relative, _, fragment = difference.evidence.partition("#")
            if not fragment:
                continue
            if relative not in headings:
                headings[relative] = _github_anchors(REPO_ROOT / relative)
            assert fragment in headings[relative], (
                f"{difference.id} cites {relative}#{fragment}, which is not a "
                f"heading in that file. Available: "
                f"{sorted(headings[relative])}"
            )

    def test_every_signed_column_exists_on_the_surface_it_signs(self) -> None:
        # A typo in a column name is the quiet failure mode here: the entry
        # matches nothing, the real divergence stays unsigned, and the only
        # symptom is a defect verdict nobody can trace back to the typo.
        reference = json.loads(
            files("microcosm.build.uk")
            .joinpath("efrs_parity_reference.json")
            .read_text(encoding="utf-8")
        )
        known = set(reference["nonzero_shares"])
        entities = set(reference["entity_stats"])
        # Columns the spine adds are signed precisely because the reference
        # does not carry them, so they are checked against the expectation
        # rather than against the reference's column set.
        net_new = {
            column
            for difference in load_uk_spine_swap_signed_differences().differences
            if difference.expectation == "column_missing_in_reference"
            for column in difference.columns
        }
        for difference in load_uk_spine_swap_signed_differences().differences:
            for column in difference.columns:
                if difference.surface == "entity_counts":
                    assert column in entities, (
                        f"{difference.id} signs entity {column!r}, which the "
                        "reference does not carry."
                    )
                elif difference.surface == "nonzero_shares" and column not in net_new:
                    assert column in known, (
                        f"{difference.id} signs column {column!r}, which is not "
                        "on the reference share surface — likely a typo, which "
                        "would leave the real divergence unsigned."
                    )

    def test_the_register_signs_no_column_twice(self) -> None:
        # Two entries covering one column on one surface make the adjudication
        # ambiguous: the reader cannot tell which rationale is the live one.
        seen: dict[tuple[str, str], str] = {}
        for difference in load_uk_spine_swap_signed_differences().differences:
            for column in difference.columns:
                key = (difference.surface, column)
                assert key not in seen, (
                    f"{column!r} on {difference.surface} is signed by both "
                    f"{seen[key]} and {difference.id}."
                )
                seen[key] = difference.id

    def test_no_committed_entry_expires(self) -> None:
        payload = json.loads(
            (
                REPO_ROOT
                / "packages/microcosm-build/src/microcosm/build/uk"
                / "spine_swap_signed_differences.json"
            ).read_text(encoding="utf-8")
        )
        for entry in payload["differences"]:
            assert "expires_on" not in entry

    def test_committed_quantitative_blocks_match_the_reference(self) -> None:
        reference = json.loads(
            files("microcosm.build.uk")
            .joinpath("efrs_parity_reference.json")
            .read_text(encoding="utf-8")
        )
        for difference in load_uk_spine_swap_signed_differences().differences:
            assert difference.quantitative is not None, (
                f"{difference.id} has no quantitative block; the verifier "
                "would have no bound to check against."
            )
            if (
                difference.surface == "nonzero_shares"
                and difference.expectation == "column_differs"
            ):
                shares = difference.quantitative["shares"]
                assert set(shares) == set(difference.columns)
                for column in difference.columns:
                    measured = shares[column]
                    incumbent = reference["nonzero_shares"][column]
                    assert measured["incumbent_share"] == incumbent
                    assert measured["direction"] in {
                        "candidate_above",
                        "candidate_below",
                    }
                    assert measured["max_abs_delta"] > 0.0
            elif difference.surface == "entity_counts":
                assert difference.quantitative["expected_deltas"] == {
                    "benunit": -12,
                    "person": 32,
                }

    def test_committed_incumbent_share_quotes_match_the_reference(self) -> None:
        reference = json.loads(
            files("microcosm.build.uk")
            .joinpath("efrs_parity_reference.json")
            .read_text(encoding="utf-8")
        )
        checked: list[tuple[str, str, str]] = []
        for difference in load_uk_spine_swap_signed_differences().differences:
            if (
                difference.surface != "nonzero_shares"
                or difference.expectation != "column_differs"
            ):
                continue
            for column in difference.columns:
                for quote in _incumbent_share_quotes(
                    difference.magnitude_evidence, column
                ):
                    checked.append((difference.id, column, quote))
                    places = len(quote.rsplit(".", 1)[1])
                    expected = Decimal(str(reference["nonzero_shares"][column]))
                    quantized = expected.quantize(
                        Decimal("1").scaleb(-places), rounding=ROUND_HALF_UP
                    )
                    assert Decimal(quote) == quantized, (
                        f"{difference.id} quotes {column} incumbent share as "
                        f"{quote}, but the packaged reference rounds to "
                        f"{quantized} at {places} decimals."
                    )
        assert len(checked) >= 10, (
            "The prose quote check matched too little evidence; it is likely "
            "not enforcing the committed register."
        )


class TestLookup:
    def test_matching_finds_the_signing_entry(self) -> None:
        register = load_uk_spine_swap_signed_differences()
        found = register.matching(
            surface="nonzero_shares",
            column="water_and_sewerage_charges",
            expectation="column_differs",
        )
        assert found is not None
        assert found.id == "scottish-water-incumbent-nan-zeroing"

    def test_a_column_signed_on_one_surface_is_not_signed_on_another(self) -> None:
        # The Scottish level change moves weighted totals but deliberately not
        # the nonzero share, and the share entry is a different adjudication.
        register = load_uk_spine_swap_signed_differences()
        weighted = register.matching(
            surface="weighted_totals",
            column="water_and_sewerage_charges",
            expectation="column_differs",
        )
        assert weighted is not None
        assert weighted.id == "scottish-water-sewerage-successor-level"
        assert (
            register.matching(
                surface="entity_counts",
                column="household",
                expectation="column_differs",
            )
            is None
        )

    def test_unsigned_column_returns_none(self) -> None:
        register = load_uk_spine_swap_signed_differences()
        assert (
            register.matching(
                surface="nonzero_shares",
                column="employment_income",
                expectation="column_differs",
            )
            is None
        )

    def test_surface_wide_entry_covers_any_column(self) -> None:
        entry = UKSignedDifference(
            id="counts",
            difference_class="rng_stream",
            surface="entity_counts",
            expectation="count_differs",
            columns=(),
            entities=("person",),
            magnitude_evidence="evidence",
            evidence="experiments/686-uk-spine-swap-receipts.md",
            adjudicator="juaristi22",
            adjudicated_on="2026-08-22",
        )
        assert entry.covers(
            surface="entity_counts", column="person", expectation="count_differs"
        )
        assert entry.covers(
            surface="entity_counts", column="benunit", expectation="count_differs"
        )
        assert not entry.covers(
            surface="nonzero_shares", column="person", expectation="count_differs"
        )


class TestExpectationAwareCoverage:
    """The expectation is consulted at lookup, not merely validated at load.

    Without this, an entry adjudicated for a column *appearing* would also
    sign an arbitrarily large *value* divergence in that same column — the
    register's own scope note names that failure mode.
    """

    def _entry(self, *, surface: str, expectation: str) -> UKSignedDifference:
        return UKSignedDifference(
            id="entry",
            difference_class="net_new_column",
            surface=surface,
            expectation=expectation,
            columns=("num_bedrooms",),
            entities=("household",),
            magnitude_evidence="evidence",
            evidence="experiments/686-uk-spine-swap-receipts.md",
            adjudicator="juaristi22",
            adjudicated_on="2026-08-22",
        )

    def test_a_net_new_entry_does_not_sign_a_value_divergence(self) -> None:
        entry = self._entry(
            surface="nonzero_shares", expectation="column_missing_in_reference"
        )
        assert entry.covers(
            surface="nonzero_shares",
            column="num_bedrooms",
            expectation="column_missing_in_reference",
        )
        assert not entry.covers(
            surface="nonzero_shares",
            column="num_bedrooms",
            expectation="column_differs",
        )

    def test_a_value_entry_does_not_sign_a_structural_difference(self) -> None:
        entry = self._entry(surface="nonzero_shares", expectation="column_differs")
        assert not entry.covers(
            surface="nonzero_shares",
            column="num_bedrooms",
            expectation="column_missing_in_reference",
        )

    def test_a_share_entry_bridges_to_the_payload_surface(self) -> None:
        # The share instrument and the payload comparator read the same
        # adjudicated fact through different measurements, so one signature
        # covers both — this is what lets --structure-only reach a verdict.
        entry = self._entry(surface="nonzero_shares", expectation="column_differs")
        assert entry.covers(
            surface="payload_column",
            column="num_bedrooms",
            expectation="column_differs",
        )

    def test_the_bridge_does_not_reach_structural_surfaces(self) -> None:
        entry = self._entry(
            surface="nonzero_shares", expectation="column_missing_in_reference"
        )
        assert not entry.covers(
            surface="payload_column",
            column="num_bedrooms",
            expectation="column_missing_in_reference",
        )
        counts = self._entry(surface="entity_counts", expectation="count_differs")
        assert not counts.covers(
            surface="payload_column",
            column="num_bedrooms",
            expectation="count_differs",
        )

    def test_the_committed_register_covers_the_payload_surface(self) -> None:
        # Every committed value adjudication must be reachable from the
        # payload comparator, or --structure-only can never pass.
        register = load_uk_spine_swap_signed_differences()
        for difference in register.differences:
            if (
                difference.surface != "nonzero_shares"
                or difference.expectation != "column_differs"
            ):
                continue
            for column in difference.columns:
                assert (
                    register.matching(
                        surface="payload_column",
                        column=column,
                        expectation="column_differs",
                    )
                    is not None
                )


class TestEntityScope:
    """A column name is not unique across tables (#747 re-review).

    Every committed entry is entity-scoped, but `covers()` matched on column
    alone — harmless while only the share surface consulted it, and
    load-bearing the moment the payload bridge let a household adjudication
    reach a per-table comparison.
    """

    def _entry(self, *, entities: list[str]) -> UKSignedDifference:
        return UKSignedDifference(
            id="household-scoped",
            difference_class="mechanism_change",
            surface="nonzero_shares",
            expectation="column_differs",
            columns=("water_and_sewerage_charges",),
            entities=tuple(entities),
            magnitude_evidence="evidence",
            evidence="experiments/686-uk-spine-swap-receipts.md",
            adjudicator="juaristi22",
            adjudicated_on="2026-08-22",
        )

    def test_an_entry_signs_only_the_entity_it_names(self) -> None:
        entry = self._entry(entities=["household"])
        assert entry.covers(
            surface="nonzero_shares",
            column="water_and_sewerage_charges",
            expectation="column_differs",
            entity="household",
        )
        assert not entry.covers(
            surface="nonzero_shares",
            column="water_and_sewerage_charges",
            expectation="column_differs",
            entity="person",
        )

    def test_the_payload_bridge_carries_the_entity_scope(self) -> None:
        # The bridge must not widen the adjudication it forwards.
        entry = self._entry(entities=["household"])
        assert entry.covers(
            surface="payload_column",
            column="water_and_sewerage_charges",
            expectation="column_differs",
            entity="household",
        )
        assert not entry.covers(
            surface="payload_column",
            column="water_and_sewerage_charges",
            expectation="column_differs",
            entity="person",
        )

    def test_a_caller_without_an_entity_gets_no_entity_filtering(self) -> None:
        entry = self._entry(entities=["household"])
        assert entry.covers(
            surface="nonzero_shares",
            column="water_and_sewerage_charges",
            expectation="column_differs",
        )

    def test_every_committed_entry_names_the_reference_entity(self) -> None:
        # An entry naming the wrong entity now fails to match rather than
        # signing across tables, so a mis-declared scope is a defect.
        reference = json.loads(
            files("microcosm.build.uk")
            .joinpath("efrs_parity_reference.json")
            .read_text(encoding="utf-8")
        )
        entities = reference["input_entities"]
        for difference in load_uk_spine_swap_signed_differences().differences:
            if difference.surface not in {"nonzero_shares", "weighted_totals"}:
                continue
            for column in difference.columns:
                actual = entities.get(column)
                if actual is None:
                    continue
                assert actual in difference.entities, (
                    f"{difference.id} scopes {column!r} to "
                    f"{list(difference.entities)}, but the reference carries "
                    f"it on {actual!r}."
                )


class TestValidation:
    def test_duplicate_ids_are_refused(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [_valid_entry(), _valid_entry()],
        }
        with pytest.raises(ValueError, match="unique"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_expiry_is_refused(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [_valid_entry(expires_on="2027-01-01")],
        }
        with pytest.raises(ValueError, match="do not expire"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("class", "not_a_class", "must be one of"),
            ("expectation", "not_an_expectation", "must be one of"),
            ("id", "Not Kebab Case", "kebab-case"),
            ("adjudicated_on", "22-08-2026", "ISO date"),
            ("magnitude_evidence", "", "non-empty string"),
            ("adjudicator", "", "non-empty string"),
            ("evidence", "", "non-empty string"),
        ],
    )
    def test_field_validation(
        self, tmp_path: Path, field: str, value: str, match: str
    ) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [_valid_entry(**{field: value})],
        }
        with pytest.raises(ValueError, match=match):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_a_column_surface_entry_without_columns_is_refused(
        self, tmp_path: Path
    ) -> None:
        # Via the payload bridge an empty columns tuple would blanket-sign
        # every column in every table.
        entry = _valid_entry()
        entry["scope"] = {
            "surface": "nonzero_shares",
            "columns": [],
            "entities": ["household"],
        }
        payload = {"schema_version": 1, "scope_note": "note", "differences": [entry]}
        with pytest.raises(ValueError, match="without naming columns"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_a_column_surface_entry_without_entities_is_refused(
        self, tmp_path: Path
    ) -> None:
        entry = _valid_entry()
        entry["scope"] = {
            "surface": "nonzero_shares",
            "columns": ["savings"],
            "entities": [],
        }
        payload = {"schema_version": 1, "scope_note": "note", "differences": [entry]}
        with pytest.raises(ValueError, match="without naming entities"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_an_entity_counts_entry_may_stay_surface_wide(self, tmp_path: Path) -> None:
        # The one surface where a column-less entry is meaningful: its
        # "column" is the entity name.
        entry = _valid_entry()
        entry["scope"] = {"surface": "entity_counts", "columns": [], "entities": []}
        entry["expectation"] = "count_differs"
        payload = {"schema_version": 1, "scope_note": "note", "differences": [entry]}
        register = load_uk_spine_swap_signed_differences(_write(tmp_path, payload))
        assert register.differences[0].columns == ()

    def test_unknown_surface_is_refused(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [
                _valid_entry(scope={"surface": "made_up", "columns": ["c"]})
            ],
        }
        with pytest.raises(ValueError, match="must be one of"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_unsupported_schema_version_is_refused(self, tmp_path: Path) -> None:
        payload = {"schema_version": 2, "scope_note": "note", "differences": []}
        with pytest.raises(ValueError, match="schema_version"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_missing_scope_is_refused(self, tmp_path: Path) -> None:
        entry = _valid_entry()
        del entry["scope"]
        payload = {"schema_version": 1, "scope_note": "note", "differences": [entry]}
        with pytest.raises(ValueError, match="scope"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_columns_must_be_a_list_of_strings(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [
                _valid_entry(scope={"surface": "nonzero_shares", "columns": "col"})
            ],
        }
        with pytest.raises(ValueError, match="list of strings"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_vocabularies_are_disjoint_and_populated(self) -> None:
        assert SIGNED_DIFFERENCE_CLASSES
        assert SIGNED_DIFFERENCE_SURFACES
        assert SIGNED_DIFFERENCE_EXPECTATIONS
        assert not SIGNED_DIFFERENCE_CLASSES & SIGNED_DIFFERENCE_SURFACES

    def test_register_rejects_duplicate_ids_at_construction(self) -> None:
        entry = UKSignedDifference(
            id="same",
            difference_class="vintage",
            surface="nonzero_shares",
            expectation="column_differs",
            columns=("a",),
            entities=("household",),
            magnitude_evidence="evidence",
            evidence="experiments/686-uk-spine-swap-receipts.md",
            adjudicator="juaristi22",
            adjudicated_on="2026-08-22",
        )
        with pytest.raises(ValueError, match="unique"):
            UKSignedDifferenceRegister(differences=(entry, entry), scope_note="note")
