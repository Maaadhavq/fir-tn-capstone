"""Deterministic IF-1 form rendering.

CONTEXT.md, scope: "We do NOT design or generate the document -- Stage D just
fills the fixed template with extracted details." This module is that template.
It walks a `FirRecord` and writes the CCTNS IF-1's fifteen numbered items in
their fixed order. No model is involved; the same record always renders to the
same text, which is what makes the output auditable and the golden test possible.

Items 1-12 are filled from the record. Items 13-15 (action taken, complainant's
signature, dispatch to court) belong to the officer and are rendered as blanks --
the system has no business writing into them.

Two render modes:

  plain      what the officer sees on the form
  annotated  the same form with each machine-filled slot tagged with its status,
             confidence and provenance count, for the verification screen

Empty slots render as a visible blank (`________`) rather than being omitted,
because on a fixed form an unfilled field is information: it tells the officer
what still has to be elicited from the complainant.
"""

from __future__ import annotations

from fir.schema.if1 import ExtractedField, FirRecord, Person, SectionCandidate

BLANK = "________"
_ACT_LABEL = {
    "BNS_2023": "Bharatiya Nyaya Sanhita, 2023",
    "BNSS_2023": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "MV_Act": "Motor Vehicles Act, 1988",
    "IT_Act_2000": "Information Technology Act, 2000",
    "TN_PHW_Act": "Tamil Nadu Prohibition of Harassment of Women Act",
    "OTHER": "Other",
}


# ---------------------------------------------------------------------------
# field formatting
# ---------------------------------------------------------------------------


def _val(f: ExtractedField, annotate: bool = False, lang: str = "en") -> str:
    """Render one wrapped field.

    Order: surface text in the requested language -> the normalised `value`
    -> surface text in the other language -> blank. The value outranks the
    other language because it is the canonical form; a Tamil verbatim string
    on an English rendering (or vice versa) is a last resort, not a preference.
    """
    if not f.is_filled:
        return BLANK
    preferred = getattr(f.text, lang, None) if f.text else None
    other = (f.text.en or f.text.ta) if f.text else None
    shown = preferred or (str(f.value) if f.value is not None else None) or other or BLANK
    if annotate and f.status != "empty":
        tag = f.status
        if f.confidence is not None:
            tag += f" {f.confidence:.2f}"
        if f.provenance:
            tag += f" src:{len(f.provenance)}"
        shown = f"{shown}  ⟨{tag}⟩"
    return shown


def _person_lines(p: Person, annotate: bool, lang: str, indent: str = "     ") -> list[str]:
    rel = f"{p.relative_type.title()}'s" if p.relative_type else "Father's / Husband's"
    passport_parts = [
        _val(p.passport.number, annotate, lang),
        _val(p.passport.date_place_of_issue, annotate, lang),
    ]
    return [
        f"{indent}(a) Name                      : {_val(p.name, annotate, lang)}",
        f"{indent}(b) {rel + ' Name':<26s}: {_val(p.relative_name, annotate, lang)}",
        f"{indent}(c) Date / Year of Birth      : {_val(p.dob_or_age, annotate, lang)}",
        f"{indent}(d) Nationality               : {_val(p.nationality, annotate, lang)}",
        f"{indent}(e) Passport No. / Issue      : {' / '.join(passport_parts)}",
        f"{indent}(f) Occupation                : {_val(p.occupation, annotate, lang)}",
        f"{indent}(g) Address                   : {_val(p.address, annotate, lang)}",
        f"{indent}    Contact                   : {_val(p.contact, annotate, lang)}",
    ]


def _section_lines(i: int, c: SectionCandidate, annotate: bool) -> list[str]:
    flags = []
    if c.mapping_ambiguous:
        flags.append("MAPPING UNVERIFIED -- officer to confirm")
    if c.cognizable is True:
        flags.append("cognizable")
    elif c.cognizable is False:
        flags.append("non-cognizable")
    lineage = f"  [from IPC {c.ipc_source}]" if c.ipc_source else ""
    tail = f"   ({'; '.join(flags)})" if flags else ""
    desc = f" -- {c.description_en}" if c.description_en else ""
    lines = [f"     ({_roman(i)}) {_ACT_LABEL.get(c.act, c.act)}, s.{c.section}{desc}{lineage}{tail}"]
    # element-wise justification: the ingredients of the offence and whether the
    # narrative evidences each. Shown on the plain form too -- this is the part
    # of the draft an officer most needs to see, not an annotation.
    if c.elements:
        for e in c.elements:
            mark = {"yes": "[x]", "no": "[ ]", "unclear": "[?]"}[e.satisfied]
            lines.append(f"           {mark} {e.element}")
            if annotate or e.satisfied != "yes":
                lines.append(f"               {e.justification}")
    elif not c.mapping_ambiguous:
        lines.append("           (elements not analysed for this section)")
    return lines


def _roman(n: int) -> str:
    return ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"][n - 1] if n <= 10 else str(n)


def _yn(b: bool) -> str:
    return "Yes" if b else "No"


# ---------------------------------------------------------------------------
# the form
# ---------------------------------------------------------------------------


def render_if1(rec: FirRecord, annotate: bool = False, lang: str = "en") -> str:
    """Render the record as the fifteen-item IF-1.

    `annotate=True` tags every machine-filled slot for the verification screen.
    `lang` picks which surface text to prefer where a field carries both.
    """
    a = annotate
    L: list[str] = []
    w = L.append

    title = "FIRST INFORMATION REPORT" if rec.status != "csr_advisory" else (
        "COMMUNITY SERVICE REGISTER -- ADVISORY (non-cognizable)"
    )
    w("=" * 78)
    w(f"{title:^78s}")
    w(f"{'(Under Section 173 BNSS, 2023)  --  Form IF-1':^78s}")
    w("=" * 78)
    w(f"Record {rec.record_id}   status: {rec.status.upper()}   "
      f"v{rec.document_version}   schema {rec.schema_version}")
    if annotate:
        n = len(rec.machine_filled_fields())
        w(f"ANNOTATED -- {n} machine-filled slot(s) marked ⟨status conf src:n⟩. "
          "Officer verification required for every one.")
    w("")

    # 1
    w("1.  District        : " + _val(rec.jurisdiction.district, a, lang))
    w("    P.S.            : " + _val(rec.jurisdiction.police_station, a, lang)
      + (f"   (code {rec.jurisdiction.ps_code})" if rec.jurisdiction.ps_code else ""))
    w(f"    Year : {rec.fir_year or BLANK}    FIR No. : {rec.fir_number or BLANK}    "
      f"Date : {rec.fir_date or BLANK}")
    w("")

    # 2
    w("2.  Acts & Sections :")
    if rec.acts_sections:
        for i, c in enumerate(rec.acts_sections, 1):
            L.extend(_section_lines(i, c, a))
    else:
        w(f"     {BLANK}   (no section could be applied -- officer to determine)")
    w("")

    # 3
    occ = rec.occurrence
    w("3.  (a) Occurrence of Offence")
    w(f"        From : {_val(occ.from_datetime, a, lang)}")
    if occ.is_interval or occ.to_datetime.is_filled:
        w(f"        To   : {_val(occ.to_datetime, a, lang)}")
    w(f"    (b) Information received at P.S. : {_val(occ.info_received_at_ps, a, lang)}")
    w(f"    (c) General Diary Reference       : {_val(occ.gd_reference, a, lang)}")
    w("")

    # 4
    it = rec.information_type
    w(f"4.  Type of Information : {'Written' if it == 'written' else 'Oral' if it == 'oral' else BLANK}")
    w("")

    # 5
    po = rec.place_of_occurrence
    w("5.  Place of Occurrence")
    w(f"    (a) Direction & distance from P.S. : {_val(po.direction_from_ps, a, lang)} / "
      f"{_val(po.distance_from_ps_km, a, lang)} km    Beat No. : {_val(po.beat_no, a, lang)}")
    w(f"    (b) Address                        : {_val(po.address, a, lang)}")
    if po.outside_ps_limits:
        w(f"    (c) Outside this P.S. limits -- P.S. : {_val(po.other_ps_name, a, lang)}   "
          f"District : {_val(po.other_district, a, lang)}")
    else:
        w("    (c) Outside this P.S. limits       : No")
    w("")

    # 6
    w("6.  Complainant / Informant")
    L.extend(_person_lines(rec.complainant, a, lang, indent="    "))
    w("")

    # 7
    w("7.  Details of known / suspected / unknown accused")
    if not rec.accused:
        w(f"     {BLANK}")
    for i, acc in enumerate(rec.accused, 1):
        kind = "known" if acc.known else "unknown" if acc.unknown else "suspected"
        w(f"     {i}. [{kind}] Name : {_val(acc.name, a, lang)}"
          + (f"   alias {_val(acc.alias, a, lang)}" if acc.alias.is_filled else ""))
        if acc.relative_name.is_filled:
            w(f"        Father's/Husband's name : {_val(acc.relative_name, a, lang)}")
        if acc.physical_description.is_filled:
            w(f"        Description : {_val(acc.physical_description, a, lang)}")
        if acc.address.is_filled:
            w(f"        Address     : {_val(acc.address, a, lang)}")
    w("")

    # 8
    w(f"8.  Reasons for delay in reporting : {_val(rec.delay_reason, a, lang)}")
    w("")

    # 9 / 10
    w("9.  Particulars of properties stolen / involved")
    if not rec.properties_involved:
        w(f"     {BLANK}")
    for i, p in enumerate(rec.properties_involved, 1):
        w(f"     {i}. {_val(p.category, a, lang)} -- {_val(p.description, a, lang)}   "
          f"₹ {_val(p.estimated_value_inr, a, lang)}")
    w(f"10. Total value of property : ₹ {_val(rec.total_property_value_inr, a, lang)}")
    w("")

    # 11
    w(f"11. Inquest Report / U.D. Case No. : {_val(rec.inquest_ud_case_no, a, lang)}")
    w("")

    # 12
    w("12. F.I.R. Contents")
    body = getattr(rec.narrative, lang, None) or rec.narrative.en or rec.narrative.ta
    if body:
        for line in body.strip().splitlines():
            w(f"     {line}")
        if annotate:
            g = rec.narrative.grounding
            ok = sum(1 for e in g if e.supported)
            gate = "PASS" if rec.narrative_is_grounded else "FAIL"
            w(f"     ⟨faithfulness gate: {gate} -- {ok}/{len(g)} sentences grounded⟩")
    else:
        w(f"     {BLANK}")
    w("")

    # cognizability (informs item 13 but is not the officer's decision)
    cog = rec.cognizability
    w(f"    Cognizability (system assessment) : {cog.decision.replace('_', '-')}  ->  {cog.route}")
    if cog.rationale:
        w(f"    Rationale : {cog.rationale}")
    if cog.schedule_ref:
        w(f"    Basis     : {cog.schedule_ref}")
    w("")

    # 13-15: officer-only
    w("13. Action taken : " + BLANK)
    w("    (Registered the case and took up investigation / directed / refused "
      "investigation / transferred to P.S. ________ on point of jurisdiction.)")
    w("    Officer name, rank & number : " + BLANK)
    w("")
    w("14. Signature / thumb impression of the complainant / informant : " + BLANK)
    w("")
    w("15. Date & time of dispatch to the court : " + BLANK)
    w("")
    w("-" * 78)
    w("DRAFT prepared by decision-support system. Not filed. The officer is the author "
      "of record and must verify every item before registration.")
    w("-" * 78)
    return "\n".join(L)
