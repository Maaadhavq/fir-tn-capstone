"""Rule-based extraction: dates, times, amounts, phones, vehicles -- with offsets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.extract.fill import fill_structured_slots  # noqa: E402
from fir.extract.rules import (  # noqa: E402
    Kind,
    extract_amounts_inr,
    extract_dates,
    extract_phone_numbers,
    extract_times,
    extract_vehicle_numbers,
)
from fir.schema.if1 import FirRecord  # noqa: E402


def _one(spans, kind):
    assert len(spans) == 1, [s.text for s in spans]
    assert spans[0].kind is kind
    return spans[0]


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,iso", [
    ("committed murder on 8.5.2010 at night", "2010-05-08"),
    ("on 08/05/2010", "2010-05-08"),
    ("on 8-5-10", "2010-05-08"),
    ("on 8 May 2010", "2010-05-08"),
    ("on 8th May, 2010", "2010-05-08"),
    ("on May 8 2010", "2010-05-08"),
    ("on 21 Sept 2025", "2025-09-21"),
    ("2010 ஆம் ஆண்டு 8 மே 2010 அன்று", "2010-05-08"),
    ("மார்ச் 15 2026 அன்று", "2026-03-15"),
])
def test_dates_parse(text, iso):
    s = _one(extract_dates(text), Kind.DATE)
    assert s.value == iso
    assert text[s.start:s.end] == s.text  # offsets point at the surface form


def test_numeric_date_is_day_first_with_month_first_as_alternative():
    s = _one(extract_dates("on 3.4.2026"), Kind.DATE)
    assert s.value == "2026-04-03"           # Indian convention
    assert s.alternatives == ["2026-03-04"]  # but the other reading is offered
    assert s.confidence < 0.9                # and confidence reflects the ambiguity


def test_unambiguous_numeric_date_has_no_alternative():
    s = _one(extract_dates("on 25.4.2026"), Kind.DATE)
    assert s.value == "2026-04-25" and not s.alternatives and s.confidence >= 0.9


def test_impossible_date_is_skipped():
    assert extract_dates("on 31.2.2026 and 45.13.2026") == []


def test_word_date_beats_numeric_confidence():
    n = _one(extract_dates("8.5.2010"), Kind.DATE)
    w = _one(extract_dates("8 May 2010"), Kind.DATE)
    assert w.confidence > n.confidence


# ---------------------------------------------------------------------------
# times
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,hhmm", [
    ("at 10:30 pm", "22:30"),
    ("at 10.30 am", "10:30"),
    ("around 7 pm", "19:00"),
    ("at 12 am", "00:00"),
    ("at 12 pm", "12:00"),
    ("at 9 o'clock in the morning", "09:00"),
    ("at 9 in the evening", "21:00"),
    ("காலை 10 மணி", "10:00"),
    ("இரவு 10 மணிக்கு", "22:00"),
    ("மாலை 6.30 மணியளவில்", "18:30"),
])
def test_times_parse(text, hhmm):
    s = _one(extract_times(text), Kind.TIME)
    assert s.value == hhmm


def test_bare_number_is_not_a_time():
    """'8' on its own is a number; only anchored numbers are times."""
    assert extract_times("there were 8 people") == []
    assert extract_times("section 302") == []


def test_unanchored_12h_time_has_reduced_confidence():
    anchored = _one(extract_times("at 10:30 pm"), Kind.TIME)
    bare = _one(extract_times("at 10:30 மணிக்கு"), Kind.TIME)  # marker, no am/pm
    assert bare.confidence < anchored.confidence


# ---------------------------------------------------------------------------
# amounts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,inr", [
    ("worth ₹50,000", 50000.0),
    ("Rs. 50000", 50000.0),
    ("Rs 1,50,000", 150000.0),         # Indian grouping
    ("INR 2500.50", 2500.5),
    ("5 lakh rupees", 500000.0),
    ("Rs 2 crore", 20000000.0),
    ("50,000 ரூபாய்", 50000.0),
    ("5 லட்சம் ரூபாய்", 500000.0),
    ("ரூ. 15000", 15000.0),
])
def test_amounts_parse(text, inr):
    s = _one(extract_amounts_inr(text), Kind.AMOUNT_INR)
    assert s.value == inr


def test_number_without_currency_is_not_an_amount():
    assert extract_amounts_inr("about 50000 people attended") == []


# ---------------------------------------------------------------------------
# phones & vehicles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,digits", [
    ("call 9876543210", "9876543210"),
    ("call +91 98765 43210", "9876543210"),
    ("call 098765-43210", "9876543210"),
])
def test_phones_parse(text, digits):
    assert _one(extract_phone_numbers(text), Kind.PHONE).value == digits


def test_non_mobile_ten_digit_is_not_a_phone():
    assert extract_phone_numbers("account 1234567890") == []  # starts with 1


@pytest.mark.parametrize("text,norm", [
    ("the car TN 01 AB 1234 sped off", "TN 01 AB 1234"),
    ("bike TN09Z4567", "TN 09 Z 4567"),
    ("KA-05-MN-9876", "KA 05 MN 9876"),
])
def test_vehicles_parse(text, norm):
    assert _one(extract_vehicle_numbers(text), Kind.VEHICLE).value == norm


# ---------------------------------------------------------------------------
# filling the record
# ---------------------------------------------------------------------------

COMPLAINT = (
    "நான் குமார். On 8.5.2026 at around 10:30 pm an unknown man stole my mobile "
    "phone worth ₹15,000 from my shop. My number is 9876543210. He left on a bike "
    "TN 09 Z 4567."
)


def test_fill_populates_unambiguous_slots_with_provenance():
    rec = FirRecord()
    rep = fill_structured_slots(rec, COMPLAINT, utterance_id="u1")

    assert set(rep.filled) == {
        "occurrence.from_datetime", "complainant.contact", "total_property_value_inr"
    }
    assert not rep.ambiguous

    occ = rec.occurrence.from_datetime
    assert occ.value == "2026-05-08T22:30"
    assert occ.status == "extracted"
    assert len(occ.provenance) == 2                # date span + time span
    assert all(p.utterance_id == "u1" for p in occ.provenance)
    assert COMPLAINT[occ.provenance[0].transcript_char_start:occ.provenance[0].transcript_char_end] == "8.5.2026"

    assert rec.complainant.contact.value == "9876543210"
    assert rec.total_property_value_inr.value == 15000.0

    # the vehicle has no IF-1 slot; it is surfaced, not dropped
    assert [s.value for s in rep.unplaced] == ["TN 09 Z 4567"]


def test_fill_refuses_to_guess_between_two_phone_numbers():
    rec = FirRecord()
    rep = fill_structured_slots(rec, "my number 9876543210, his is 9123456789")
    assert "complainant.contact" in rep.ambiguous
    assert rec.complainant.contact.status == "empty"
    assert rec.complainant.contact.value is None
    assert "officer to choose" in rec.complainant.contact.officer_note
    assert "9876543210" in rec.complainant.contact.officer_note


def test_fill_reads_two_ordered_dates_as_an_interval():
    rec = FirRecord()
    rep = fill_structured_slots(rec, "between 1.5.2026 and 3.5.2026 they harassed her")
    assert rec.occurrence.is_interval
    assert rec.occurrence.from_datetime.value == "2026-05-01"
    assert rec.occurrence.to_datetime.value == "2026-05-03"
    assert set(rep.filled) == {"occurrence.from_datetime", "occurrence.to_datetime"}


def test_fill_leaves_untouched_slots_empty():
    rec = FirRecord()
    fill_structured_slots(rec, "he hit me and ran away")
    assert rec.occurrence.from_datetime.status == "empty"
    assert rec.complainant.contact.status == "empty"
    assert rec.total_property_value_inr.status == "empty"
    assert rec.machine_filled_fields() == []


def test_filled_slots_appear_in_machine_filled_list():
    rec = FirRecord()
    fill_structured_slots(rec, COMPLAINT)
    assert set(rec.machine_filled_fields()) == {
        "occurrence.from_datetime", "complainant.contact", "total_property_value_inr"
    }


# ---------------------------------------------------------------------------
# spoken amounts via ITN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,inr,surface", [
    ("அவர் ஐம்பதாயிரம் ரூபாய் எடுத்தார்", 50000.0, "ஐம்பதாயிரம் ரூபாய்"),
    ("இரண்டு லட்சம் ரூபாய் மோசடி", 200000.0, "இரண்டு லட்சம் ரூபாய்"),
    ("he took fifty thousand rupees", 50000.0, "fifty thousand rupees"),
    ("Rs two lakh fifty thousand was paid", 250000.0, "Rs two lakh fifty thousand"),
    ("ரூ. ஐநூறு மட்டும்", 500.0, "ரூ. ஐநூறு"),
])
def test_spoken_amounts_are_extracted(text, inr, surface):
    s = _one(extract_amounts_inr(text), Kind.AMOUNT_INR)
    assert s.value == inr
    assert s.text == surface
    assert text[s.start:s.end].strip() == surface   # provenance covers the spoken words


def test_spoken_number_without_currency_is_not_an_amount():
    """ஆறு is 'six' and 'river'; without ரூபாய் next to it, it is not money."""
    assert extract_amounts_inr("ஆறு பேர் வந்தார்கள்") == []
    assert extract_amounts_inr("fifty people attended") == []


def test_spoken_ordinal_is_not_an_amount():
    assert extract_amounts_inr("பதினைந்தாம் தேதி ரூபாய் கொடுத்தார்") == []


def test_digit_and_spoken_amounts_in_one_text():
    spans = extract_amounts_inr("₹500 first, then ஐம்பதாயிரம் ரூபாய் later")
    assert sorted(s.value for s in spans) == [500.0, 50000.0]


# ---------------------------------------------------------------------------
# spoken times and dates via ITN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,hhmm,surface", [
    ("இரவு பத்து மணிக்கு வந்தார்", "22:00", "இரவு பத்து மணிக்கு"),
    ("காலை எட்டு மணி", "08:00", "காலை எட்டு மணி"),
    ("at ten in the morning", "10:00", "ten in the morning"),
    ("around seven pm", "19:00", "seven pm"),
])
def test_spoken_times(text, hhmm, surface):
    s = _one(extract_times(text), Kind.TIME)
    assert s.value == hhmm
    assert s.text == surface                  # provenance covers the spoken words
    assert text[s.start:s.end].strip() == surface


@pytest.mark.parametrize("text,iso,surface", [
    ("மே எட்டாம் தேதி சம்பவம் நடந்தது", "--05-08", "மே எட்டாம் தேதி"),
    ("மே மாதம் எட்டாம் தேதி", "--05-08", "மே மாதம் எட்டாம் தேதி"),
    ("on the 8th of May he came", "--05-08", "8th of May"),
    ("on May 8 he came", "--05-08", "May 8"),
    ("பதினைந்தாம் தேதி ஜூன்", "--06-15", "பதினைந்தாம் தேதி ஜூன்"),
])
def test_yearless_spoken_dates(text, iso, surface):
    s = _one(extract_dates(text), Kind.DATE)
    assert s.value == iso
    assert s.confidence < 0.7
    assert s.alternatives == ["year not stated"]
    assert s.text == surface


def test_full_date_is_not_also_reported_as_yearless():
    """'8 May 2026' must yield one full date, not a full one plus a partial."""
    spans = extract_dates("on 8 May 2026 at the shop")
    assert [s.value for s in spans] == ["2026-05-08"]


def test_yearless_date_still_validates_the_day():
    assert extract_dates("on the 31st of February") == []
    assert extract_dates("பிப்ரவரி முப்பதாம் தேதி") == []


def test_spoken_amount_still_carries_original_offsets_after_refactor():
    text = "அவர் ஐம்பதாயிரம் ரூபாய் எடுத்தார்"
    s = _one(extract_amounts_inr(text), Kind.AMOUNT_INR)
    assert text[s.start:s.end].strip() == "ஐம்பதாயிரம் ரூபாய்"
    assert s.confidence == 0.85               # spoken: a notch below typed digits


def test_fill_does_not_build_an_interval_from_a_yearless_and_a_full_date():
    rec = FirRecord()
    rep = fill_structured_slots(rec, "on May 8 he came, and again on 10.5.2026")
    assert not rec.occurrence.is_interval
    assert "occurrence.from_datetime" in rep.ambiguous    # two dates, cannot order -> ask


def test_fill_accepts_a_single_yearless_date_with_a_note():
    rec = FirRecord()
    rep = fill_structured_slots(rec, "மே மாதம் எட்டாம் தேதி காலை ஏழு மணிக்கு நடந்தது")
    assert "occurrence.from_datetime" in rep.filled
    assert rec.occurrence.from_datetime.value == "--05-08T07:00"
    note = rec.occurrence.from_datetime.officer_note or ""
    assert "year not stated" in note
    assert "day/month" not in note      # the note must describe the *actual* ambiguity
