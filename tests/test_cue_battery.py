"""Cue-scan battery over the station-house offences added on 2026-09-13.

Each case is a short complaint as a citizen would give it, in English or Tamil.
`must` are the sections whose every ingredient the text evidences; `must_not`
are the tempting wrong answers (a threat to kill is 506, not 307; a road death
is 304A, not 302). The scan is a recall net for the officer, so extra hits are
tolerated -- wrong *exclusions* and the specific confusions listed are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.statute.elements import (  # noqa: E402
    ELEMENTS,
    check_elements,
    covered_sections,
    scan_all,
)

CASES = [
    # (name, narrative, must, must_not)
    ("robbery_en", "Two men on a bike snatched my gold chain at knifepoint on Anna Salai and fled.",
     {"392"}, {"302", "307"}),
    ("robbery_ta", "இரண்டு பேர் பைக்கில் வந்து கத்தியை காட்டி மிரட்டி என் செயினை பறித்துச் சென்றனர்.",
     {"392"}, {"302"}),
    ("breach_of_trust", "I gave Rs 2 lakh to Kumar for a chit fund. He did not return the money and has stopped answering calls.",
     {"406"}, {"379", "392"}),
    ("kidnapping_en", "My 9-year-old son did not come home from school. A man in a white van took him away. He is missing since evening.",
     {"363"}, {"302", "366"}),
    ("kidnapping_ta", "என் மகள் பள்ளியில் இருந்து வீடு திரும்பவில்லை. ஒருவர் அவளை வேனில் கடத்திச் சென்றார்.",
     {"363"}, set()),
    ("road_death_en", "A lorry driven at high speed hit my father while he was crossing the road. He died on the spot.",
     {"279", "304A"}, {"302"}),
    ("road_death_ta", "லாரி வேகமாக வந்து என் தந்தையை மோதியது. அவர் இறந்துவிட்டார்.",
     {"279", "304A"}, {"302"}),
    ("rash_driving", "A drunk auto driver dashed against my bike on the main road and I fell down.",
     {"279"}, {"304A", "302"}),
    ("trespass_with_threat", "Murugan entered my land, cut the fence and threatened to kill me if I complained.",
     {"447", "506"}, {"307", "302"}),
    ("house_break_in", "Three persons broke into my house at night and damaged the door and the TV.",
     {"447", "448"}, {"302"}),
    ("stalking", "A man follows me every day from the bus stop and sends obscene messages to my phone.",
     {"509"}, {"354", "376"}),
    ("otp_fraud", "A person called me claiming he was from the bank and asked for the OTP, then Rs 40,000 was taken from my account.",
     {"419", "420"}, {"379"}),
    ("dowry_suicide", "My daughter committed suicide by hanging because her husband harassed her for dowry.",
     {"306", "498A"}, {"302"}),
    ("confinement", "My employer locked me in the godown for two days and did not allow me to leave.",
     {"341", "342"}, set()),
    ("extortion", "The rowdies demanded Rs 5,000 as mamool every month and threatened to break my shop.",
     {"384", "506"}, {"392"}),
    ("stabbing_survived", "Ravi stabbed me in the stomach with a knife near the bus stand and ran away. I was admitted in the hospital.",
     {"307", "324"}, {"302"}),
    ("theft_ta_inflected_phone", "நேற்று இரவு என் கடையில் இருந்து ஒருவர் என் போனை திருடிச் சென்றார்.",
     {"379", "380"}, set()),
    # negatives: nothing here evidences a covered offence in full
    ("defamation_only", "He published defamatory statements about me in the newspaper.", set(), set(ELEMENTS)),
    ("argument_slap", "the neighbour slapped him during an argument", set(), {"323", "324", "325", "307"}),
]


@pytest.mark.parametrize("name,text,must,must_not", CASES, ids=[c[0] for c in CASES])
def test_cue_scan_battery(name, text, must, must_not):
    hits = {h.ipc_section for h in scan_all(text)}
    assert must <= hits, f"{name}: missing {sorted(must - hits)}; got {sorted(hits)}"
    assert not (must_not & hits), f"{name}: wrongly fired {sorted(must_not & hits)}"


def test_second_batch_is_covered():
    for sec in ("392", "384", "406", "411", "341", "342", "363", "366", "324", "325", "307", "306",
                "279", "304A", "427", "447", "448", "509", "294", "419"):
        assert sec in covered_sections(), sec
    assert len(covered_sections()) == 29


def test_threat_to_kill_is_not_attempt_to_murder():
    """Same lesson as 302: 'I will kill you' is criminal intimidation."""
    checks = {c.element: c for c in check_elements("307", "he said he will kill me if I go to the police")}
    intent = next(c for k, c in checks.items() if k.startswith("act done with intention"))
    assert intent.satisfied == "unclear"
    assert "307" not in {h.ipc_section for h in scan_all("he said he will kill me if I go to the police")}


def test_every_new_cue_compiles_and_matches_something():
    """Guards against a regex that can never fire (a typo'd escape, an
    impossible group): each element's pattern must match at least one of its
    own cues once the regex syntax is stripped."""
    import re

    for sec, els in ELEMENTS.items():
        for el in els:
            pat = re.compile("|".join(f"(?:{c})" for c in el.cues), re.IGNORECASE)
            probes = [re.sub(r"\\b|\(\?:|\)|\?|\\d|\[.*?\]|\+|\*|\|", " ", c) for c in el.cues]
            assert any(pat.search(p) for p in probes) or any(pat.search(c) for c in el.cues), (sec, el.name)
