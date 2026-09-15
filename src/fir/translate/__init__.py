"""Translation stage. `default_translator()` is what the graph builders use."""

from __future__ import annotations


def default_translator():
    """An IndicTranslator if IndicTrans2 is in the local cache, else None.

    Returning None (rather than an unavailable translator) keeps the graph a
    clean no-op on English-only boxes; the node itself reports the missing
    model when Tamil input actually arrives and a translator *is* configured.
    """
    from fir.translate.indictrans import IndicTranslator

    tr = IndicTranslator()
    return tr if tr.available() else None
