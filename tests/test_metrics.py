"""Tests para src/unravel/metrics.py."""
from __future__ import annotations

import pytest

from unravel.metrics import cer, cer_macro, cer_micro, edit_distance, wer


# === Edit distance ===

def test_edit_distance_identicos() -> None:
    assert edit_distance("aunque", "aunque") == 0


def test_edit_distance_sustitucion() -> None:
    assert edit_distance("annque", "aunque") == 1


def test_edit_distance_insercion() -> None:
    assert edit_distance("auue", "auque") == 1


def test_edit_distance_borrado() -> None:
    assert edit_distance("aunques", "aunque") == 1


def test_edit_distance_strings_vacios() -> None:
    assert edit_distance("", "") == 0
    assert edit_distance("a", "") == 1
    assert edit_distance("", "abc") == 3


def test_edit_distance_normalizacion_unicode_nfc_vs_nfd() -> None:
    """'á' en NFC (1 codepoint) y NFD (2 codepoints): edit_distance debe dar 0."""
    nfc = "\u00e1"           # á compuesta
    nfd = "a\u0301"          # a + combining acute
    assert edit_distance(nfc, nfd) == 0


def test_edit_distance_ignora_whitespace_exterior() -> None:
    assert edit_distance("  aunque  ", "aunque") == 0


# === CER (single sample) ===

def test_cer_perfecto() -> None:
    assert cer("aunque", "aunque") == 0.0


def test_cer_un_error_en_seis() -> None:
    assert cer("annque", "aunque") == pytest.approx(1 / 6)


def test_cer_levanta_si_referencia_vacia() -> None:
    with pytest.raises(ValueError):
        cer("algo", "")


def test_cer_no_aplica_lowercase() -> None:
    """Mayúsculas son consideradas letras distintas → CER > 0."""
    valor = cer("AUNQUE", "aunque")
    assert valor == pytest.approx(1.0)


def test_cer_no_quita_tildes() -> None:
    """Predicción sin tilde vs referencia con tilde → 1 error."""
    assert cer("aun", "aún") == pytest.approx(1 / 3)


# === CER micro ===

def test_cer_micro_caso_ejemplo_de_notas() -> None:
    """Caso documentado en notas-para-informe sección 3.6.1.

    real='a', pred=''       → edit_dist=1, len_ref=1
    real='aunque', pred='annque' → edit_dist=1, len_ref=6
    micro = (1+1)/(1+6) = 2/7
    """
    valor = cer_micro(["", "annque"], ["a", "aunque"])
    assert valor == pytest.approx(2 / 7)


# === CER macro ===

def test_cer_macro_caso_ejemplo_de_notas() -> None:
    """Mismo caso pero macro: promedio simple.

    cer por palabra: 1.0 y 1/6
    macro = (1.0 + 1/6) / 2
    """
    valor = cer_macro(["", "annque"], ["a", "aunque"])
    assert valor == pytest.approx((1.0 + 1 / 6) / 2)


def test_cer_micro_y_macro_difieren() -> None:
    micro = cer_micro(["", "annque"], ["a", "aunque"])
    macro = cer_macro(["", "annque"], ["a", "aunque"])
    assert micro != macro


# === WER ===

def test_wer_todas_correctas() -> None:
    assert wer(["aunque", "fuerzas"], ["aunque", "fuerzas"]) == 0.0


def test_wer_una_mal() -> None:
    assert wer(["aunque", "feurzas"], ["aunque", "fuerzas"]) == 0.5


def test_wer_todas_mal() -> None:
    assert wer(["x", "y"], ["aunque", "fuerzas"]) == 1.0


def test_wer_un_solo_caracter_distinto_ya_es_error() -> None:
    """WER no es gradual: 0 o 1 por palabra."""
    assert wer(["annque"], ["aunque"]) == 1.0


def test_wer_distinto_de_cer_micro_en_mismo_dataset() -> None:
    """WER 100% mientras CER es bajo: hay errores pero pequeños."""
    preds = ["annque"]
    refs = ["aunque"]
    assert wer(preds, refs) == 1.0
    assert cer_micro(preds, refs) == pytest.approx(1 / 6)


# === Validaciones ===

def test_cer_micro_levanta_si_longitudes_distintas() -> None:
    with pytest.raises(ValueError):
        cer_micro(["a"], ["a", "b"])


def test_wer_levanta_si_listas_vacias() -> None:
    with pytest.raises(ValueError):
        wer([], [])
