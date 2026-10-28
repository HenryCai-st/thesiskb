"""Entity keys and the merge rule (Milestone 14), no database."""

import pytest

from thesiskb.ingest.entities import decide, entity_key


@pytest.mark.parametrize(
    "a, b",
    [
        ("lake surtan", "lake-surtan"),
        ("Lake_Surtan", "LAKE  surtan"),
        ("Monte-Carlo sampling", "Monte Carlo sampling"),
        ("Super Resolution", "super‑resolution"),  # non-breaking hyphen
        ("Straße", "STRASSE"),  # casefold
        ("ﬁne-tuning", "fine tuning"),  # NFKC ligature
    ],
)
def test_key_variants_are_one_entity(a, b):
    assert entity_key(a) == entity_key(b)


def test_key_keeps_different_names_apart():
    assert entity_key("GAN") != entity_key("GANs")  # plural: left to the merge rule
    assert entity_key("C++") != entity_key("C")


@pytest.mark.parametrize(
    "a, b, sim, shared, want",
    [
        ("GAN", "GANs", 0.916, 0, ("merged", "variant")),
        ("Variational Autoencoders", "Variational Auto-Encoder", 0.916, 0, ("merged", "variant")),
        ("Lebensmittelvergiftung", "Lebensmittel Vergiftung", 0.84, 0, ("merged", "variant")),
        ("Foresight process", "foresight processes", 0.944, 0, ("merged", "variant")),
        ("Bachelor Project", "Bachelor Projekt", 0.976, 0, ("merged", "similar")),
        ("Riemannian manifold", "manifold of Riemann", 0.944, 1, ("merged", "shared")),
        ("Possible futures", "Probable futures", 0.932, 0, ("suggested", "similar")),
        ("Jetson Nano", "Jetson Nano Super", 0.928, 0, ("suggested", "similar")),
        ("Song et al. [287]", "Song et al. [282]", 0.970, 1, None),  # different numbers
        ("Figure 2", "Figure 4", 0.990, 0, None),
        ("process", "proces", 0.70, 0, None),  # below every threshold
    ],
)
def test_merge_rule(a, b, sim, shared, want):
    assert decide(a, b, sim, shared) == want
