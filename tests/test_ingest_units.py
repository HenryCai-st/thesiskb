from datetime import date

from thesiskb.ingest import detect
from thesiskb.ingest.documents import (
    _furniture,
    _heading_depth,
    _strip_furniture,
    detect_language,
)
from thesiskb.ingest.metadata import arxiv_id_from_name, parse_arxiv_atom


def test_detect_hashes_classifies_and_dedupes(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"same")
    (tmp_path / "copy of a.pdf").write_bytes(b"same")
    (tmp_path / "talk.mp3").write_bytes(b"audio")
    (tmp_path / "notes.xyz").write_bytes(b"ignored")
    files = detect.detect([tmp_path])
    assert sorted(f.modality for f in files) == ["audio", "pdf"]
    assert all(len(f.sha256) == 64 for f in files)


def test_heading_depth():
    assert _heading_depth("2.1 Denoising Diffusion") == 2
    assert _heading_depth("3 DIFFUSION MODELS") == 1
    assert _heading_depth("A.2 Proofs") == 2
    assert _heading_depth("Contents") is None
    assert _heading_depth("2022 was a good year") == 1  # accepted ambiguity


def test_furniture_detects_running_headers():
    items = [("Accepted by ACM Computing Surveys", p) for p in range(1, 6)]
    items += [("Unique paragraph text", 2)]
    f = _furniture(items)
    assert f == {"Accepted by ACM Computing Surveys"}
    assert _strip_furniture("Accepted by ACM Computing Surveys robust learning", f) == (
        "robust learning"
    )


def test_detect_language():
    assert detect_language("Die Methode ist nicht für alle Daten geeignet und wir") == "de"
    assert detect_language("The method is not suitable for all data and we") == "en"


ATOM = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><title>Diffusion Models:
  A Survey</title><published>2022-09-02T17:00:00Z</published>
<author><name>Ling Yang</name></author><author><name>Yang Song</name></author>
</entry></feed>"""


def test_arxiv_metadata():
    assert arxiv_id_from_name("2209.00796v15.pdf") == "2209.00796v15"
    assert arxiv_id_from_name("my notes.pdf") is None
    meta = parse_arxiv_atom(ATOM, "2209.00796v15")
    assert meta.title == "Diffusion Models: A Survey"
    assert meta.authors == ["Ling Yang", "Yang Song"]
    assert meta.published == date(2022, 9, 2)
    assert meta.origin == "arXiv:2209.00796v15"
