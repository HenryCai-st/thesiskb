"""Image reading with the VLM and Tesseract faked."""

import json
import subprocess
from types import SimpleNamespace

from PIL import Image

from thesiskb.ingest import images
from thesiskb.ingest.chunking import merge_locators
from thesiskb.ingest.images import (
    ImageReading,
    OcrResult,
    VisionDescription,
    choose_text,
    image_blocks,
    read_image,
    run_ocr,
)


class FakeVLM:
    """Confident only when the image is landscape (our 'upright' in this test)."""

    def __init__(self, description):
        self.description = description
        self.calls = 0

    def _upright(self, messages):
        import base64
        import io

        url = messages[0]["content"][1]["image_url"]["url"]
        im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        return im.width > im.height

    def chat_with_logprobs(self, messages, role="vision", **kw):
        self.calls += 1
        good = self._upright(messages)
        return ("10.10.2026 - Proposal" if good else "garbled"), [-0.05 if good else -0.3] * 5

    def chat(self, messages, role="vision", **kw):
        return json.dumps(self.description)


def test_read_image_picks_confident_rotation(tmp_path, monkeypatch):
    src = tmp_path / "portrait.jpg"
    Image.new("RGB", (60, 100), "white").save(src)  # portrait: needs a 90° turn
    monkeypatch.setattr(images, "run_ocr", lambda *a, **k: OcrResult(text="x", mean_conf=40))
    vlm = FakeVLM({"kind": "handwritten_note", "description": "A timeline.", "notes": []})

    r = read_image(vlm, src, tmp_path / "out", ["eng"])
    assert r.rotation in (90, 270)
    assert r.size == (100, 60)
    assert r.transcription == "10.10.2026 - Proposal"
    assert r.chosen == "vlm"
    assert (tmp_path / "out" / "image.json").exists()
    assert (tmp_path / "out" / "upright.jpg").exists()
    assert vlm.calls == 5  # 4 orientation probes + 1 full transcription


def test_unparsable_description_keeps_raw_text(tmp_path, monkeypatch):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (100, 60), "white").save(src)
    monkeypatch.setattr(images, "run_ocr", lambda *a, **k: OcrResult())
    vlm = FakeVLM({})
    vlm.chat = lambda *a, **k: "sorry, no json"
    r = read_image(vlm, src, tmp_path / "out", ["eng"])
    assert r.vision.description == "sorry, no json"


def test_choose_text_prefers_ocr_only_for_confident_print():
    good = OcrResult(text="t", mean_conf=92, n_words=50)
    assert choose_text("printed_document", good) == "ocr"
    assert choose_text("handwritten_note", good) == "vlm"
    assert choose_text("printed_document", OcrResult(mean_conf=50, n_words=50)) == "vlm"


def _reading(**vision):
    return ImageReading(
        rotation=90,
        orientation_scores={0: -0.3, 90: -0.05, 180: -0.3, 270: -0.3},
        size=(100, 60),
        transcription="10.10.2026 - Proposal",
        transcription_logprob=-0.05,
        vision=VisionDescription(**vision),
        ocr=OcrResult(),
        chosen="vlm",
    )


def test_blocks_for_notes_and_text():
    plain = image_blocks(_reading(kind="handwritten_note", description="Timeline."), "/x/a.jpg")
    assert [b.kind for b in plain] == ["caption", "text"]
    assert plain[0].text == "Handwritten note: Timeline."
    assert all(b.locator == {"image": "/x/a.jpg", "rotation": 90} for b in plain)

    notes = image_blocks(
        _reading(
            kind="post_its",
            notes=[
                {"text": "Buy GPU", "color": "yellow", "position": "top-left"},
                {"text": "Email Prof", "position": "somewhere"},
            ],
        ),
        "/x/b.jpg",
    )
    assert [b.kind for b in notes] == ["caption", "note", "note"]
    assert notes[1].text == "Buy GPU (yellow note)"
    assert notes[1].locator["region"] == [0, 0, 0.5, 0.5]
    assert "region" not in notes[2].locator
    merged = merge_locators([b.locator for b in notes])
    assert merged["image"] == "/x/b.jpg" and merged["regions"] == [[0, 0, 0.5, 0.5]]


TSV = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t\n"
    "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tHello\n"
    "5\t1\t1\t1\t1\t2\t0\t0\t10\t10\t80\tWorld\n"
    "5\t1\t1\t1\t2\t1\t0\t0\t10\t10\t70\tAgain\n"
)


def test_run_ocr_parses_tsv(tmp_path, monkeypatch):
    seen = {}

    def fake_run(args, **kw):
        seen["args"] = args
        return SimpleNamespace(stdout=TSV)

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = run_ocr(tmp_path / "x.jpg", ["deu", "eng"], "tess", "/td")
    assert r.text == "Hello World\nAgain"
    assert r.n_words == 3 and r.mean_conf == 80.0
    assert seen["args"][1:3] == ["--tessdata-dir", "/td"]
    assert "deu+eng" in seen["args"]


def test_run_ocr_missing_binary(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("tesseract")

    monkeypatch.setattr(subprocess, "run", boom)
    assert run_ocr(tmp_path / "x.jpg", ["eng"]) == OcrResult()
