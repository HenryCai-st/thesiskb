"""EPUB/HTML: blocks without pages get section locators (they used to be dropped → 0 chunks)."""

import zipfile

from docling_core.types.doc import DocItemLabel, DoclingDocument

from thesiskb.answer.generate import locator_label
from thesiskb.api.app import text_fragment
from thesiskb.ingest.chunking import merge_locators
from thesiskb.ingest.detect import DetectedFile, default_source_type, detect
from thesiskb.ingest.documents import document_meta, extract_blocks


def _pageless_doc(tmp_path):
    d = DoclingDocument(name="book")
    d.add_title(text="Table of Contents")
    d.add_list_item(text="Chapter 1")
    d.add_title(text="Chapter 1")
    d.add_text(label=DocItemLabel.TEXT, text="First chapter text.")
    pic = d.add_picture()
    pic.captions.append(
        d.add_text(label=DocItemLabel.CAPTION, text="Wireframes of the app").get_ref()
    )
    d.add_title(text="Chapter 2")
    d.add_text(label=DocItemLabel.TEXT, text="Second chapter text.")
    out = tmp_path / "docling.json"
    d.save_as_json(out)
    return out


def test_pageless_blocks_get_section_locators(tmp_path):
    parsed = extract_blocks(_pageless_doc(tmp_path))
    texts = [b.text for b in parsed.blocks]
    assert "Chapter 1" not in texts  # the navigation (table of contents) is skipped
    assert texts == ["First chapter text.", "Image: Wireframes of the app", "Second chapter text."]
    assert [b.locator["part"] for b in parsed.blocks] == ["Chapter 1", "Chapter 1", "Chapter 2"]
    assert [b.locator["order"] for b in parsed.blocks] == [0, 1, 2]
    assert parsed.blocks[2].section == ["Chapter 2"] and parsed.page_texts == {}


def test_merged_locator_and_label():
    loc = merge_locators([{"part": "Chapter 2", "order": 4}, {"part": "Chapter 2", "order": 6}])
    assert loc == {"part": "Chapter 2", "order": 4, "order_end": 6}
    assert locator_label(loc) == "Chapter 2"
    assert "page" not in loc


def test_text_fragment_escapes_dashes_and_image_prefix():
    frag = text_fragment("Image: Low-fidelity wireframes, with flow\nmore")
    assert frag == "#:~:text=Low%2Dfidelity%20wireframes%2C%20with%20flow"


def test_epub_and_html_metadata(tmp_path):
    epub = tmp_path / "b.epub"
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/>'
            "</rootfiles></container>",
        )
        z.writestr(
            "OEBPS/content.opf",
            "<package><metadata><dc:title id='t'>Sample &amp; Book</dc:title>"
            "<dc:creator>Ann Author</dc:creator></metadata></package>",
        )
    assert document_meta(epub, "epub") == ("Sample & Book", ["Ann Author"])
    page = tmp_path / "p.html"
    page.write_text("<html><head><title>SideBy | Portfolio</title></head></html>", encoding="utf-8")
    assert document_meta(page, "document") == ("SideBy | Portfolio", [])
    assert document_meta(tmp_path / "missing.epub", "epub") == (None, [])


def test_saved_page_assets_are_not_separate_sources(tmp_path):
    (tmp_path / "Page.html").write_text("<html></html>", encoding="utf-8")
    assets = tmp_path / "Page_files"
    assets.mkdir()
    (assets / "cover.png").write_bytes(b"png")
    (tmp_path / "photo.jpg").write_bytes(b"jpg")
    names = sorted(f.path.name for f in detect([tmp_path]))
    assert names == ["Page.html", "photo.jpg"]


def test_default_source_types(tmp_path):
    f = lambda name, m: DetectedFile(path=tmp_path / name, sha256="0" * 64, modality=m)  # noqa: E731
    assert default_source_type(f("b.epub", "epub")) == "book"
    assert default_source_type(f("p.html", "document")) == "web"
    assert default_source_type(f("x.pdf", "pdf")) == "paper"
