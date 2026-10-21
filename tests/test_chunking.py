from thesiskb.core.config import ChunkingConfig
from thesiskb.ingest.chunking import Block, chunk_blocks, merge_locators, split_sentences


def _blocks(section, n, words=50, page=1):
    return [
        Block(
            text=" ".join(f"w{i}" for i in range(words)) + ". Second sentence here.",
            section=section,
            locator={"page": page + j // 3, "bbox": [0, 0, 1, 1]},
        )
        for j in range(n)
    ]


CFG = ChunkingConfig(target_tokens=200, max_tokens=300, overlap_ratio=0.15)


def test_never_crosses_section_boundary():
    blocks = _blocks(["1 Intro"], 3) + _blocks(["2 Method"], 3)
    chunks = chunk_blocks(blocks, CFG, title="Paper: X")
    assert {tuple(c.section) for c in chunks} == {("1 Intro",), ("2 Method",)}
    for c in chunks:
        assert all(b in c.text for b in ["w0"])  # no empty chunks


def test_respects_target_size_and_overlaps():
    chunks = chunk_blocks(_blocks(["1 Intro"], 12), CFG)
    assert len(chunks) > 1
    assert all(c.n_tokens <= CFG.max_tokens for c in chunks)
    # overlap: the second chunk starts with the tail sentence of the first
    assert chunks[1].text.startswith("Second sentence here.")


def test_oversized_block_is_split():
    big = Block(text=" ".join(["word."] * 1000), section=["S"], locator={"page": 2})
    chunks = chunk_blocks([big], CFG)
    assert len(chunks) > 1
    assert all(c.n_tokens <= CFG.max_tokens for c in chunks)


def test_every_chunk_has_locator_and_header():
    chunks = chunk_blocks(_blocks(["3 Results"], 5, page=4), CFG, title="Paper: X (2024)")
    for c in chunks:
        assert c.locator["page"] >= 4
        assert c.context_header == "Paper: X (2024) › 3 Results"


def test_merge_locators_pages_and_times():
    loc = merge_locators([{"page": 3, "bbox": [1, 2, 3, 4]}, {"page": 5, "bbox": [0, 0, 1, 1]}])
    assert loc["page"] == 3 and loc["page_end"] == 5 and len(loc["boxes"]) == 2
    t = merge_locators([{"t_start": 10.0, "t_end": 20.0}, {"t_start": 20.0, "t_end": 35.5}])
    assert t == {"t_start": 10.0, "t_end": 35.5}


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
