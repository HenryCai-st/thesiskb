"""Audio/video logic with Whisper, pyannote, ffmpeg and the VLM faked."""

import json
import subprocess
from types import SimpleNamespace

from thesiskb.core.config import ChunkingConfig
from thesiskb.ingest import video
from thesiskb.ingest.audio import (
    SpeakerTurn,
    Transcript,
    Word,
    assign_speakers,
    build_transcript,
    merge_turns,
    suggest_names,
    time_windows,
    transcript_blocks,
    ts,
)
from thesiskb.ingest.chunking import chunk_blocks

DIAR = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.5, "end": 9.0, "speaker": "SPEAKER_01"},
]


def _words(*spec):
    return [Word(start=a, end=b, text=f" {t}") for a, b, t in spec]


def test_assign_speakers_by_overlap_and_nearest_gap():
    words = assign_speakers(
        _words(
            (1, 2, "hello"),
            (4.8, 5.4, "edge"),
            (6, 7, "reply"),
            (9.5, 9.8, "late"),
            (20, 21, "lost"),
        ),
        DIAR,
    )
    assert [w.speaker for w in words] == [
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_01",
        None,
    ]


def test_merge_turns_splits_on_speaker_and_long_pause():
    ws = _words((0, 1, "a"), (1, 2, "b"), (6, 7, "c"), (7, 8, "d"))
    for w, s in zip(ws, ["S0", "S0", "S0", "S1"], strict=True):
        w.speaker = s
    turns = merge_turns(ws, max_pause=2.0)
    assert [(t.speaker, t.text) for t in turns] == [("S0", "a b"), ("S0", "c"), ("S1", "d")]


def test_build_transcript():
    asr = {
        "language": "en",
        "duration": 9.0,
        "words": [w.model_dump() for w in _words((1, 2, "Hi"), (6, 7, "Hello"))],
    }
    t = build_transcript(asr, DIAR)
    assert t.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert [x.text for x in t.turns] == ["Hi", "Hello"]


def test_time_windows():
    items = [(i * 20.0, i * 20.0 + 15) for i in range(12)]  # 0..235 s
    windows = time_windows(items, 60, 120)
    assert all(z - a <= 120 for a, z in windows)
    assert windows[0][0] == 0 and windows[-1][1] == 235


def test_transcript_blocks_names_frames_and_chunks():
    tr = Transcript(
        language="en",
        turns=[
            SpeakerTurn(start=23.2, end=33.4, speaker="SPEAKER_03", text="My name is Will I.M."),
            SpeakerTurn(start=35.0, end=41.0, speaker="SPEAKER_02", text="Developers want..."),
        ],
        speakers=["SPEAKER_02", "SPEAKER_03"],
    )
    frames = video.frame_blocks([video.Keyframe(t=30.0, path="f/30.jpg", caption="A box.")])
    blocks = transcript_blocks(tr, {"SPEAKER_03": "Will.i.am"}, (60, 120), frames)
    assert [b.kind for b in blocks] == ["turn", "frame", "turn"]  # interleaved by time
    assert blocks[0].text == "[0:23] Will.i.am: My name is Will I.M."
    assert blocks[2].text.startswith("[0:35] SPEAKER_02:")
    assert blocks[0].section == ["0:23–0:41"]

    chunks = chunk_blocks(blocks, ChunkingConfig(), "Meeting: X")
    assert len(chunks) == 1
    loc = chunks[0].locator
    assert (loc["t_start"], loc["t_end"]) == (23.2, 41.0)
    assert loc["speakers"] == ["SPEAKER_02", "SPEAKER_03"]
    assert loc["frames"] == [{"t": 30.0, "frame": "f/30.jpg"}]


def test_suggest_names():
    tr = Transcript(
        turns=[
            SpeakerTurn(start=0, end=1, speaker="S1", text="I'm a musician."),
            SpeakerTurn(
                start=1, end=2, speaker="S2", text="Hi, my name is Will I.M. I make music."
            ),
            SpeakerTurn(start=2, end=3, speaker="S3", text="Ich bin Anna Schmidt und forsche."),
        ]
    )
    assert suggest_names(tr) == {"S2": "Will I.M", "S3": "Anna Schmidt"}


def test_ts():
    assert ts(65) == "1:05" and ts(3725) == "1:02:05"


def test_pick_times_merges_scenes_and_fallback():
    times = video.pick_times([3.0, 3.5, 47.0], duration=70, every=30, min_gap=2)
    assert times == [0, 3.0, 30, 47.0, 60]


def test_scene_times_parses_showinfo(monkeypatch, tmp_path):
    stderr = "[Parsed_showinfo_1] n:0 pts:1 pts_time:4.2 \n[Parsed_showinfo_1] n:1 pts_time:17.08"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stderr=stderr))
    assert video.scene_times(tmp_path / "v.mp4", 0.3) == [4.2, 17.08]


def test_read_keyframes_captions_each_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "_probe_duration", lambda v: 65.0)
    monkeypatch.setattr(video, "scene_times", lambda v, t: [10.0])

    def fake_extract(v, times, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for t in times:
            p = out_dir / f"{int(t * 1000):08d}.jpg"
            p.write_bytes(b"\xff\xd8")
            frames.append(video.Keyframe(t=t, path=p.as_posix()))
        return frames

    monkeypatch.setattr(video, "extract_frames", fake_extract)

    class FakeVLM:
        def chat(self, messages, role="vision", **kw):
            assert role == "vision"
            return "Jetson board on a table. Text: $249"

    frames = video.read_keyframes(FakeVLM(), tmp_path / "v.mp4", tmp_path, every=30)
    assert [f.t for f in frames] == [0, 10.0, 30, 60]
    saved = json.loads((tmp_path / "frames.json").read_text(encoding="utf-8"))
    assert saved[1]["caption"].endswith("$249")


def test_suggest_names_trims_following_pronoun():
    tr = Transcript(
        turns=[
            SpeakerTurn(start=0, end=1, speaker="S", text="My name is Will I.M. I'm a musician.")
        ]
    )
    assert suggest_names(tr) == {"S": "Will I.M"}


def test_long_monologue_split_at_sentence_ends():
    ws = []
    for i in range(30):  # 30 two-second sentences by one speaker, no pauses
        ws.append(Word(start=i * 2.0, end=i * 2.0 + 1.9, text=f" Sentence {i}.", speaker="S0"))
    turns = merge_turns(ws, max_pause=2.0, max_turn=20.0)
    assert len(turns) == 3
    assert all(t.end - t.start <= 22 for t in turns)
    assert turns[1].start == 22.0  # first sentence end after 20 s
