# ADR 0004 — Reading photos: rotation by confidence, VLM for handwriting

Date: 2026-10-21 · Status: accepted (answers BUILD_GUIDE §11.2 "OCR vs VLM" for now)

Fixture: a handwritten thesis timeline photographed sideways, no EXIF orientation tag.

## Finding the upright rotation
- Transcribing the sideways photo made qwen2.5vl:7b invent content ("120 pages", "Jun 2027").
- Asking the VLM which way the text is rotated returned the same answer for all four rotations.
- Tesseract OSD on handwriting: confidence 1.0, script "Japanese" — unusable.
- **Chosen:** transcribe a 48-token prefix at 0/90/180/270° and keep the rotation with the
  highest mean token logprob. 4/4 correct on the fixture rotated all four ways (−0.05 upright vs
  ≤ −0.11 otherwise). Cost: 4 short calls (~1 s each) per image.
- Logprobs need Ollama's OpenAI-compatible endpoint (`openai/qwen2.5vl:7b` at
  `{ollama_base_url}/v1`); LiteLLM's `ollama_chat` provider rejects `logprobs`.

## OCR vs VLM
- Handwriting: Tesseract (deu+eng) mean word confidence 53, text unreadable ("Sal, mission" for
  "Submission"). The VLM transcribed every item correctly once upright.
- **Chosen:** index the VLM transcription, except for `printed_document`/`screenshot` images where
  Tesseract is confident (mean ≥ 80, ≥ 20 words). Both readings are kept in `image.json`.
- Transcription prompt asks to keep a date and its label on one line ("Nov - Start thesis").
  Without it, the answer LLM paired "weekly meeting" with "March".

## Not done: reading order from arrows
Asked for the items "in arrow order", the 7B VLM consistently returned page-position order
(March before January on a clockwise loop). A wrong order stated as fact is worse than none, so
it is not indexed. Ordering by date is left to the answer LLM and to the tasks table (Milestone 5).

## Post-its
The VLM returns `notes: [{text, color, position}]`; each note becomes its own segment with an
approximate region (fraction of the upright image) in its locator. Not yet tested on a real
post-it photo — add one to the fixtures.
