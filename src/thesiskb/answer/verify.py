"""Citation checks (BUILD_GUIDE §8):
1. every cited number refers to a chunk that was in the context (and therefore exists);
2. every factual sentence has a citation;
3. an LLM judge checks each cited sentence is supported by its passages.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, ValidationError

from thesiskb.answer.generate import REFUSAL, Answer
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You check whether a claim is supported by the passages it cites.
Supported: the passages state it or directly imply it. Paraphrase and summarising are fine.
Unsupported: the claim adds facts, numbers or names that are not in the passages, or contradicts them.
Search the passages carefully for every name in the claim before deciding.

Passages:
{passages}

Claim: {claim}

Return JSON: {{"supported": true|false, "reason": "<one short sentence>"}}"""


class _Judgement(BaseModel):
    supported: bool
    reason: str = ""


def check_citations(ans: Answer) -> None:
    valid = {c.n for c in ans.context}
    for s in ans.sentences:
        bad = [n for n in s.citations if n not in valid]
        if s.status == "structural":
            continue
        if bad:
            s.status, s.note = "invalid_citation", f"cites unknown passage(s) {bad}"
        elif not s.citations:
            if REFUSAL.rstrip(".").lower() in s.text.lower():
                s.status, s.note = "supported", "refusal"
            else:
                s.status, s.note = "uncited", "no citation"


def judge_support(gateway: LLMGateway, ans: Answer) -> None:
    """One judge call per claim, with only the passages that claim cites.

    Judging all claims at once against every passage produced false "unsupported" verdicts
    with a 14B model (names present in a passage were reported missing).
    """
    by_n = {c.n: c for c in ans.context}
    for s in ans.sentences:
        if s.status != "unchecked":
            continue
        # the header (source title, section) is evidence too, e.g. a video titled "Jetson Orin
        # Nano Super" whose speaker only says "Jetson Nano Super"
        passages = "\n\n".join(
            f"[S{n}] ({by_n[n].context_header})\n{by_n[n].text}"
            if by_n[n].context_header
            else f"[S{n}] {by_n[n].text}"
            for n in s.citations
        )
        prompt = JUDGE_PROMPT.format(passages=passages, claim=s.text)
        try:
            raw = gateway.chat(
                [{"role": "user", "content": prompt}],
                role=gateway.role_for("judge", any(by_n[n].local_only for n in s.citations)),
                response_format={"type": "json_object"},
            )
            j = _Judgement.model_validate(json.loads(_strip_fences(raw)))
        except (ValidationError, ValueError) as e:
            log.warning("support judge returned unparsable output: %s", e)
            continue  # sentence stays "unchecked"
        s.status = "supported" if j.supported else "unsupported"
        s.note = j.reason or None


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return raw


def verify(gateway: LLMGateway, ans: Answer, check_support: bool = True) -> Answer:
    check_citations(ans)
    if check_support:
        judge_support(gateway, ans)
    return ans
