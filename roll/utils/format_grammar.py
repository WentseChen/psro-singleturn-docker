"""Grammar-constrained decoding for the CoT move format (FORMAT_GRAMMAR=1).

The prompt is prefilled with "<think>", so a completion must be
    <thinking> "</think>\\n<answer>" <answer> "</answer>"
with non-blank thinking and answer and no tag inside either.  This is exactly
what red_team.utils.cot_format_check_and_extract accepts, so a grammar-sampled
move can never be a format violation; only max_tokens truncation remains, and
v7 still scores that as an invalid move.  "<" is allowed inside text only when
followed by a space, "=", or a digit (comparisons), which rules out any tag.

Sampling is from the constrained policy pi_c.  vLLM applies the grammar mask
before it computes the returned log-probs, so a token the grammar forces comes
back with log-prob 0: the trainer drops those positions from the policy loss
and the KL (see FORCED_LOGP_EPS), because the model had no choice there.
"""

FORMAT_GRAMMAR_VERSION = "cot-format-grammar-v3-strict"

# Strict: EOS only after "</answer>" (v3).  v2 let the model stop anywhere and
# had the client finish the move; the stripped EOS decision then never got a
# gradient and an attacker collapsed to one-word moves (A1, 150-step check).
# Run this on pi0, which already writes the format, so the mask rarely binds;
# on the raw base it forces garbage when the model wants to stop early.
GRAMMAR = r'''
root ::= think "</think>\n<answer>" answer "</answer>"
think ::= [ \t\n]* [^< \t\n] char*
answer ::= [ \t\n]* [^< \t\n] char*
char ::= [^<] | "<" [ =0-9]
'''

# A sampled token whose constrained log-prob is above this was forced (or so
# certain that its gradient is ~0 anyway).
FORCED_LOGP_EPS = -1e-4


def structured_outputs() -> dict:
    return {"grammar": GRAMMAR}


# --- budget forcing: a move can never end truncated -------------------------
#
# A grammar cannot stop max_tokens truncation.  With FORMAT_GRAMMAR=1 the
# first pass gets max_tokens - RESCUE_RESERVE; if it stops on length, the
# missing closing tags are appended (forced, log-prob 0, masked from the loss)
# and the model writes the rest of its answer under the grammar within the
# reserve; if that runs out too, "</answer>" is forced.  Every move therefore
# ends in a well-formed "</answer>" (s1-style budget forcing).

import os

THINK_CLOSE = "</think>\n<answer>"
ANSWER_CLOSE = "</answer>"
ANSWER_GRAMMAR = r'''
root ::= answer "</answer>"
answer ::= [ \t\n]* [^< \t\n] char*
char ::= [^<] | "<" [ =0-9]
'''
CONTINUE_ANSWER_GRAMMAR = r'''
root ::= char* "</answer>"
char ::= [^<] | "<" [ =0-9]
'''


def rescue_reserve() -> int:
    return int(os.environ.get("FORMAT_RESCUE_RESERVE", "256"))


def _remainder(text: str, target: str) -> str:
    """The part of `target` still missing when `text` may already end with a
    prefix of it (e.g. a truncated "</thi")."""
    for k in range(min(len(target), len(text)), 0, -1):
        if text.endswith(target[:k]):
            return target[k:]
    return target


import re as _re

_TRAILING_SPECIAL = _re.compile(r"(?:<\|[A-Za-z0-9_]+\|>\s*)+$")


def _strip_special(text: str) -> tuple[str, int]:
    """Drop trailing special tokens (the EOS comes back as "<|eot_id|>" with
    skip_special_tokens=False); returns (text, how many were dropped)."""
    m = _TRAILING_SPECIAL.search(text)
    if not m:
        return text, 0
    return text[:m.start()], len(_re.findall(r"<\|[A-Za-z0-9_]+\|>", m.group(0)))


def needs_completion(text: str, finish_reason) -> bool:
    """Only truncation is completed: under the strict grammar a stopped move
    always ends in "</answer>", and the stop itself is the model's choice."""
    return finish_reason == "length"


def rescue_plan(text: str, finish_reason="length"):
    """(forced_text, grammar or None) to finish an incomplete move.  A move
    that stopped (EOS) inside its answer is closed, not continued."""
    if finish_reason != "length" and "<answer>" in text and ANSWER_CLOSE not in text:
        return _remainder(text, ANSWER_CLOSE), None
    if THINK_CLOSE.split("\n")[0] not in text:            # still thinking
        return _remainder(text, THINK_CLOSE), ANSWER_GRAMMAR
    after = text.split("<answer>", 1)
    if len(after) == 1:                                    # "</think>" but no "<answer>"
        return _remainder(text, "\n<answer>"), ANSWER_GRAMMAR
    tail = _remainder(text, ANSWER_CLOSE)
    if tail != ANSWER_CLOSE:                               # truncated inside "</answer>"
        return tail, None
    return "", (ANSWER_GRAMMAR if not after[1].strip() else CONTINUE_ANSWER_GRAMMAR)


async def budget_force(session, url, payload, choice, want_logprobs):
    """Finish an incomplete (truncated or early-stopped) completion in place.  `payload` is the original
    request (its max_tokens already reduced by the reserve); returns
    (token_ids, text, logprobs or None) of the whole move."""
    ids = list(choice.get("token_ids") or [])
    text = choice.get("text", "")
    lps = None
    if want_logprobs:
        lps = list((choice.get("logprobs") or {}).get("token_logprobs") or [])
    # an EOS must never sit in the middle of the move: drop it before forcing
    text, n_special = _strip_special(text)
    eos_ids = []
    if n_special:
        eos_ids = ids[-n_special:]
        ids = ids[:-n_special]
        if lps is not None:
            lps = lps[:-n_special]

    async def post_json(path, body):
        # keep-alive races (ServerDisconnected) are transient: retry, never
        # let one dropped connection kill the rollout pump
        import asyncio
        for attempt in range(6):
            try:
                async with session.post(f"{url}{path}", json=body) as r:
                    return r.status, await r.json(content_type=None)
            except Exception:
                if attempt == 5:
                    raise
                await asyncio.sleep(0.25 * (attempt + 1))

    async def tokenize(s):
        # The tokenizer does not depend on the adapter.  Asking with the LoRA
        # name raced adapter rotation: an in-flight move whose adapter was
        # unloaded got 404 and killed the rollout pump (orchard abl4b step 44).
        # Use the served base name first, the request's model as fallback.
        reply = None
        for model in dict.fromkeys((os.environ.get("ROLLOUT_BASE_NAME", "base"), payload["model"])):
            status, reply = await post_json("/tokenize", {"model": model, "prompt": s,
                                                          "add_special_tokens": False})
            if status == 200 and isinstance(reply, dict) and "tokens" in reply:
                return reply["tokens"]
        raise RuntimeError(f"/tokenize failed for every model name: {str(reply)[:200]}")

    forced, grammar = rescue_plan(text, choice.get("finish_reason"))
    for _ in range(2):
        if forced:
            f_ids = await tokenize(forced)
            ids += f_ids
            text += forced
            if lps is not None:
                lps += [0.0] * len(f_ids)
        if grammar is None:
            break
        # never ask for more than the server's context can still hold
        room = int(os.environ.get("ROLLOUT_MAX_MODEL_LEN", "4096")) - len(payload["prompt"]) - len(ids) - 8
        out = None
        if room > 0:
            body = dict(payload, prompt=list(payload["prompt"]) + ids, max_tokens=min(rescue_reserve(), room),
                        structured_outputs={"grammar": grammar})
            try:
                _, reply = await post_json("/v1/completions", body)
                out = (reply.get("choices") or [None])[0] if isinstance(reply, dict) else None
            except Exception:
                out = None   # fall back to closing the move with forced tags
        if out is None:
            # no room (or the server refused): close the move with forced tags
            forced = _remainder(text, ANSWER_CLOSE) if "<answer>" in text else ""
            if forced:
                f_ids = await tokenize(forced)
                ids += f_ids
                text += forced
                if lps is not None:
                    lps += [0.0] * len(f_ids)
            break
        o_ids = list(out.get("token_ids") or [])
        o_text, n_special = _strip_special(out.get("text", ""))
        o_lps = list((out.get("logprobs") or {}).get("token_logprobs") or [])
        if n_special:
            eos_ids = eos_ids or o_ids[-n_special:]
            o_ids, o_lps = o_ids[:-n_special], o_lps[:-n_special]
        ids += o_ids
        text += o_text
        if lps is not None:
            lps += o_lps
        # stopped or truncated inside the answer: close it
        forced, grammar = ("" if text.rstrip().endswith(ANSWER_CLOSE) else _remainder(text, ANSWER_CLOSE)), None
    # end like a natural stop: re-append the EOS the model emitted (forced,
    # log-prob 0, so it stays out of the loss); the text stays tag-clean
    if eos_ids:
        ids += eos_ids
        if lps is not None:
            lps += [0.0] * len(eos_ids)
    return ids, text, lps
