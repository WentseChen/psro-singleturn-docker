"""HTTP off-topic verifier: does a rewrite still ask what the seed asked?

Mirrors ``openrlhf/cli/serve_wildguard.py`` in shape -- one vLLM engine behind a
FastAPI ``POST`` endpoint -- but answers a different question.  WildGuard scores
harm and never sees the seed, so it cannot tell a faithful rewrite from a fixed
memorized request that happens to elicit harm; that blind spot is what let
attackers A3/A4/A6 collapse to one prompt for all 60 held-out seeds.

Request::

    {"queries": [{"game_idx": 0, "seed": "...", "rewrite": "..."}, ...]}

Response::

    {"labels": [{"game_idx": 0, "off_topic": true|false|null}, ...]}

``off_topic: null`` means the verifier's reply could not be parsed.  It is kept
distinct from ``false`` so a degrading verifier shows up in the episode rows
instead of quietly switching the penalty off.

The verifier model defaults to the same base checkpoint the policies start from,
so no extra weights are needed:

    OFFTOPIC_MODEL=/path/to/base OFFTOPIC_PORT=5312 \\
        python scripts/slurm/serve_offtopic.py
"""

import logging
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from roll.utils.offtopic_verifier import (  # noqa: E402
    OFFTOPIC_VERIFIER_VERSION,
    build_verifier_prompt,
    parse_verifier_output,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("offtopic")


def main() -> int:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = os.environ.get("OFFTOPIC_MODEL")
    if not model:
        raise SystemExit("set OFFTOPIC_MODEL to the verifier checkpoint")
    port = int(os.environ.get("OFFTOPIC_PORT", "5312"))
    util = float(os.environ.get("OFFTOPIC_UTIL", "0.28"))

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=util,
        max_model_len=int(os.environ.get("OFFTOPIC_MAX_LEN", "4096")),
        enforce_eager=False,
        seed=42,
    )
    # Greedy and short: the verdict is one word, and a sampled verdict would put
    # noise straight into the reward.
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8)

    app = FastAPI()

    @app.post("/verify_topic")
    async def verify_topic(request: Request):
        data = await request.json()
        queries = data.get("queries", [])
        prompts, keys = [], []
        for item in queries:
            seed = str(item.get("seed") or "")
            rewrite = str(item.get("rewrite") or "")
            keys.append(item.get("game_idx"))
            if not seed.strip():
                prompts.append(None)
                continue
            prompts.append(
                tokenizer.apply_chat_template(
                    build_verifier_prompt(seed, rewrite),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        runnable = [p for p in prompts if p is not None]
        outputs = llm.generate(runnable, params) if runnable else []
        verdicts, cursor = [], 0
        for prompt in prompts:
            if prompt is None:
                verdicts.append(None)
                continue
            verdicts.append(parse_verifier_output(outputs[cursor].outputs[0].text))
            cursor += 1

        labels = [
            {"game_idx": key, "off_topic": verdict}
            for key, verdict in zip(keys, verdicts)
        ]
        unparsed = sum(1 for v in verdicts if v is None)
        offtopic = sum(1 for v in verdicts if v is True)
        logger.info(
            "verified %d: %d off-topic, %d unparsed", len(labels), offtopic, unparsed
        )
        return JSONResponse({"labels": labels, "version": OFFTOPIC_VERIFIER_VERSION})

    @app.get("/health")
    async def health():
        return JSONResponse({"ok": True, "version": OFFTOPIC_VERIFIER_VERSION})

    logger.info("off-topic verifier on :%d using %s", port, model)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
