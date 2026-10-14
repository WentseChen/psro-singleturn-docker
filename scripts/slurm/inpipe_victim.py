#!/usr/bin/env python3
"""inpipe_victim.py REPO PROMPTS_JSON OUT_JSON  (run with the rollout vLLM env, CUDA_VISIBLE_DEVICES = one victim card)

One local victim for the in-pipeline attacker eval: {arm: [prompt]} -> {arm: [reply]}, greedy, 512 new tokens,
thinking disabled (same protocol as ~/psro_orch/patches/local_victims.py / the 09-16 API runs). Empty prompts get
"__ERROR__ empty prompt". Adds "_load_s" (model load + warmup seconds) for the phase profile."""
import json
import os
import sys
import time

repo, pin, pout = sys.argv[1:4]
arms = json.load(open(pin))
t0 = time.time()
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

path = snapshot_download(repo, local_files_only=True)
tok = AutoTokenizer.from_pretrained(path)


def render(p):
    msgs = [{"role": "user", "content": p}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


llm = LLM(model=path, gpu_memory_utilization=float(os.environ.get("INPIPE_VICTIM_UTIL", "0.80")),
          max_model_len=4096, dtype="bfloat16", trust_remote_code=True)
load_s = time.time() - t0
sp = SamplingParams(temperature=0.0, max_tokens=512)
out = {"_load_s": round(load_s, 1)}
for arm, prompts in arms.items():
    idx = [i for i, p in enumerate(prompts) if str(p).strip()]
    res = llm.generate([render(prompts[i]) for i in idx], sp)
    replies = ["__ERROR__ empty prompt"] * len(prompts)
    for i, r in zip(idx, res):
        replies[i] = r.outputs[0].text
    out[arm] = replies
out["_gen_s"] = round(time.time() - t0 - load_s, 1)
json.dump(out, open(pout, "w"))
print(f"{repo}: load {load_s:.0f}s gen {out['_gen_s']:.0f}s", flush=True)
os._exit(0)
