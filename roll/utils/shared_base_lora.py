"""Shared-base serving (ONEJOB_SHARED_BASE=1, 10-08).

pi0_v7_s0 = abliterated Llama-3.1-8B-Instruct + a merged rank-64 SFT LoRA, and the
topic/reference/fulfil judge serves that same abliterated model.  With shared base, one
vLLM server per card serves the abliterated weights (judge = model "local-topic-judge"),
pi0 = the static LoRA "base" (SHARED_BASE_PI0_LORA, extracted by extract_pi0_lora.py),
and every trained adapter is served as ONE rank-128 LoRA = concat(pi0 LoRA, adapter):
    W_base + B_pi0 A_pi0 + s_t B_t A_t   with   A = [A_pi0; A_t],  B = [B_pi0, s_t B_t].
merged_path(path) returns the directory to hand to /v1/load_lora_adapter: the input path
when shared base is off, else a merged copy (inside the adapter dir for /dev/shm publish
dirs, so the publisher's rmtree removes it; else under SHARED_BASE_CACHE).
"""
import hashlib
import json
import os
import threading

_lock = threading.Lock()
_pi0 = None


def enabled() -> bool:
    return os.environ.get("ONEJOB_SHARED_BASE", "0") == "1"


def _load(path):
    from safetensors.torch import load_file
    return load_file(os.path.join(path, "adapter_model.safetensors"))


def _pi0_tensors():
    global _pi0
    with _lock:
        if _pi0 is None:
            d = os.environ["SHARED_BASE_PI0_LORA"]
            cfg = json.load(open(os.path.join(d, "adapter_config.json")))
            assert cfg["r"] == cfg["lora_alpha"], "pi0 LoRA must have scale 1"
            _pi0 = (_load(d), cfg["r"])
        return _pi0


def _norm(key: str) -> str:
    return key.replace(".default.", ".")


def merge(src: str, out: str) -> None:
    import torch
    from safetensors.torch import save_file
    pt, rp = _pi0_tensors()
    cfg = json.load(open(os.path.join(src, "adapter_config.json")))
    rt, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError(f"{src}: rslora/dora/rank_pattern adapters are not supported by the shared-base merge")
    st = alpha / rt
    tt = {_norm(k): v for k, v in _load(src).items()}
    prefixes = sorted({k.rsplit(".lora_", 1)[0] for k in list(pt) + list(tt)})
    out_t = {}
    for p in prefixes:
        ap, bp = pt.get(p + ".lora_A.weight"), pt.get(p + ".lora_B.weight")
        at, bt = tt.get(p + ".lora_A.weight"), tt.get(p + ".lora_B.weight")
        ref_a, ref_b = (ap, bp) if ap is not None else (at, bt)
        n_in, n_out = ref_a.shape[1], ref_b.shape[0]
        if ap is None:
            ap, bp = torch.zeros(rp, n_in), torch.zeros(n_out, rp)
        if at is None:
            at, bt = torch.zeros(rt, n_in), torch.zeros(n_out, rt)
        out_t[p + ".lora_A.weight"] = torch.cat([ap.float(), at.float()], 0).to(torch.bfloat16).contiguous()
        out_t[p + ".lora_B.weight"] = torch.cat([bp.float(), bt.float() * st], 1).to(torch.bfloat16).contiguous()
    r = rp + rt
    mods = sorted({p.rsplit(".", 1)[-1] for p in prefixes})
    tmp = out + ".tmp%d" % os.getpid()
    os.makedirs(tmp, exist_ok=True)
    save_file(out_t, os.path.join(tmp, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": r, "lora_dropout": 0.0,
               "bias": "none", "target_modules": mods, "fan_in_fan_out": False, "inference_mode": True,
               "use_rslora": False, "use_dora": False, "modules_to_save": None,
               "shared_base_merge": {"src": os.path.abspath(src), "pi0": os.environ["SHARED_BASE_PI0_LORA"],
                                     "src_r": rt, "src_scale": st}},
              open(os.path.join(tmp, "adapter_config.json"), "w"), indent=1)
    if os.path.isdir(out):
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)


def merged_path(path: str) -> str:
    if not enabled() or not path:
        return path
    src = os.path.abspath(path)
    publish_root = os.path.abspath(os.environ.get("ROLLOUT_LORA_DIR", "/dev/shm/ssp_rollout_lora"))
    if src.startswith(publish_root + os.sep) or src.startswith("/dev/shm/"):
        out = os.path.join(src, "sb128")
    else:
        mtime = os.path.getmtime(os.path.join(src, "adapter_model.safetensors"))
        key = hashlib.sha1(f"{src}:{mtime}".encode()).hexdigest()[:16]
        out = os.path.join(os.environ.get("SHARED_BASE_CACHE", "/dev/shm/ssp_sb_merged"), key)
    # 10-09 fix: version names restart at v0 on every resumed segment and published dirs are kept
    # (ROLLOUT_KEEP_VERSIONS), so a stale sb128 from an earlier segment was served under a reused name
    # (behaviour != trainer policy, token_mask 4-17% right after every restart). Re-merge when stale.
    dst = os.path.join(out, "adapter_model.safetensors")
    raw = os.path.join(src, "adapter_model.safetensors")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(raw):
        import shutil
        shutil.rmtree(out, ignore_errors=True)
        merge(src, out)
    return out
