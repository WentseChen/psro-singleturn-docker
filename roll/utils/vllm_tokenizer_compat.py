"""Tokenizer attribute vLLM expects on newer Transformers backends."""

from __future__ import annotations


def _patch_vllm_tokenizer_runtime_compat() -> list[str]:
    """Add the property vLLM 0.10.2 expects to tokenizer backends.

    Recent Transformers can return ``TokenizersBackend`` rather than the
    historical ``PreTrainedTokenizerFast`` class.  The image-level patch is
    retained for worker subprocesses; this explicit runtime patch also covers
    the exact class loaded by this standalone evaluator before ``LLM(...)``
    constructs vLLM's cached-tokenizer wrapper.
    """

    def all_special_tokens_extended(self):
        tokens = []
        try:
            tokens.extend(list(self.added_tokens_decoder.values()))
        except Exception:
            pass
        try:
            for token in self.all_special_tokens:
                if token not in tokens:
                    tokens.append(token)
        except Exception:
            pass
        return tokens

    candidates = (
        ("transformers", "PreTrainedTokenizerBase"),
        ("transformers", "PreTrainedTokenizer"),
        ("transformers", "PreTrainedTokenizerFast"),
        ("transformers", "TokenizersBackend"),
        ("transformers.tokenization_utils_base", "PreTrainedTokenizerBase"),
        ("transformers.tokenization_utils", "PreTrainedTokenizer"),
        ("transformers.tokenization_utils_fast", "PreTrainedTokenizerFast"),
        ("transformers.tokenization_utils_tokenizers", "TokenizersBackend"),
    )
    patched: list[str] = []
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            tokenizer_class = getattr(module, class_name, None)
            if tokenizer_class is None:
                continue
            if not hasattr(tokenizer_class, "all_special_tokens_extended"):
                setattr(
                    tokenizer_class,
                    "all_special_tokens_extended",
                    property(all_special_tokens_extended),
                )
            if not hasattr(tokenizer_class, "all_special_tokens_extended"):
                raise RuntimeError(
                    f"failed to patch {module_name}.{class_name}"
                )
            qualified = f"{module_name}.{class_name}"
            if qualified not in patched:
                patched.append(qualified)
        except (ImportError, AttributeError):
            continue
    if not patched:
        raise RuntimeError(
            "No supported Transformers tokenizer class was available for "
            "the vLLM 0.10.2 compatibility patch"
        )
    return patched
