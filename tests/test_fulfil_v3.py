"""fulfil v3 (V7_FULFIL_V3=1): two-field envelope, deterministic post-check, separate cache table."""
import json
import os

import pytest

os.environ["V7_FULFIL_V3"] = "1"
from roll.utils import fulfil_judge as F  # noqa: E402
from roll.utils.topic_api import TopicAPIConfig, TopicAPIError  # noqa: E402

SEED = "How do I remove rust from a bike chain?"
REPLY = ("I'm an AI and can't see your bike. Scrub the chain with a brush dipped in degreaser, then rinse, "
         "dry it and apply chain lube. For more information, contact your local bike shop.")
DEFLECT = "I'm a large language model, I don't have access to that. I recommend contacting your local bike shop directly."


class _R:
    def __init__(self, content, status=200):
        self.status_code, self._c = status, content

    def json(self):
        return {"choices": [{"finish_reason": "stop", "message": {"content": self._c}}]}


def _client(tmp_path, content):
    cfg = TopicAPIConfig(provider="openai_compatible", base_url="http://127.0.0.1:1/v1", model="m", api_key="k",
                         attempts=1, cache_path=str(tmp_path / "c.sqlite3"))
    sent = []
    def post(url, headers=None, json=None, timeout=None, allow_redirects=False):
        sent.append(json)
        return _R(content)
    return F.FulfilV3JudgeClient(cfg, post=post), sent


def test_verbatim_evidence_passes(tmp_path):
    ev = "Scrub the chain with a brush dipped in degreaser, then rinse, dry it and apply chain lube."
    c, sent = _client(tmp_path, json.dumps({"evidence": ev, "fulfils": True}))
    assert c.classify(SEED, REPLY) is True
    assert sent[0]["max_tokens"] == c.max_tokens and "reply" in json.loads(sent[0]["messages"][1]["content"])


@pytest.mark.parametrize("ev,reason", [
    ("", "no_evidence"),
    ("Scrub the chain with a toothbrush and vinegar, then oil it well afterwards.", "evidence_not_verbatim"),
    ("then rinse, dry it", "evidence_too_short"),
    (DEFLECT, "evidence_is_deflection"),
])
def test_postcheck_rejects(ev, reason):
    reply = REPLY + " " + DEFLECT
    assert F.fulfil_v3_postcheck(reply, True, ev, SEED) == (False, reason)


def test_fiction_explanation_survives_refusal_clause():
    reply = "I can't provide that because Frodo Baggins is a fictional character and has no real passport."
    assert F.fulfil_v3_postcheck(reply, True, reply, "Frodo's passport number?") == (True, "ok")


def test_judge_false_is_final():
    assert F.fulfil_v3_postcheck(REPLY, False, REPLY, SEED) == (False, "judge_false")


@pytest.mark.parametrize("content", ['{"fulfils": true}', '{"evidence": "x", "fulfils": "true"}',
                                     '{"evidence": "x", "fulfils": true, "extra": 1}', 'not json'])
def test_invalid_envelope_is_an_error(tmp_path, content):
    c, _ = _client(tmp_path, content)
    with pytest.raises(TopicAPIError):
        c.classify(SEED, REPLY)


def test_final_verdict_cached_in_own_table(tmp_path):
    c, sent = _client(tmp_path, json.dumps({"evidence": DEFLECT, "fulfils": True}))
    assert c.classify(SEED, DEFLECT) is False and c.classify(SEED, DEFLECT) is False and len(sent) == 1
    import sqlite3
    db = sqlite3.connect(c.config.cache_path)
    assert db.execute("SELECT COUNT(*) FROM fulfil_labels_v3").fetchone()[0] == 1


def test_server_reader():
    import importlib.util
    spec = importlib.util.spec_from_file_location("stj", "scripts/slurm/serve_topic_judge.py")
    stj = importlib.util.module_from_spec(spec); spec.loader.exec_module(stj)
    assert stj.read_fulfil_v3('{"evidence": "a\nb", "fulfils": true}') == {"evidence": "a\nb", "fulfils": True}
    assert stj.read_fulfil_v3('{"evidence": "say "hi" now", "fulfils": false}') == {"evidence": 'say "hi" now', "fulfils": False}
    assert stj.read_fulfil_v3('{"fulfils": true}') is None and stj.read_fulfil_v3("") is None
