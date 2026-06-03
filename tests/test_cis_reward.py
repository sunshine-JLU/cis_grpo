"""Unit test for CISRewardManager on a hand-built mini batch."""

import os
import sys

import numpy as np
import torch
from tensordict import TensorDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
for p in (_PROJECT, os.path.join(_PROJECT, "verl")):
    if p not in sys.path:
        sys.path.insert(0, p)

from recipes.cis_grpo.cis_reward_manager import CISRewardManager  # noqa: E402
from verl import DataProto  # noqa: E402


class _TinyTok:
    eos_token = "<eos>"

    def __init__(self, vocab):
        self.vocab = vocab
        self.id2tok = {i: t for i, t in enumerate(vocab)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.id2tok.get(int(i), "?") for i in ids)


def _make_item(prompt_ids, prompt_mask, resp_ids, resp_mask, gt, is_cf, uid, data_source="ViRL39K"):
    prompt = torch.tensor(prompt_ids, dtype=torch.long)
    response = torch.tensor(resp_ids, dtype=torch.long)
    full_mask = torch.tensor(prompt_mask + resp_mask, dtype=torch.long)
    return {
        "prompts": prompt,
        "responses": response,
        "attention_mask": full_mask,
        "reward_model": {"ground_truth": gt},
        "data_source": data_source,
        "extra_info": {"is_cf": is_cf},
        "uid": uid,
    }


def _stack(items):
    plen = items[0]["prompts"].shape[0]
    rlen = items[0]["responses"].shape[0]
    batch = TensorDict(
        {
            "prompts": torch.stack([it["prompts"] for it in items]),
            "responses": torch.stack([it["responses"] for it in items]),
            "attention_mask": torch.stack([it["attention_mask"] for it in items]),
        },
        batch_size=[len(items)],
    )
    non_tensor = {
        "reward_model": np.array([it["reward_model"] for it in items], dtype=object),
        "data_source": np.array([it["data_source"] for it in items], dtype=object),
        "extra_info": np.array([it["extra_info"] for it in items], dtype=object),
        "uid": np.array([it["uid"] for it in items], dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor), plen, rlen


def _ids(s, vocab):
    return [vocab.index(c) for c in s]


def main():
    # vocab is just enough to encode the test strings.
    chars = list("<eos>") + ["\n", " "] + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}\\=,()-?:<>./")
    # Dedup keeping order.
    seen, vocab = set(), []
    for c in chars:
        if c not in seen:
            seen.add(c)
            vocab.append(c)
    tok = _TinyTok(vocab)

    # Each response is fixed-length (we pad with spaces). prompt is "Q "; valid_prompt_length=2.
    PROMPT = _ids("Q ", vocab)
    pmask = [1, 1]

    def make_resp(text):
        # Encode response text; pad with spaces; build mask: 1s for real chars, 0s for padding.
        ids = _ids(text, vocab)
        max_len = 64
        pad = [vocab.index(" ")] * (max_len - len(ids))
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        return ids + pad, mask

    # Group A (uid=g1): real rollouts get correct \boxed{42}; CF rollout ALSO gets \boxed{42}
    # → language shortcut detected → real bonus 0, CF penalty -beta*1.
    r_correct = "<think>r</think> \\boxed{42}"
    r_resp, r_mask = make_resp(r_correct)
    real_a1 = _make_item(PROMPT, pmask, r_resp, r_mask, "\\boxed{42}", is_cf=False, uid="g1")
    real_a2 = _make_item(PROMPT, pmask, r_resp, r_mask, "\\boxed{42}", is_cf=False, uid="g1")
    cf_a1 = _make_item(PROMPT, pmask, r_resp, r_mask, "\\boxed{42}", is_cf=True, uid="g1")

    # Group B (uid=g2): real says \boxed{7}, CF says \boxed{99} (different) → bonus alpha, CF penalty 0 (incorrect).
    real_b_text = "<think>r</think> \\boxed{7}"
    cf_b_text = "<think>r</think> \\boxed{99}"
    rb_resp, rb_mask = make_resp(real_b_text)
    cb_resp, cb_mask = make_resp(cf_b_text)
    real_b1 = _make_item(PROMPT, pmask, rb_resp, rb_mask, "\\boxed{7}", is_cf=False, uid="g2")
    cf_b1 = _make_item(PROMPT, pmask, cb_resp, cb_mask, "\\boxed{7}", is_cf=True, uid="g2")

    data, plen, rlen = _stack([real_a1, real_a2, cf_a1, real_b1, cf_b1])
    print(f"batch size: {len(data)}  prompt_len={plen} resp_len={rlen}")

    mgr = CISRewardManager(tok, num_examine=2, alpha=0.2, beta=0.3, use_format_in_base=True)
    out = mgr(data, return_dict=True)
    rt = out["reward_tensor"]
    info = out["reward_extra_info"]

    # Last valid token reward per item.
    def last_r(i):
        valid = data.batch["attention_mask"][i, plen:].sum().item()
        return rt[i, valid - 1].item()

    rs = [last_r(i) for i in range(len(data))]
    print("Per-item shaped reward:", [round(x, 3) for x in rs])
    print("info.acc =", info["acc"])
    print("info.sens_bonus =", info["sens_bonus"])
    print("info.is_cf =", info["is_cf"])

    # Group A: base_score = 1.0 (acc=1, format=1) → real shaped = 1.0 (no bonus), cf shaped = -0.3
    # Group B: real shaped = 1.0 + 0.2 = 1.2; cf shaped = -0.3 * 0 = 0
    expected = [1.0, 1.0, -0.3, 1.2, 0.0]
    for i, (got, exp) in enumerate(zip(rs, expected)):
        ok = abs(got - exp) < 1e-5
        print(f"  item {i}: got={got:.3f} expected={exp:.3f}  {'OK' if ok else 'FAIL'}")
        assert ok, f"item {i} mismatch"
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
