from __future__ import annotations
from typing import List, Tuple, Dict
import math
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleScorer:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(Device).eval()

    @torch.no_grad()
    def ppl(self, text: str) -> float:
        """普通 PPL"""
        if not text.strip():
            return 1.0
        enc = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(input_ids=enc["input_ids"].to(Device),
                             labels=enc["input_ids"].to(Device))
        loss = outputs.loss.item()
        return math.exp(loss)

    @torch.no_grad()
    def ppl_conditional(self, q: str, d: str) -> float:
        """条件 PPL：只计算 q 的 loss"""
        if not q.strip():
            return 1.0
        dq = (d + " " + q).strip() if d.strip() else q
        enc = self.tokenizer(dq, return_tensors="pt")
        ids_all = enc["input_ids"]

        q_ids = self.tokenizer(q, return_tensors="pt")["input_ids"]
        q_len = q_ids.size(1)

        # mask 掉 d，只保留 q 的 token
        labels = torch.full_like(ids_all, -100)
        labels[:, -q_len:] = ids_all[:, -q_len:]

        outputs = self.model(input_ids=ids_all.to(Device),
                             labels=labels.to(Device))
        loss = outputs.loss.item()
        return math.exp(loss)

    # ----- MoC 指标 -----
    def bc(self, q: str, d: str) -> float:
        """Boundary Clarity"""
        base = self.ppl(q)
        cond = self.ppl_conditional(q, d)
        return cond / max(base, 1e-6)

    def edge(self, q: str, d: str) -> float:
        """Edge strength"""
        base = self.ppl(q)
        cond = self.ppl_conditional(q, d)
        val = (base - cond) / max(base, 1e-6)
        return float(min(1.0, max(0.0, val)))


# --------- BC & CS 的计算函数 ---------
def bc_per_boundary(chunks: List[str], scorer: SimpleScorer) -> List[float]:
    return [scorer.bc(chunks[i+1], chunks[i]) for i in range(len(chunks)-1)]


def compute_cs(chunks: List[str], scorer: SimpleScorer,
               mode: str = "sequential", K: float = 0.5) -> Tuple[float, Dict]:
    n = len(chunks)
    edges = {}
    deg = [0] * n

    def add_edge(i, j, w):
        if w > K:
            edges[(i, j)] = w
            deg[i] += 1
            deg[j] += 1

    if mode == "complete":
        for i in range(n):
            for j in range(i+1, n):
                w = max(scorer.edge(chunks[j], chunks[i]),
                        scorer.edge(chunks[i], chunks[j]))
                add_edge(i, j, w)
    else:  # sequential
        for i in range(n-1):
            w = max(scorer.edge(chunks[i+1], chunks[i]),
                    scorer.edge(chunks[i], chunks[i+1]))
            add_edge(i, i+1, w)

    m = len(edges)
    if m == 0:
        return 0.0, {"edges": {}, "degrees": deg}

    total = 2 * m
    cs = 0.0
    for h in deg:
        if h > 0:
            p = h / total
            cs += -p * math.log(p, 2)

    return cs, {"edges": edges, "degrees": deg}
