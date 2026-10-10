"""Gold-evidence recall of HyperMem retrieval configs on the cached LoCoMo hypergraphs
(<outputs>/hypermem_cache/). CPU only, minutes per conversation. Produced the tuning
tables in adaptive_memory_structures/README_HYPERGRAPH.md. Run after `source env.sh`.

A question counts as a hit when every LoCoMo gold evidence turn is among the
retrieved turns (first 50 chars matched); mean-coverage is the fraction found.
Always reports "strict" (the original traversal) and two flat baselines over the
same turns (MiniLM only; MiniLM + BM25), plus each NAME:ENV config given.

usage: python3 scripts/hypermem_recall.py <samples, e.g. 1,2,3> [NAME:VAR=val,VAR=val ...]
  e.g. python3 scripts/hypermem_recall.py 1,2,3 soft:HYPERMEM_MODE=soft k60:HYPERMEM_RRF_K=60
"""
import json, os, sys, logging
import numpy as np
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.join(os.environ["CAIMMS_ROOT"], "adaptive_memory_structures"))
from sentence_transformers import SentenceTransformer
import locomo_hypermem as lh

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
memo = {}
def enc_batch(texts):
    todo = [t for t in texts if t not in memo]
    if todo:
        for t, v in zip(todo, model.encode(todo, batch_size=128, show_progress_bar=False)):
            memo[t] = v
    return np.stack([memo[t] for t in texts])
enc1 = lambda t: enc_batch([t])[0]

data_all = json.load(open(os.environ["CAIMMS_DATA_FILE"]))
samples = [int(x) for x in sys.argv[1].split(",")]
configs = [("strict", {"HYPERMEM_MODE": "strict"})]
for spec in sys.argv[2:]:
    name, _, rest = spec.partition(":")
    configs.append((name, dict(kv.split("=", 1) for kv in rest.split(",") if kv)))

def hit_frac(ev, gold, dia):
    g = [x for x in gold if x in dia]
    if not g:
        return None
    return sum(any(dia[x][:50] in e for e in ev) for x in g) / len(g)

rows = {name: [] for name, _ in configs + [("flat_dense", {}), ("flat_hybrid", {})]}
for si in samples:
    s = data_all[si]; conv = s["conversation"]
    dia = {t["dia_id"]: t["text"] for k, v in conv.items() if k.startswith("session_") and isinstance(v, list) for t in v}
    data = json.load(open(os.path.join(os.environ["CAIMMS_OUTPUT_DIR"], "hypermem_cache", f"sample_{si}.json")))
    qs = [q for q in s["qa"] if int(q["category"]) != 5 and q.get("evidence")]
    base_env = dict(os.environ)
    for name, env in configs:
        os.environ.clear(); os.environ.update(base_env); os.environ.update(env)
        r = lh.HyperMemRetriever(data, enc1, encode_batch=enc_batch)
        for q in qs:
            f = hit_frac(r.retrieve(q["question"]), [g.strip() for g in q["evidence"]], dia)
            if f is not None:
                rows[name].append((int(q["category"]), f))
    os.environ.clear(); os.environ.update(base_env)
    # flat baselines over the same turns
    r = lh.HyperMemRetriever(data, enc1, encode_batch=enc_batch)
    bm = r.bm["turn"]
    for q in qs:
        qv = r._embed([q["question"]])[0]
        dense = r.U @ qv
        fd = [r.turns[i] for i in np.argsort(-dense)[:r.max_turns]]
        fused = 1 / (60 + lh._ranks(dense)) + 1 / (60 + lh._ranks(bm.scores(q["question"])))
        fh = [r.turns[i] for i in np.argsort(-fused)[:r.max_turns]]
        g = [x.strip() for x in q["evidence"]]
        for name, ev in (("flat_dense", fd), ("flat_hybrid", fh)):
            f = hit_frac(ev, g, dia)
            if f is not None:
                rows[name].append((int(q["category"]), f))

print(f"samples {samples}  n={len(rows['strict'])}   all-gold-hit / mean-coverage  [multi temp open single]")
for name, rs in rows.items():
    a = np.array([f for _, f in rs])
    per = " ".join(f"{np.mean([f == 1 for c, f in rs if c == cat]):.2f}" for cat in (1, 2, 3, 4))
    print(f"  {name:14} {np.mean(a == 1):.3f} / {a.mean():.3f}   [{per}]")
