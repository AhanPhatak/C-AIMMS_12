"""Gold-evidence recall of HyperMem retrieval configs on the cached LoCoMo hypergraphs
(<outputs>/hypermem_cache/). CPU only, minutes per conversation. Produced the tuning
tables in adaptive_memory_structures/README_HYPERGRAPH.md. Run after `source env.sh`.

A question counts as a hit when every LoCoMo gold evidence turn is among the
retrieved turns (first 50 chars matched, after undoing the graph's date
resolution); mean-coverage is the fraction found.
Always reports "strict" (the original traversal) and two flat baselines over the
same turns (MiniLM only; MiniLM + BM25), plus each NAME:ENV config given.

usage: python3 scripts/hypermem_recall.py <samples, e.g. 1,2,3> [NAME:VAR=val,VAR=val ...]
  e.g. python3 scripts/hypermem_recall.py 1,2,3 default:HYPERMEM_MODE=soft \
         plain:HYPERMEM_PRF=0,HYPERMEM_RERANK=0,HYPERMEM_FACTS=0
Each config is the retriever's defaults plus the given overrides ("plain" above
is soft retrieval without PRF, rerank or facts -- the "soft" rows in the README).
"""
import json, os, re, sys, logging
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

# The cached graphs carry date-resolved turn text ("tomorrow (20 June, 2023)"), so
# the inserted parentheticals are stripped before matching against the dataset's
# raw text; gold ids are parsed leniently ("D8:6; D9:17", "D:11:26", "D30:05").
_RESOLVED = re.compile(r" \([^()]*\d{4}\)")
def gold_ids(evidence):
    return [f"D{a}:{int(b)}" for x in evidence for a, b in re.findall(r"D:?(\d+):(\d+)", x)]

def hit_frac(ev, gold, dia):
    """(fraction of gold turns found, n evidence, fraction of evidence that is gold)"""
    g = [x for x in gold if x in dia]
    if not g:
        return None
    ev = [_RESOLVED.sub("", e) for e in ev]
    found = sum(any(dia[x][:50] in e for e in ev) for x in g) / len(g)
    prec = sum(any(dia[x][:50] in e for x in g) for e in ev) / max(1, len(ev))
    return found, len(ev), prec

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
            f = hit_frac(r.retrieve(q["question"]), gold_ids(q["evidence"]), dia)
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
        g = gold_ids(q["evidence"])
        for name, ev in (("flat_dense", fd), ("flat_hybrid", fh)):
            f = hit_frac(ev, g, dia)
            if f is not None:
                rows[name].append((int(q["category"]), f))

print(f"samples {samples}  n={len(rows['strict'])}   all-gold-hit / mean-coverage  [multi temp open single]  n_ev  precision")
for name, rs in rows.items():
    a = np.array([f[0] for _, f in rs])
    per = " ".join(f"{np.mean([f[0] == 1 for c, f in rs if c == cat]):.2f}" for cat in (1, 2, 3, 4))
    print(f"  {name:14} {np.mean(a == 1):.3f} / {a.mean():.3f}   [{per}]  {np.mean([f[1] for _, f in rs]):5.1f}  {np.mean([f[2] for _, f in rs]):.3f}")
