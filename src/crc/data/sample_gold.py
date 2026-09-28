"""Build the Agent 3 methodology gold-evaluation set — for a HUMAN to label.

Agent 3's 0.657 test number is agreement with the *weak* labels, not with truth.
The only honest measure is a hand-labelled gold set. This script does everything
except the labelling itself (which must be human):

  1. samples ~200 papers, stratified by weak design label so every design — the
     four trainable ones AND the arXiv-starved human-centric ones — is
     represented, plus some the weak labeller abstained on,
  2. shuffles them so the annotator cannot infer the design from the order,
  3. runs the trained classifier to record its prediction (hidden from the
     annotator, kept in a key file for scoring),
  4. emits a self-contained offline HTML tool with the codebook built in.

Outputs to `system/gold/`:
  * `annotate_methodology.html` — open in a browser, label, click Export.
  * `gold_key.json`            — weak label + model prediction per paper (do NOT
                                 open while labelling; it holds the comparisons).

Run:
    python -m crc.data.sample_gold
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from crc.taxonomy.methodology import DESIGNS, DESIGN_NAMES, TRAINABLE_DESIGNS

# Every design's cues, longest first so multi-word phrases win over their parts.
# They are highlighted in ONE neutral style with no design label attached: the
# annotator sees which sentences carry methodological signal (a paper usually has
# several, from different designs) but must still decide the primary design. This
# speeds reading without anchoring the answer.
_ALL_CUES = sorted({c for d in DESIGNS for c in d.cues}, key=len, reverse=True)
_CUE_RE = re.compile("|".join(re.escape(c) for c in _ALL_CUES), re.IGNORECASE)


def highlight(text: str) -> str:
    esc = _html.escape(text or "")
    return _CUE_RE.sub(lambda m: f"<mark>{m.group(0)}</mark>", esc)

WORK = Path(r"C:\Users\ziada\gp_data")
POOL = WORK / "corpus" / "methodology_pool.parquet"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
GOLD = PROJECT / "gold"

#: Options the annotator can choose — all nine designs plus two escape hatches.
OPTIONS = DESIGN_NAMES + ["Unsure / borderline", "Not empirical research / N/A"]


def model_predictions(texts: list[str]) -> list[str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    d = MODELS / "methodology-scibert"
    tok = AutoTokenizer.from_pretrained(str(d))
    model = AutoModelForSequenceClassification.from_pretrained(str(d))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    id2label = model.config.id2label
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, padding=True,
                      max_length=320, return_tensors="pt").to(dev)
            ids = model(**enc).logits.argmax(-1).cpu().tolist()
            out.extend(id2label[j] for j in ids)
    return out


def sample(df: pd.DataFrame, per_trainable: int, per_rare: int,
           n_unlabelled: int, rng) -> pd.DataFrame:
    rare = [d for d in DESIGN_NAMES if d not in TRAINABLE_DESIGNS]
    picks = []
    for d in TRAINABLE_DESIGNS:
        g = df[df["design"] == d]
        picks.append(g.sample(n=min(per_trainable, len(g)),
                              random_state=int(rng.integers(1 << 31))))
    for d in rare:
        g = df[df["design"] == d]
        if len(g):
            picks.append(g.sample(n=min(per_rare, len(g)),
                                 random_state=int(rng.integers(1 << 31))))
    un = df[df["design"].isna()]
    if len(un):
        picks.append(un.sample(n=min(n_unlabelled, len(un)),
                              random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["paper_id"])
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_html(papers: list[dict], codebook: list[dict], dsid: str) -> str:
    tmpl = _HTML
    tmpl = tmpl.replace("/*__PAPERS__*/", json.dumps(papers, ensure_ascii=False))
    tmpl = tmpl.replace("/*__CODEBOOK__*/", json.dumps(codebook, ensure_ascii=False))
    tmpl = tmpl.replace("/*__OPTIONS__*/", json.dumps(OPTIONS, ensure_ascii=False))
    tmpl = tmpl.replace("__DSID__", dsid)
    return tmpl


def main() -> None:
    ap = argparse.ArgumentParser()
    # Small human-feasible default: ~40 papers (9 per trainable design + 1 per
    # available rare design). Bump these for a larger set if there is appetite.
    ap.add_argument("--per-trainable", type=int, default=9)
    ap.add_argument("--per-rare", type=int, default=1)
    ap.add_argument("--unlabelled", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(POOL)
    picks = sample(df, args.per_trainable, args.per_rare, args.unlabelled, rng)
    print(f"sampled {len(picks)} papers for gold labelling")
    print(picks["design"].fillna("(weak-abstained)").value_counts().to_string())

    preds = model_predictions(picks["text"].tolist())

    GOLD.mkdir(parents=True, exist_ok=True)
    papers = [{"paper_id": str(r.paper_id), "discipline": r.discipline,
               "text_html": highlight((r.text or "")[:3500])}
              for r in picks.itertuples(index=False)]
    key = {str(r.paper_id): {"discipline": r.discipline,
                             "weak_design": (r.design if pd.notna(r.design) else None),
                             "model_pred": p}
           for r, p in zip(picks.itertuples(index=False), preds)}
    (GOLD / "gold_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False))

    codebook = [{"name": d.name, "description": d.description,
                 "boundary": d.contrast,
                 "trainable": d.name in TRAINABLE_DESIGNS} for d in DESIGNS]
    dsid = f"meth-gold-{len(papers)}"
    html = build_html(papers, codebook, dsid)
    (GOLD / "annotate_methodology.html").write_text(html, encoding="utf-8")

    print(f"\ntool -> {GOLD / 'annotate_methodology.html'}")
    print(f"key  -> {GOLD / 'gold_key.json'}  (do not open while labelling)")


# --------------------------------------------------------------------------
# Self-contained annotation tool. Placeholders are replaced above; the data is
# embedded so it works offline from a double-clicked file (no server, no fetch).
# --------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Methodology Gold Labelling</title>
<style>
 *{box-sizing:border-box} body{font-family:"Segoe UI",Arial,sans-serif;margin:0;
   color:#1a1a1a;background:#f4f6f9}
 header{background:#2b3a55;color:#fff;padding:10px 16px;display:flex;
   align-items:center;gap:16px;position:sticky;top:0;z-index:5}
 header h1{font-size:15px;margin:0;font-weight:600}
 .bar{flex:1;height:8px;background:#4a5878;border-radius:5px;overflow:hidden;max-width:340px}
 .bar>div{height:100%;background:#5fd08a;width:0}
 header .stat{font-size:12px;font-variant-numeric:tabular-nums}
 button{font:inherit;cursor:pointer;border:0;border-radius:6px;padding:6px 12px}
 .btn{background:#5b7cc4;color:#fff} .btn.ghost{background:#e6ebf3;color:#223}
 .wrap{max-width:1080px;margin:14px auto;padding:0 14px;display:grid;
   grid-template-columns:1fr 320px;gap:16px}
 .card{background:#fff;border:1px solid #e2e7ee;border-radius:10px;padding:14px 16px}
 .disc{display:inline-block;background:#eef2f8;color:#3a4a66;font-size:11px;
   padding:2px 8px;border-radius:10px;margin-bottom:8px}
 .paper{white-space:pre-wrap;line-height:1.55;font-size:13px;max-height:52vh;
   overflow:auto;background:#fbfcfe;border:1px solid #eef1f6;border-radius:8px;
   padding:10px 12px}
 .paper mark{background:#fff3c4;color:inherit;padding:0 1px;border-radius:2px}
 .opts label{display:block;padding:6px 8px;border:1px solid #e2e7ee;border-radius:7px;
   margin:5px 0;font-size:13px;cursor:pointer}
 .opts label:hover{background:#f3f7fd}
 .opts input{margin-right:8px}
 .opts .kbd{color:#9aa;font-size:11px;float:right}
 textarea{width:100%;margin-top:8px;border:1px solid #dde3ec;border-radius:7px;
   padding:7px;font:inherit;font-size:12px;resize:vertical;min-height:44px}
 .nav{display:flex;gap:8px;margin-top:10px;align-items:center}
 .code h3{font-size:12px;margin:10px 0 4px;color:#2b6cb0}
 .code .d{font-size:11.5px;margin:0 0 8px;padding-bottom:7px;border-bottom:1px solid #eef1f6}
 .code .d b{font-size:12px} .code .bd{color:#667;font-style:italic}
 .code .rare{opacity:.7}
 .tag{font-size:9px;background:#ffe9c7;color:#8a5a00;border-radius:4px;padding:0 4px;margin-left:4px}
 small.hint{color:#889;font-size:11px}
</style></head><body>
<header>
  <h1>Agent 3 · Methodology Gold Labelling</h1>
  <div class="bar"><div id="pb"></div></div>
  <span class="stat" id="stat">0 / 0</span>
  <button class="btn" id="export">Export labels</button>
  <button class="btn ghost" id="jump">Next unlabelled</button>
</header>
<div class="wrap">
  <div>
    <div class="card">
      <span class="disc" id="disc"></span> <small class="hint" id="pos"></small>
      <div class="paper" id="text"></div>
      <div class="opts" id="opts"></div>
      <textarea id="notes" placeholder="optional note (why this design / what's ambiguous)"></textarea>
      <div class="nav">
        <button class="btn ghost" id="prev">&larr; Prev</button>
        <button class="btn ghost" id="next">Next &rarr;</button>
        <small class="hint">keys: 1-9 pick design · 0 unsure · &larr;/&rarr; navigate</small>
      </div>
    </div>
  </div>
  <div class="card code" id="code"><h3>Codebook — pick the paper's PRIMARY research design</h3></div>
</div>
<script>
const PAPERS=/*__PAPERS__*/; const CODEBOOK=/*__CODEBOOK__*/;
const OPTIONS=/*__OPTIONS__*/; const DSID="__DSID__"; const LS="methgold_"+DSID;
let state=JSON.parse(localStorage.getItem(LS)||"{}"); let i=0;
const $=id=>document.getElementById(id);
function save(){localStorage.setItem(LS,JSON.stringify(state));render();}
function render(){
 const p=PAPERS[i], cur=state[p.paper_id]||{};
 $("disc").textContent=p.discipline; $("pos").textContent="paper "+(i+1)+" of "+PAPERS.length+" · id "+p.paper_id;
 $("text").innerHTML=p.text_html;
 let h=""; OPTIONS.forEach((o,k)=>{const key=k<9?(k+1):(k===9?"0":"");
   h+=`<label><input type="radio" name="d" value="${o.replace(/"/g,'&quot;')}" ${cur.design===o?"checked":""}>${o}<span class="kbd">${key}</span></label>`;});
 $("opts").innerHTML=h;
 document.querySelectorAll('input[name=d]').forEach(r=>r.onchange=()=>{
   state[p.paper_id]={design:r.value,notes:$("notes").value};save();});
 $("notes").value=cur.notes||"";
 $("notes").onblur=()=>{if(state[p.paper_id]){state[p.paper_id].notes=$("notes").value;save();}};
 const done=Object.keys(state).length; $("stat").textContent=done+" / "+PAPERS.length;
 $("pb").style.width=(100*done/PAPERS.length)+"%";
}
function go(d){i=Math.max(0,Math.min(PAPERS.length-1,i+d));render();}
$("prev").onclick=()=>go(-1); $("next").onclick=()=>go(1);
$("jump").onclick=()=>{for(let k=0;k<PAPERS.length;k++){const j=(i+1+k)%PAPERS.length;
  if(!state[PAPERS[j].paper_id]){i=j;break;}}render();};
$("export").onclick=()=>{const rows=PAPERS.map(p=>({paper_id:p.paper_id,
  design:(state[p.paper_id]||{}).design||null,notes:(state[p.paper_id]||{}).notes||""}));
 const b=new Blob([JSON.stringify(rows,null,2)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);
 a.download="gold_labels.json";a.click();};
document.onkeydown=e=>{if(e.target.tagName==="TEXTAREA")return;
 if(e.key==="ArrowLeft")go(-1); else if(e.key==="ArrowRight")go(1);
 else if(/^[1-9]$/.test(e.key)){const o=OPTIONS[+e.key-1];if(o){state[PAPERS[i].paper_id]={design:o,notes:$("notes").value};save();}}
 else if(e.key==="0"){state[PAPERS[i].paper_id]={design:OPTIONS[9],notes:$("notes").value};save();}};
let ch="<h3>Codebook — pick the paper's PRIMARY research design</h3>";
CODEBOOK.forEach(d=>{ch+=`<div class="d ${d.trainable?"":"rare"}"><b>${d.name}</b>${d.trainable?"":'<span class="tag">rare</span>'}<br>${d.description}<br><span class="bd">${d.boundary}</span></div>`;});
ch+='<small class="hint"><b>Highlighted</b> phrases are method-relevant cues to help you find the key sentences fast — a paper usually has several, from different designs, so you still decide the PRIMARY one. Rare designs are kept so you can flag them (the encoder can\'t predict them). Use "Unsure" freely — it is data, not failure.</small>';
$("code").innerHTML=ch;
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
