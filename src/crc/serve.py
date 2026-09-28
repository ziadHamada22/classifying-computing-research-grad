"""Local web UI for trying the three agents interactively.

    python -m crc.serve                      # calibrated SciBERT, opens on :8000
    python -m crc.serve --ensemble           # calibrated SciBERT+DeBERTa+TF-IDF
    python -m crc.serve --model models/scibert --port 8080

The UI runs the same ``Pipeline`` as the CLI (``Pipeline.stages``), so the
discipline decision, the calibrated shortlist and the field readings a user sees
here are exactly the ones the evaluation measured. (Until 2026-09 the UI built
its own uncalibrated two-model ensemble and its own shortlist, and so could
disagree with the CLI.)

Runs entirely on localhost with no network calls and no third-party web
framework — just `http.server` — so it works from the project venv as-is.
The model is loaded once at start-up, so classification latency in the UI is the
real inference cost, not model loading.

Files are posted as base64 inside JSON rather than as multipart form data: it
avoids a deprecated stdlib parser and keeps the request handling small.
"""
from __future__ import annotations

import argparse
import base64
import json
import tempfile
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from crc.ingest import IngestError

PROJECT = Path(__file__).resolve().parents[2]
MODELS = PROJECT / "models"

# Populated in main(); the handler reads them.
PIPELINE = None
MODEL_LABEL = ""


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Computing Research Classifier</title>
<style>
  /* "Ink & iris": warm stone neutrals, one colour per agent --
     indigo (discipline), coral (field), mint (research design). */
  :root {
    --paper:#f7f5f1; --panel:#ffffff; --panel-2:#efebe4;
    --ink:#1d1b29; --ink-soft:#4e4b5e; --ink-faint:#7c788c;
    --line:#e4dfd6;
    --accent:#4f46d8; --accent-soft:#4f46d814;
    --a1:#4f46d8; --a2:#cc4f36; --a3:#0c8a73;
    --good:#1e8a57; --warn:#a95a0c; --crit:#c0362c;
    --warn-soft:#a95a0c12; --good-soft:#1e8a5712;
    --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper:#12111c; --panel:#1a1928; --panel-2:#242238;
      --ink:#eceaf6; --ink-soft:#b3b0c8; --ink-faint:#7f7c97;
      --line:#2e2b45;
      --accent:#8e86ff; --accent-soft:#8e86ff22;
      --a1:#8e86ff; --a2:#ff8b6f; --a3:#3fd1b2;
      --good:#5fd08f; --warn:#f0a24b; --crit:#ff7b72;
      --warn-soft:#f0a24b16; --good-soft:#5fd08f16;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
       line-height:1.5;-webkit-font-smoothing:antialiased}
  .wrap{max-width:940px;margin:0 auto;padding:28px 20px 80px}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;
           text-transform:uppercase;color:var(--accent);font-weight:600}
  h1{font-size:26px;margin:8px 0 4px;letter-spacing:-.02em}
  .sub{color:var(--ink-soft);font-size:14px;margin:0 0 22px}
  .sub code{font-family:var(--mono);font-size:12.5px;background:var(--panel-2);
            padding:1px 5px;border-radius:4px}
  textarea{width:100%;min-height:170px;padding:14px;border-radius:10px;
           border:1px solid var(--line);background:var(--panel);color:var(--ink);
           font-family:var(--sans);font-size:14px;resize:vertical}
  textarea:focus{outline:2px solid var(--accent);outline-offset:1px}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
  button{font-family:var(--sans);font-size:14px;font-weight:600;padding:10px 20px;
         border-radius:9px;border:1px solid transparent;background:var(--accent);
         color:#fff;cursor:pointer}
  button:hover{filter:brightness(1.08)}
  button:disabled{opacity:.55;cursor:default}
  button.ghost{background:transparent;color:var(--ink-soft);border-color:var(--line)}
  .drop{margin-top:12px;border:1.5px dashed var(--line);border-radius:10px;
        padding:16px;text-align:center;color:var(--ink-faint);font-size:13px;
        background:var(--panel)}
  .drop.over{border-color:var(--accent);background:var(--accent-soft);color:var(--accent)}
  .samples{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
  .chip{font-size:12px;padding:5px 11px;border-radius:999px;border:1px solid var(--line);
        background:var(--panel);color:var(--ink-soft);cursor:pointer}
  .chip:hover{border-color:var(--accent);color:var(--accent)}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
        padding:20px;margin-top:18px;box-shadow:0 1px 2px #1d1b290a}
  .card.a1{border-top:4px solid var(--a1)} .card.a1 .conf{color:var(--a1)}
  .card.a2{border-top:4px solid var(--a2)} .card.a2 .conf{color:var(--a2)}
  .card.a3{border-top:4px solid var(--a3)} .card.a3 .conf{color:var(--a3)}
  .card.a1 .bfill{background:var(--a1)} .card.a2 .bfill{background:var(--a2)}
  .card.a3 .bfill{background:var(--a3)}
  .card.a1>.meta:first-child{color:var(--a1);font-weight:600}
  .card.a2>.meta:first-child{color:var(--a2);font-weight:600}
  .card.a3>.meta:first-child{color:var(--a3);font-weight:600}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin:-12px 0 20px;font-size:12.5px;
          color:var(--ink-soft)}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .legend i{width:10px;height:10px;border-radius:3px;display:inline-block}
  .verdict{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
  .verdict .lab{font-size:26px;font-weight:700;letter-spacing:-.02em}
  .verdict .conf{font-family:var(--mono);font-size:22px;color:var(--accent);
                 font-variant-numeric:tabular-nums}
  .meta{margin-top:8px;font-size:12.5px;color:var(--ink-faint);font-family:var(--mono)}
  .bars{margin-top:18px;display:grid;gap:9px}
  .brow{display:grid;grid-template-columns:180px 1fr 60px;align-items:center;gap:11px}
  @media(max-width:620px){.brow{grid-template-columns:120px 1fr 52px}}
  .bl{font-size:13px}
  .btrack{height:10px;background:var(--panel-2);border-radius:999px;overflow:hidden}
  .bfill{height:100%;border-radius:999px;background:var(--accent);
         transition:width .45s cubic-bezier(.22,1,.36,1)}
  .bv{font-family:var(--mono);font-size:12.5px;text-align:right;
      font-variant-numeric:tabular-nums;color:var(--ink-soft)}
  .flag{margin-top:16px;padding:12px 14px;border-radius:9px;font-size:13px;
        border:1px solid;display:block;line-height:1.55}
  .flag b{margin-right:4px}
  a{color:var(--accent)}
  .flag.b{border-color:var(--warn);background:var(--warn-soft);color:var(--warn)}
  .flag.ok{border-color:var(--good);background:var(--good-soft);color:var(--good)}
  details{margin-top:16px}
  summary{cursor:pointer;font-size:13px;color:var(--ink-soft);font-family:var(--mono)}
  table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:10px}
  th,td{text-align:left;padding:7px 9px;border-top:1px solid var(--line)}
  th{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
     letter-spacing:.05em;color:var(--ink-faint)}
  td.n{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
  .err{color:var(--crit);font-size:13px;margin-top:12px}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid #fff6;
        border-top-color:#fff;border-radius:50%;animation:s .7s linear infinite;
        vertical-align:-2px;margin-right:7px}
  @keyframes s{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Computing Research Classifier</div>
  <h1>Discipline, field and research design</h1>
  <p class="sub">Paste an abstract, an article, or drop a PDF/DOCX. Model: <code id="ml">—</code></p>
  <div class="legend">
    <span><i style="background:var(--a1)"></i>Agent 1 · discipline</span>
    <span><i style="background:var(--a2)"></i>Agent 2 · field</span>
    <span><i style="background:var(--a3)"></i>Agent 3 · research design</span>
  </div>

  <textarea id="t" placeholder="Paste a title and abstract, a full paper, or any text…"></textarea>

  <div class="samples" id="samples"></div>

  <div class="row">
    <button id="go">Classify</button>
    <button class="ghost" id="clear">Clear</button>
    <span id="stat" class="meta"></span>
  </div>

  <div class="drop" id="drop">Drop a PDF or DOCX here, or click to choose a file
    <input type="file" id="file" accept=".pdf,.docx,.txt,.md" hidden>
  </div>

  <div id="out"></div>
</div>

<script>
const SAMPLES = {
  "Transformer (CS/DS?)": "Attention Is All You Need. The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
  "Flaky tests (SE)": "An Empirical Study of Flaky Tests in Continuous Integration. Flaky tests non-deterministically pass or fail on the same code, eroding developer trust in CI pipelines. We analyse 200 open-source projects and propose a static analysis that identifies order-dependent tests.",
  "FPGA accelerator (CE)": "A Systolic Array Accelerator for CNN Inference on FPGA. We present a register-transfer-level design that optimises dataflow and on-chip SRAM allocation for convolutional neural network inference, achieving 3.2x better energy efficiency than a GPU baseline.",
  "Query optimiser (IS)": "Learned Cardinality Estimation for Distributed Query Optimisation. We present a query optimiser for distributed relational databases that reduces join cost using a learned cardinality estimator over the data warehouse schema.",
  "Zero-trust networking (IT)": "Deploying Zero-Trust Microsegmentation in Enterprise Networks. We report on operational experience migrating a 12,000-host campus network to a zero-trust architecture, measuring latency overhead and incident response times over 18 months.",
  "Causal inference (DS)": "Doubly Robust Estimation of Treatment Effects under Covariate Shift. We derive a doubly robust estimator for average treatment effects when the covariate distribution differs between study and target populations, and establish its asymptotic normality."
};
const $ = s => document.querySelector(s);
const sBox = $("#samples");
for (const [k, v] of Object.entries(SAMPLES)) {
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = k;
  b.onclick = () => { $("#t").value = v; };
  sBox.appendChild(b);
}
fetch("/meta").then(r => r.json()).then(d => { $("#ml").textContent = d.model; });

const drop = $("#drop"), fileIn = $("#file");
drop.onclick = () => fileIn.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files[0]) sendFile(e.dataTransfer.files[0]); };
fileIn.onchange = () => { if (fileIn.files[0]) sendFile(fileIn.files[0]); };

function sendFile(f) {
  const rd = new FileReader();
  rd.onload = () => post({ filename: f.name, data: rd.result.split(",")[1] });
  rd.readAsDataURL(f);
}
$("#go").onclick = () => {
  const v = $("#t").value.trim();
  if (v) post({ text: v });
};
$("#clear").onclick = () => { $("#t").value = ""; $("#out").innerHTML = ""; $("#stat").textContent = ""; };

function post(body) {
  $("#go").disabled = true;
  $("#stat").innerHTML = '<span class="spin"></span>classifying…';
  $("#out").innerHTML = "";
  const t0 = performance.now();
  fetch("/classify", { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(body) })
    .then(r => r.json())
    .then(d => {
      $("#go").disabled = false;
      $("#stat").textContent = "round trip " + Math.round(performance.now() - t0) + " ms";
      if (d.error) { $("#out").innerHTML = '<div class="err">' + esc(d.error) + '</div>'; return; }
      render(d);
    })
    .catch(e => { $("#go").disabled = false; $("#stat").textContent = "";
      $("#out").innerHTML = '<div class="err">' + esc(String(e)) + '</div>'; });
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function render(d) {
  const ranked = Object.entries(d.probs).sort((a, b) => b[1] - a[1]);
  let bars = "";
  for (const [name, p] of ranked) {
    bars += '<div class="brow"><div class="bl">' + esc(name) + '</div>' +
      '<div class="btrack"><div class="bfill" style="width:0%" data-w="' + (p * 100) + '"></div></div>' +
      '<div class="bv">' + (p * 100).toFixed(1) + '%</div></div>';
  }
  const flag = d.borderline
    ? '<div class="flag b"><b>Contested.</b> ' + esc(d.borderline_reason || "") +
      ' — treat the discipline as uncertain; the field is also read under the other candidates below.</div>'
    : '<div class="flag ok"><b>Confident.</b> ' + (d.conformal_alpha
        ? 'Only ' + esc(d.label) + ' survives the calibrated check at ' +
          Math.round((1 - d.conformal_alpha) * 100) + '% confidence.'
        : 'No other discipline is plausible at the calibrated confidence level.') + '</div>';

  let chunks = "";
  if (d.chunks && d.chunks.length) {
    chunks = '<details><summary>per-chunk evidence (' + d.chunks.length + ' chunks)</summary>' +
      '<table><tr><th>#</th><th>section</th><th>words</th><th>weight</th><th>prediction</th><th>conf</th><th>text</th></tr>';
    for (const c of d.chunks.slice(0, 40)) {
      chunks += '<tr><td class="n">' + c.index + '</td><td>' + esc(c.section) + '</td>' +
        '<td class="n">' + c.n_words + '</td><td class="n">' + c.weight.toFixed(2) + '</td>' +
        '<td>' + esc(c.label) + '</td><td class="n">' + (c.confidence * 100).toFixed(0) + '%</td>' +
        '<td style="color:var(--ink-faint)">' + esc(c.text_preview.slice(0, 60)) + '…</td></tr>';
    }
    chunks += '</table></details>';
  }
  let sect = "";
  if (d.section_mass && Object.keys(d.section_mass).length) {
    sect = '<details><summary>evidence weight by section</summary><table>';
    for (const [k, v] of Object.entries(d.section_mass).sort((a, b) => b[1] - a[1]))
      sect += '<tr><td>' + esc(k) + '</td><td class="n">' + (v * 100).toFixed(1) + '%</td></tr>';
    sect += '</table></details>';
  }

  // Agent 2 — the field within the predicted discipline.
  let fieldBlock = "";
  if (d.field) {
    const f = d.field;
    const franked = Object.entries(f.probs).sort((a, b) => b[1] - a[1]);
    let fbars = "";
    for (const [name, p] of franked) {
      fbars += '<div class="brow"><div class="bl">' + esc(name) + '</div>' +
        '<div class="btrack"><div class="bfill" style="width:0%" data-w="' + (p * 100) + '"></div></div>' +
        '<div class="bv">' + (p * 100).toFixed(1) + '%</div></div>';
    }
    let alt = "";
    const alist = d.field_alternatives || (d.field_alternative ? [d.field_alternative] : []);
    if (alist.length) {
      const rows = alist.map(function (a) {
        return esc(a.discipline) + ' &rarr; <b>' + esc(a.field) + '</b> (' +
          (a.confidence * 100).toFixed(1) + '%)';
      }).join('<br>');
      const lvl = d.shortlist_alpha
        ? 'At the stricter ' + Math.round((1 - d.shortlist_alpha) * 100) +
          '% level used for the field shortlist'
        : 'At the calibrated shortlist level';
      alt = '<div class="flag b" style="margin-top:12px"><b>Alternative reading' +
        (alist.length > 1 ? 's' : '') + '.</b> ' + lvl + ', these disciplines' +
        ' cannot be ruled out either, so the field is also read under each of them:<br>' +
        rows + '<br>Treat all as candidates.</div>';
    }
    fieldBlock = '<div class="card a2"><div class="meta" style="margin:0 0 4px">' +
      'AGENT 2 · field within ' + esc(d.label) + '</div>' +
      '<div class="verdict"><span class="lab">' + esc(f.label) + '</span>' +
      '<span class="conf">' +
      (f.confidence * 100).toFixed(1) + '%</span></div>' +
      '<div class="meta">runner-up ' + esc(f.runner_up) + ' · gap ' +
      f.gap.toFixed(3) + ' · ' + (d.timing_ms.field || 0) + ' ms</div>' +
      '<div class="bars">' + fbars + '</div>' + alt +
      (f.borderline ? '<div class="flag b">' + esc(f.borderline_reason || "") + '</div>' : "") +
      '</div>';
  }

  let methBlock = "";
  if (d.methodology && d.methodology.n_blocks > 0) {
    const m = d.methodology;
    let mbars = "";
    const mr = Object.entries(m.probs).sort((a, b) => b[1] - a[1]);
    for (const [name, p] of mr) {
      mbars += '<div class="brow"><div class="bl">' + esc(name) + '</div>' +
        '<div class="btrack"><div class="bfill" style="width:0%" data-w="' + (p * 100) + '"></div></div>' +
        '<div class="bv">' + (p * 100).toFixed(1) + '%</div></div>';
    }
    const fac = (m.facets_present || []).length
      ? m.facets_present.map(esc).join(", ") : "none above threshold";
    const how = m.design_source === "facets"
      ? "derived from the facets by the rule: " + esc(m.derived_rule || "")
      : "chosen by the no-evidence fallback: " + esc(m.derived_rule || "");
    let sim = "";
    for (const s of (m.similar_papers || [])) {
      sim += '<div class="meta" style="margin:2px 0">' + s.similarity.toFixed(2) + ' · ' +
        '<a href="https://arxiv.org/abs/' + encodeURIComponent(s.paper_id) +
        '" target="_blank" rel="noopener">' + esc(s.title || s.paper_id) + '</a> → ' +
        esc(s.design) + '</div>';
    }
    const scope = '<div class="meta" style="margin-top:10px">facets observed: ' + fac +
      '</div><div class="meta">design ' + how + '. Worldview and method are looked up ' +
      'from the design, not predicted.</div>' +
      (sim ? '<div class="meta" style="margin-top:10px"><b>Most similar reference papers</b></div>' + sim : "");
    methBlock = '<div class="card a3"><div class="meta" style="margin:0 0 4px">' +
      'AGENT 3 · research design' +
      (m.read_from && m.read_from !== "abstract" ? ' · read from ' + esc(m.read_from) : "") +
      '</div><div class="verdict"><span class="lab">' + esc(m.design) + '</span>' +
      '<span class="conf">' + (m.confidence * 100).toFixed(1) + '%</span></div>' +
      '<div class="meta">worldview ' + esc(m.worldview) + ' · method ' + esc(m.method) +
      ' · ' + ((d.timing_ms && d.timing_ms.methodology) || 0) + ' ms</div>' +
      '<div class="bars">' + mbars + '</div>' + scope +
      (m.borderline ? '<div class="flag b">' + esc(m.borderline_reason || "") + '</div>' : "") +
      '</div>';
  }

  let warn = "";
  for (const w of (d.warnings || [])) {
    warn += '<div class="flag b" style="margin:0 0 12px"><b>Warning.</b> ' + esc(w) + '</div>';
  }
  if (d.title) {
    warn += '<div class="meta" style="margin:0 0 12px">' + esc(d.title) + '</div>';
  }

  $("#out").innerHTML = warn + '<div class="card a1">' +
    '<div class="meta" style="margin:0 0 4px">AGENT 1 · discipline</div>' +
    '<div class="verdict"><span class="lab">' + esc(d.label) + '</span>' +
    '<span class="conf">' + (d.confidence * 100).toFixed(1) + '%</span></div>' +
    '<div class="meta">runner-up ' + esc(d.runner_up) + ' · gap ' + d.gap.toFixed(3) +
    ' · ' + esc(d.doc_type) + ' · ' + d.n_chunks + ' chunk' + (d.n_chunks === 1 ? "" : "s") +
    ' · parser ' + esc(d.parser) + ' · ' + d.timing_ms.total + ' ms</div>' +
    '<div class="bars">' + bars + '</div>' + flag + sect + chunks + '</div>' +
    fieldBlock + methBlock;

  requestAnimationFrame(() => document.querySelectorAll(".bfill")
    .forEach(e => e.style.width = e.dataset.w + "%"));
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/meta":
            self._json(200, {"model": MODEL_LABEL})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/classify":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return

        tmp: Path | None = None
        try:
            t0 = time.perf_counter()
            if payload.get("data"):
                suffix = Path(payload.get("filename", "upload")).suffix or ".bin"
                raw = base64.b64decode(payload["data"])
                fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                fd.write(raw)
                fd.close()
                tmp = Path(fd.name)
                source = str(tmp)
            elif payload.get("text", "").strip():
                source = payload["text"].strip()
            else:
                self._json(400, {"error": "send either `text` or `data`"})
                return

            from crc.ingest import ingest as _ingest

            doc = _ingest(source)
            st = PIPELINE.stages(doc)
            result = st["discipline"]
            elapsed = (time.perf_counter() - t0) * 1000
            out = json.loads(result.to_json())
            out["timing_ms"] = {"total": round(elapsed, 1), **st["timing_ms"]}

            # Agent 2, reported under the predicted discipline plus every other
            # discipline in the calibrated shortlist, so a fork in the cascade is
            # visible rather than hidden. Conditioning is a mask, so all of them
            # cost one encoder pass.
            if st["fields"]:
                cands, readings = st["candidates"], st["fields"]
                out["field"] = json.loads(readings[0].to_json())
                alts = [{"discipline": name, "field": r.label,
                         "confidence": r.confidence}
                        for name, r in zip(cands[1:], readings[1:])]
                if alts:
                    out["field_alternatives"] = alts
                    out["field_alternative"] = alts[0]   # back-compat

            if st["methodology"] is not None:
                out["methodology"] = json.loads(st["methodology"].to_json())

            if payload.get("filename"):
                out["source"] = payload["filename"]
            out["title"] = st.get("title")
            out["warnings"] = st.get("warnings", [])
            out["shortlist_alpha"] = (PIPELINE.shortlist.alpha
                                      if PIPELINE.shortlist is not None else None)
            self._json(200, out)
        except (FileNotFoundError, IngestError) as exc:
            # Unreadable input (scanned PDF, empty text, unsupported type): say
            # why, rather than classify nothing.
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def do_HEAD(self):  # noqa: N802
        # Health checks and link previews probe with HEAD; answer like GET, no body.
        self.send_response(200 if self.path in ("/", "/index.html") else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def log_message(self, fmt, *args):
        # One tidy line per request instead of the noisy default. ``args[0]`` is
        # the request line for normal requests but an HTTPStatus when the server
        # rejects a malformed one, so it is stringified before inspection.
        first = str(args[0]) if args else ""
        if "POST" in first:
            print(f"  · {first}", flush=True)


def main() -> None:
    global PIPELINE, MODEL_LABEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Discipline model dir (default: models/scibert).")
    ap.add_argument("--ensemble", action="store_true",
                    help="Use the calibrated SciBERT + DeBERTa + TF-IDF ensemble.")
    ap.add_argument("--single", action="store_true",
                    help="Accepted for compatibility; one model is now the default.")
    ap.add_argument("--no-field", action="store_true",
                    help="Skip Agent 2 even if it is trained.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    from crc.pipeline import Pipeline

    print("loading models...", flush=True)
    t0 = time.perf_counter()
    PIPELINE = Pipeline.load(discipline_dir=args.model, ensemble=args.ensemble)
    if args.no_field:
        PIPELINE.field_clf = None
    disc = PIPELINE.discipline_clf
    MODEL_LABEL = getattr(disc, "name", "discipline")
    if PIPELINE.field_clf is not None:
        MODEL_LABEL += "  +  " + PIPELINE.field_clf.name
    if PIPELINE.methodology_clf is not None:
        MODEL_LABEL += "  +  " + PIPELINE.methodology_clf.name
    if PIPELINE.shortlist is not None:
        MODEL_LABEL += f"  (calibrated shortlist, alpha={PIPELINE.shortlist.alpha:g})"
    print(f"loaded {MODEL_LABEL} in {time.perf_counter()-t0:.1f}s", flush=True)

    url = f"http://localhost:{args.port}/"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  Agent 1 live at {url}\n  Ctrl-C to stop\n", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
