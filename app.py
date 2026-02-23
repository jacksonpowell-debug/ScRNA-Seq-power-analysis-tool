"""
scRNA-Seq Analysis Pipeline — Web Interface
============================================
Run with:
    pip install flask
    python app.py

Then open http://localhost:5050 in your browser.

Place this file in the same directory as scRNAseq_pipeline.py
"""

import os
import sys
import json
import queue
import threading
import traceback
import numpy as np

from flask import Flask, Response, render_template_string, request, jsonify, send_file

# ── Bootstrap: add script dir to path so pipeline module is importable ────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

app = Flask(__name__)

# Global job state
_job_queue  = queue.Queue()
_job_lock   = threading.Lock()
_job_active = False
OUTPUT_DIR  = SCRIPT_DIR   # images saved here

# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>scRNA-Seq Analysis Pipeline</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #080c10;
    --surface:   #0e1520;
    --border:    #1c2a3a;
    --accent:    #00e5b0;
    --accent2:   #0088ff;
    --warn:      #ff6b35;
    --text:      #c9d8e8;
    --muted:     #4a6070;
    --heading:   #ffffff;
    --mono:      'Space Mono', monospace;
    --sans:      'Syne', sans-serif;
    --radius:    6px;
    --transition: 200ms cubic-bezier(.4,0,.2,1);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Background grid ─────────────────────────────────────────────────── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,229,176,.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,176,.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── Layout ──────────────────────────────────────────────────────────── */
  .app {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: 420px 1fr;
    grid-template-rows: auto 1fr;
    min-height: 100vh;
  }

  /* ── Header ──────────────────────────────────────────────────────────── */
  header {
    grid-column: 1 / -1;
    padding: 24px 36px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 20px;
    background: rgba(8,12,16,.8);
    backdrop-filter: blur(12px);
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .logo {
    width: 40px;
    height: 40px;
    flex-shrink: 0;
  }

  .logo-helix {
    animation: spin 12s linear infinite;
    transform-origin: 20px 20px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .header-text h1 {
    font-family: var(--sans);
    font-weight: 800;
    font-size: 18px;
    color: var(--heading);
    letter-spacing: -.02em;
  }

  .header-text p {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
  }

  .header-badge {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--accent);
    border: 1px solid rgba(0,229,176,.3);
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: .08em;
  }

  /* ── Left panel ──────────────────────────────────────────────────────── */
  .panel {
    border-right: 1px solid var(--border);
    padding: 28px 28px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 28px;
    background: rgba(14,21,32,.6);
  }

  /* ── Section card ────────────────────────────────────────────────────── */
  .card {
    background: rgba(255,255,255,.02);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: border-color var(--transition);
  }

  .card:focus-within {
    border-color: rgba(0,229,176,.35);
  }

  .card-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(0,0,0,.2);
  }

  .card-num {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 1px solid var(--muted);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    flex-shrink: 0;
    transition: all var(--transition);
  }

  .card.active .card-num {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(0,229,176,.1);
  }

  .card-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--heading);
  }

  .card-body { padding: 16px; }

  /* ── Form elements ───────────────────────────────────────────────────── */
  label {
    display: block;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 6px;
  }

  input[type="text"], select, textarea {
    width: 100%;
    background: rgba(0,0,0,.4);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 9px 12px;
    outline: none;
    transition: border-color var(--transition);
    appearance: none;
  }

  input[type="text"]:focus, select:focus, textarea:focus {
    border-color: rgba(0,229,176,.5);
  }

  input[type="text"]::placeholder { color: var(--muted); }

  select option { background: #0e1520; }

  .field + .field { margin-top: 14px; }

  .field-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .hint {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 5px;
    line-height: 1.4;
  }

  /* ── Radio / toggle row ──────────────────────────────────────────────── */
  .toggle-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .toggle-btn {
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    transition: all var(--transition);
  }

  .toggle-btn.active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(0,229,176,.08);
  }

  /* ── Checkbox ────────────────────────────────────────────────────────── */
  .check-row {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    padding: 8px 0;
  }

  .check-row input[type="checkbox"] {
    appearance: none;
    width: 16px;
    height: 16px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: transparent;
    cursor: pointer;
    flex-shrink: 0;
    transition: all var(--transition);
    position: relative;
  }

  .check-row input[type="checkbox"]:checked {
    background: var(--accent);
    border-color: var(--accent);
  }

  .check-row input[type="checkbox"]:checked::after {
    content: '✓';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 10px;
    color: #000;
    font-weight: 700;
  }

  .check-label {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
  }

  /* ── Tags input (genes) ──────────────────────────────────────────────── */
  .tags-wrap {
    min-height: 44px;
    background: rgba(0,0,0,.4);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 6px 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    cursor: text;
    transition: border-color var(--transition);
  }

  .tags-wrap:focus-within { border-color: rgba(0,229,176,.5); }

  .tag {
    background: rgba(0,229,176,.12);
    border: 1px solid rgba(0,229,176,.3);
    border-radius: 4px;
    color: var(--accent);
    font-family: var(--mono);
    font-size: 11px;
    padding: 2px 8px 2px 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .tag-x {
    cursor: pointer;
    opacity: .6;
    transition: opacity var(--transition);
    font-size: 13px;
    line-height: 1;
  }

  .tag-x:hover { opacity: 1; }

  #gene-input {
    border: none;
    background: transparent;
    outline: none;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    flex: 1;
    min-width: 100px;
    padding: 2px 4px;
  }

  /* ── Run button ──────────────────────────────────────────────────────── */
  .run-btn {
    width: 100%;
    padding: 14px;
    background: var(--accent);
    color: #000;
    font-family: var(--sans);
    font-weight: 700;
    font-size: 14px;
    letter-spacing: .06em;
    text-transform: uppercase;
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    transition: all var(--transition);
    position: relative;
    overflow: hidden;
  }

  .run-btn:hover { background: #00ffca; transform: translateY(-1px); }
  .run-btn:active { transform: translateY(0); }
  .run-btn:disabled {
    background: var(--border);
    color: var(--muted);
    cursor: not-allowed;
    transform: none;
  }

  .run-btn .spinner {
    display: none;
    width: 16px;
    height: 16px;
    border: 2px solid rgba(0,0,0,.3);
    border-top-color: #000;
    border-radius: 50%;
    animation: rotate .7s linear infinite;
    margin: 0 auto;
  }

  @keyframes rotate { to { transform: rotate(360deg); } }

  /* ── Right panel: output ─────────────────────────────────────────────── */
  .output-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .output-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(8,12,16,.6);
    flex-shrink: 0;
  }

  .output-title {
    font-family: var(--sans);
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--heading);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
    transition: all var(--transition);
  }

  .status-dot.running {
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 1.2s ease-in-out infinite;
  }

  .status-dot.done { background: var(--accent); }
  .status-dot.error { background: var(--warn); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: .4; }
  }

  .status-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-left: auto;
  }

  .output-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.8;
    color: var(--text);
  }

  .output-scroll::-webkit-scrollbar { width: 6px; }
  .output-scroll::-webkit-scrollbar-track { background: transparent; }
  .output-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .log-line { display: block; white-space: pre-wrap; word-break: break-word; }
  .log-line.info  { color: var(--text); }
  .log-line.accent { color: var(--accent); }
  .log-line.warn  { color: var(--warn); }
  .log-line.muted { color: var(--muted); }
  .log-line.head  { color: var(--heading); font-weight: 700; margin-top: 8px; }
  .log-line.r     { color: #7ec8e3; }

  .placeholder-msg {
    color: var(--muted);
    font-family: var(--mono);
    font-size: 12px;
    text-align: center;
    margin-top: 80px;
    line-height: 2;
  }

  /* ── Results gallery ─────────────────────────────────────────────────── */
  .results-section {
    border-top: 1px solid var(--border);
    padding: 20px 24px;
    background: rgba(0,0,0,.2);
    flex-shrink: 0;
  }

  .results-title {
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .results-grid {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }

  .result-thumb {
    width: 90px;
    height: 70px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid var(--border);
    cursor: pointer;
    transition: all var(--transition);
    background: var(--surface);
  }

  .result-thumb:hover {
    border-color: var(--accent);
    transform: scale(1.05);
  }

  /* ── Lightbox ────────────────────────────────────────────────────────── */
  .lightbox {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.85);
    z-index: 999;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(8px);
  }

  .lightbox.open { display: flex; }

  .lightbox img {
    max-width: 90vw;
    max-height: 90vh;
    border-radius: 6px;
    box-shadow: 0 40px 80px rgba(0,0,0,.6);
  }

  .lightbox-close {
    position: absolute;
    top: 20px;
    right: 24px;
    color: white;
    font-size: 28px;
    cursor: pointer;
    opacity: .7;
    transition: opacity var(--transition);
    background: none;
    border: none;
    line-height: 1;
  }

  .lightbox-close:hover { opacity: 1; }

  /* ── Responsive ──────────────────────────────────────────────────────── */
  @media (max-width: 900px) {
    .app { grid-template-columns: 1fr; grid-template-rows: auto auto 1fr; }
    .panel { border-right: none; border-bottom: 1px solid var(--border); }
    .output-panel { min-height: 60vh; }
  }

  /* ── Utility ─────────────────────────────────────────────────────────── */
  .hidden { display: none !important; }
  .mt8 { margin-top: 8px; }
  .mt14 { margin-top: 14px; }
</style>
</head>
<body>
<div class="app">

  <!-- ── Header ────────────────────────────────────────────────────────── -->
  <header>
    <svg class="logo" viewBox="0 0 40 40" fill="none">
      <g class="logo-helix">
        <circle cx="20" cy="4"  r="3" fill="#00e5b0" opacity=".9"/>
        <circle cx="28" cy="10" r="2.5" fill="#00e5b0" opacity=".7"/>
        <circle cx="30" cy="18" r="2" fill="#00e5b0" opacity=".5"/>
        <circle cx="28" cy="26" r="2.5" fill="#00e5b0" opacity=".7"/>
        <circle cx="20" cy="32" r="3" fill="#00e5b0" opacity=".9"/>
        <circle cx="12" cy="26" r="2.5" fill="#0088ff" opacity=".7"/>
        <circle cx="10" cy="18" r="2" fill="#0088ff" opacity=".5"/>
        <circle cx="12" cy="10" r="2.5" fill="#0088ff" opacity=".7"/>
        <line x1="20" y1="4" x2="28" y2="10" stroke="#00e5b0" stroke-width="1" opacity=".4"/>
        <line x1="28" y1="10" x2="30" y2="18" stroke="#00e5b0" stroke-width="1" opacity=".4"/>
        <line x1="20" y1="4" x2="12" y2="10" stroke="#0088ff" stroke-width="1" opacity=".4"/>
        <line x1="12" y1="10" x2="10" y2="18" stroke="#0088ff" stroke-width="1" opacity=".4"/>
      </g>
      <circle cx="20" cy="18" r="4" fill="none" stroke="#00e5b0" stroke-width="1.5" opacity=".6"/>
    </svg>
    <div class="header-text">
      <h1>scRNA-Seq Analysis Pipeline</h1>
      <p>Single-cell RNA sequencing · QC · ZINB Power Analysis</p>
    </div>
    <span class="header-badge">v2.0</span>
  </header>

  <!-- ── Left panel ─────────────────────────────────────────────────────── -->
  <aside class="panel">

    <!-- Step 1: Input -->
    <div class="card active" id="card-1">
      <div class="card-header">
        <div class="card-num">1</div>
        <span class="card-title">Data Input</span>
      </div>
      <div class="card-body">
        <div class="field">
          <label>File Type</label>
          <div class="toggle-row" id="filetype-toggle">
            <button class="toggle-btn active" data-val="h5ad" onclick="setFiletype('h5ad',this)">H5AD</button>
            <button class="toggle-btn" data-val="mtx" onclick="setFiletype('mtx',this)">10x MTX</button>
          </div>
        </div>
        <div class="field mt8">
          <label>File / Directory Path</label>
          <input type="text" id="input-path" placeholder="/path/to/file.h5ad or /path/to/mtx_dir">
          <p class="hint">Paste the full path to your .h5ad file or 10x MTX directory</p>
        </div>
        <!-- MTX-only: combine datasets -->
        <div class="field mt8 hidden" id="mtx-extra">
          <label class="check-row" style="text-transform:none;letter-spacing:0;">
            <input type="checkbox" id="combine-check" onchange="toggleCombine()">
            <span class="check-label">Combine two MTX datasets</span>
          </label>
          <div class="hidden mt8" id="path2-field">
            <label>Second Dataset Path</label>
            <input type="text" id="input-path2" placeholder="/path/to/second_mtx_dir or specific file">
          </div>
        </div>
      </div>
    </div>

    <!-- Step 2: Cluster -->
    <div class="card" id="card-2">
      <div class="card-header">
        <div class="card-num">2</div>
        <span class="card-title">Cell Type Selection</span>
      </div>
      <div class="card-body">
        <div class="field">
          <label>Annotation Column</label>
          <select id="clust-col" onchange="loadClusterValues()">
            <option value="">— load file first —</option>
          </select>
        </div>
        <div class="field mt8 hidden" id="clust-values-field">
          <label>Cell Types to Analyse</label>
          <select id="clust-values" multiple size="7" style="height:auto;">
          </select>
          <p class="hint">Hold ⌘/Ctrl to select multiple. Leave all selected to include all.</p>
        </div>
      </div>
    </div>

    <!-- Step 3: Genes -->
    <div class="card" id="card-3">
      <div class="card-header">
        <div class="card-num">3</div>
        <span class="card-title">Gene Selection</span>
      </div>
      <div class="card-body">
        <div class="field">
          <label>Gene Names</label>
          <div class="tags-wrap" onclick="document.getElementById('gene-input').focus()">
            <div id="gene-tags"></div>
            <input id="gene-input" type="text" placeholder="Type gene name + Enter">
          </div>
          <p class="hint">Press Enter or comma after each gene name</p>
        </div>
      </div>
    </div>

    <!-- Step 4: Settings -->
    <div class="card" id="card-4">
      <div class="card-header">
        <div class="card-num">4</div>
        <span class="card-title">Analysis Settings</span>
      </div>
      <div class="card-body">
        <div class="field-row">
          <div class="field">
            <label>Simulations</label>
            <input type="text" id="nsim" value="1000">
          </div>
          <div class="field">
            <label>Power Target</label>
            <input type="text" id="power-target" value="0.95">
          </div>
        </div>
        <div class="field mt8">
          <label>MTX Doublet UMI Max <span style="opacity:.5">(0 = skip)</span></label>
          <input type="text" id="doublet-max" value="0" placeholder="0 to skip">
        </div>
        <div class="field mt8">
          <label>Output Directory</label>
          <input type="text" id="out-dir" placeholder="Same as input (default)">
        </div>
      </div>
    </div>

    <!-- Run -->
    <button class="run-btn" id="run-btn" onclick="runPipeline()">
      <span id="run-label">Run Analysis</span>
      <div class="spinner" id="run-spinner"></div>
    </button>

  </aside>

  <!-- ── Right panel: output ────────────────────────────────────────────── -->
  <div class="output-panel">
    <div class="output-header">
      <div class="status-dot" id="status-dot"></div>
      <span class="output-title">Live Output</span>
      <span class="status-label" id="status-label">Idle</span>
    </div>
    <div class="output-scroll" id="output">
      <p class="placeholder-msg">
        Configure your analysis on the left,<br>
        then click <strong style="color:var(--accent)">Run Analysis</strong> to begin.<br><br>
        Output will stream here in real time.
      </p>
    </div>
    <div class="results-section hidden" id="results-section">
      <div class="results-title">Output Files — click to enlarge</div>
      <div class="results-grid" id="results-grid"></div>
    </div>
  </div>

</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <button class="lightbox-close" onclick="closeLightbox()">×</button>
  <img id="lightbox-img" src="" alt="">
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let genes   = [];
let filetype = 'h5ad';
let es       = null;   // EventSource

// ── File type toggle ───────────────────────────────────────────────────────
function setFiletype(val, btn) {
  filetype = val;
  document.querySelectorAll('#filetype-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('mtx-extra').classList.toggle('hidden', val !== 'mtx');
}

function toggleCombine() {
  const checked = document.getElementById('combine-check').checked;
  document.getElementById('path2-field').classList.toggle('hidden', !checked);
}

// ── Gene tag input ─────────────────────────────────────────────────────────
document.getElementById('gene-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ',') {
    e.preventDefault();
    addGene(e.target.value.trim().replace(/,$/, ''));
    e.target.value = '';
  } else if (e.key === 'Backspace' && e.target.value === '' && genes.length) {
    removeGene(genes[genes.length - 1]);
  }
});

function addGene(name) {
  if (!name || genes.includes(name)) return;
  genes.push(name);
  renderTags();
}

function removeGene(name) {
  genes = genes.filter(g => g !== name);
  renderTags();
}

function renderTags() {
  const container = document.getElementById('gene-tags');
  container.innerHTML = genes.map(g => `
    <span class="tag">
      ${g}
      <span class="tag-x" onclick="removeGene('${g}')">×</span>
    </span>`).join('');
}

// ── Load cluster columns from server ──────────────────────────────────────
document.getElementById('input-path').addEventListener('blur', loadClusterCols);

async function loadClusterCols() {
  const path = document.getElementById('input-path').value.trim();
  if (!path) return;
  const sel = document.getElementById('clust-col');
  sel.innerHTML = '<option>Loading...</option>';
  try {
    const res  = await fetch('/api/load_meta', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, filetype })
    });
    const data = await res.json();
    if (data.error) {
      sel.innerHTML = '<option>Error loading file</option>';
      appendLog('warn', 'Meta load error: ' + data.error);
      return;
    }
    sel.innerHTML = data.columns.map(c =>
      `<option value="${c.name}">${c.name} (${c.n_unique} values)</option>`
    ).join('');
    document.getElementById('card-2').classList.add('active');
    loadClusterValues();
  } catch(e) {
    sel.innerHTML = '<option>Could not connect</option>';
  }
}

async function loadClusterValues() {
  const path = document.getElementById('input-path').value.trim();
  const col  = document.getElementById('clust-col').value;
  if (!path || !col) return;

  const valSel = document.getElementById('clust-values');
  valSel.innerHTML = '<option>Loading...</option>';
  document.getElementById('clust-values-field').classList.remove('hidden');

  try {
    const res  = await fetch('/api/load_clusters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, filetype, column: col })
    });
    const data = await res.json();
    if (data.error) { valSel.innerHTML = '<option>Error</option>'; return; }
    valSel.innerHTML = data.values.map(v =>
      `<option value="${v.name}" selected>${v.name} (${v.count.toLocaleString()} cells)</option>`
    ).join('');
  } catch(e) {
    valSel.innerHTML = '<option>Error</option>';
  }
}

// ── Run pipeline ──────────────────────────────────────────────────────────
function runPipeline() {
  const path = document.getElementById('input-path').value.trim();
  if (!path) { alert('Please enter an input file/directory path.'); return; }
  if (genes.length === 0) { alert('Please add at least one gene name.'); return; }

  // Collect selected cluster values
  const valSel   = document.getElementById('clust-values');
  const selClusts = Array.from(valSel.selectedOptions).map(o => o.value);

  const config = {
    path,
    filetype,
    path2: document.getElementById('combine-check').checked
      ? document.getElementById('input-path2').value.trim() : null,
    clust_col:      document.getElementById('clust-col').value,
    chosen_clusters: selClusts,
    genes,
    nsim:           parseInt(document.getElementById('nsim').value) || 1000,
    power_target:   parseFloat(document.getElementById('power-target').value) || 0.95,
    doublet_max:    parseInt(document.getElementById('doublet-max').value) || 0,
    out_dir:        document.getElementById('out-dir').value.trim() || null,
  };

  // UI state
  setRunning(true);
  document.getElementById('output').innerHTML = '';
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('results-grid').innerHTML = '';

  // Send config, then open SSE stream
  fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  }).then(r => r.json()).then(d => {
    if (d.error) { appendLog('warn', d.error); setRunning(false); return; }
    openStream();
  });
}

function openStream() {
  if (es) es.close();
  es = new EventSource('/api/stream');

  es.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'done') {
      setRunning(false);
      setStatus('done');
      es.close();
      loadResults();
    } else if (msg.type === 'error') {
      appendLog('warn', msg.text);
      setRunning(false);
      setStatus('error');
      es.close();
    } else {
      appendLog(msg.cls || 'info', msg.text);
    }
  };

  es.onerror = () => {
    setRunning(false);
    setStatus('error');
  };
}

function appendLog(cls, text) {
  const out  = document.getElementById('output');
  const line = document.createElement('span');
  line.className = 'log-line ' + cls;
  line.textContent = text;
  out.appendChild(line);
  out.appendChild(document.createElement('br'));
  out.scrollTop = out.scrollHeight;
}

function setRunning(running) {
  const btn = document.getElementById('run-btn');
  btn.disabled = running;
  document.getElementById('run-label').style.display  = running ? 'none' : '';
  document.getElementById('run-spinner').style.display = running ? 'block' : 'none';
  setStatus(running ? 'running' : 'idle');
}

function setStatus(state) {
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  dot.className = 'status-dot ' + state;
  label.textContent = { idle:'Idle', running:'Running…', done:'Complete', error:'Error' }[state] || state;
}

// ── Results ───────────────────────────────────────────────────────────────
async function loadResults() {
  const res  = await fetch('/api/results');
  const data = await res.json();
  if (!data.files || data.files.length === 0) return;
  const section = document.getElementById('results-section');
  const grid    = document.getElementById('results-grid');
  section.classList.remove('hidden');
  grid.innerHTML = data.files.map(f => `
    <img class="result-thumb" src="/results/${f}?t=${Date.now()}"
         alt="${f}" title="${f}" onclick="openLightbox('/results/${f}?t=${Date.now()}')">
  `).join('');
}

// ── Lightbox ──────────────────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT STREAMING
# ══════════════════════════════════════════════════════════════════════════════

class QueueStream:
    """Captures print() output and routes it to a queue for SSE streaming."""
    def __init__(self, q):
        self.q = q

    def write(self, text):
        if not text.strip():
            return
        # Classify lines for coloring in the UI
        t = text.strip()
        if t.startswith('[R]') or t.startswith('  [R]'):
            cls = 'r'
        elif any(t.startswith(p) for p in ('══', '──', '╔', '╚', '║')):
            cls = 'head'
        elif any(kw in t for kw in ('Error', 'error', 'failed', 'WARNING', '⚠')):
            cls = 'warn'
        elif any(kw in t for kw in ('✓', 'Saved:', 'Found', 'loaded', 'complete')):
            cls = 'accent'
        elif t.startswith('#') or t.startswith('//'):
            cls = 'muted'
        else:
            cls = 'info'
        self.q.put(json.dumps({'type': 'log', 'cls': cls, 'text': t}))

    def flush(self):
        pass


_stream_queue = queue.Queue()
_results      = []


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/load_meta', methods=['POST'])
def api_load_meta():
    """Return categorical obs columns for cluster selection."""
    data     = request.json
    path     = data.get('path', '').strip()
    filetype = data.get('filetype', 'h5ad')

    try:
        import anndata
        if filetype == 'h5ad':
            # Fast: only load obs metadata
            actual_path = _resolve_h5ad(path)
            if not actual_path:
                return jsonify({'error': 'File not found: ' + path})
            adata = anndata.read_h5ad(actual_path, backed='r')
        else:
            # MTX: no obs metadata to load — return dataset column only
            if not os.path.isdir(path):
                return jsonify({'error': 'Directory not found: ' + path})
            return jsonify({'columns': [{'name': 'dataset', 'n_unique': 1}]})

        PRIORITY = ['cell_type', 'author_cell_type', 'majorclass', 'subclass_label',
                    'celltype', 'CellType', 'cluster', 'leiden', 'louvain']
        obs = adata.obs
        all_cols  = [c for c in obs.columns
                     if obs[c].dtype.name in ('category', 'object')
                     and 1 < obs[c].nunique() < 500]
        priority  = [c for c in PRIORITY if c in all_cols]
        others    = sorted(c for c in all_cols if c not in priority)
        cols      = priority + others
        result    = [{'name': c, 'n_unique': int(obs[c].nunique())} for c in cols]
        return jsonify({'columns': result})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/load_clusters', methods=['POST'])
def api_load_clusters():
    """Return unique values + counts for a given obs column."""
    data     = request.json
    path     = data.get('path', '').strip()
    filetype = data.get('filetype', 'h5ad')
    column   = data.get('column', '')

    try:
        import anndata
        actual_path = _resolve_h5ad(path) if filetype == 'h5ad' else None
        if filetype != 'h5ad':
            return jsonify({'values': [{'name': 'All', 'count': 0}]})

        adata   = anndata.read_h5ad(actual_path, backed='r')
        vc      = adata.obs[column].astype(str).value_counts().sort_values(ascending=False)
        values  = [{'name': str(k), 'count': int(v)} for k, v in vc.items()]
        return jsonify({'values': values})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/run', methods=['POST'])
def api_run():
    """Validate config and launch pipeline in background thread."""
    global _job_active, _results
    if _job_active:
        return jsonify({'error': 'A job is already running. Please wait.'})

    config = request.json
    _results = []

    # Clear stream queue
    while not _stream_queue.empty():
        try: _stream_queue.get_nowait()
        except: pass

    thread = threading.Thread(target=_run_pipeline, args=(config,), daemon=True)
    thread.start()
    return jsonify({'ok': True})


@app.route('/api/stream')
def api_stream():
    """SSE endpoint that streams pipeline output."""
    def generate():
        while True:
            try:
                msg = _stream_queue.get(timeout=30)
                yield 'data: {}\n\n'.format(msg)
                if json.loads(msg).get('type') in ('done', 'error'):
                    break
            except queue.Empty:
                yield 'data: {}\n\n'.format(json.dumps({'type':'log','cls':'muted','text':'...'}))
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/results')
def api_results():
    """Return list of output PNG files."""
    out_dir = OUTPUT_DIR
    pngs    = sorted(f for f in os.listdir(out_dir) if f.endswith('.png'))
    return jsonify({'files': pngs})


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename), mimetype='image/png')


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_h5ad(path):
    """Resolve path to an actual .h5ad file."""
    if os.path.isfile(path) and path.lower().endswith('.h5ad'):
        return path
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.lower().endswith('.h5ad'):
                return os.path.join(path, f)
    return None


def _run_pipeline(config):
    global _job_active
    _job_active = True
    orig_stdout = sys.stdout
    sys.stdout  = QueueStream(_stream_queue)

    try:
        import scRNAseq_pipeline as pl
        import anndata as ad
        import scipy.sparse as sp

        path     = config['path']
        filetype = config['filetype']
        out_dir  = config.get('out_dir') or os.path.dirname(path) or SCRIPT_DIR

        print('══════════════════════════════════════════════════════')
        print('  scRNA-Seq Pipeline started')
        print('  Input   : ' + path)
        print('  Type    : ' + filetype.upper())
        print('══════════════════════════════════════════════════════')

        # ── Load data ─────────────────────────────────────────────────────
        if filetype == 'h5ad':
            actual = _resolve_h5ad(path)
            if not actual:
                raise FileNotFoundError('No .h5ad file found at: ' + path)

            print('\n── H5AD input detected ─────────────────────────────')
            print('Reading .h5ad file (memory-mapped)...')
            adata = ad.read_h5ad(actual, backed='r')
            print('Loaded: {:,} genes x {:,} cells'.format(adata.n_vars, adata.n_obs))

            # Subset to chosen clusters
            clust_col       = config.get('clust_col', '')
            chosen_clusters = config.get('chosen_clusters', [])
            if clust_col and chosen_clusters:
                mask = adata.obs[clust_col].astype(str).isin(chosen_clusters)
                row_idx = np.where(mask)[0]
                print('Loading {:,} cells into memory...'.format(len(row_idx)))
                adata = adata[row_idx].to_memory()
                print('Subset: {:,} cells from: {}'.format(
                    adata.n_obs, ', '.join(chosen_clusters)))
            else:
                print('Loading full dataset into memory...')
                adata = adata.to_memory()

            # Switch to raw counts
            if adata.raw is not None:
                print('Using adata.raw.X (raw UMI counts)')
                raw_X = adata.raw.X
                if not sp.issparse(raw_X):
                    raw_X = sp.csr_matrix(raw_X)
                adata = ad.AnnData(X=raw_X, obs=adata.obs, var=adata.raw.var)
            else:
                print('No raw slot — using adata.X as-is')

            cell_clusters = (adata.obs[clust_col].astype(str).values
                             if clust_col and clust_col in adata.obs.columns else None)

        else:
            # MTX — delegate to pipeline loader
            adata, cell_clusters = pl.load_mtx(path)

        print('\nFinal matrix: {:,} genes x {:,} cells'.format(adata.n_vars, adata.n_obs))

        # ── Validate genes ─────────────────────────────────────────────────
        available   = list(adata.var_names)
        avail_lower = {a.lower(): a for a in available}
        target_genes = []
        missing = []
        for g in config.get('genes', []):
            if g in available:
                target_genes.append(g)
            elif g.lower() in avail_lower:
                corrected = avail_lower[g.lower()]
                print("Matched '{}' -> '{}' (case corrected)".format(g, corrected))
                target_genes.append(corrected)
            else:
                missing.append(g)

        if missing:
            print('⚠  Genes not found: ' + ', '.join(missing))
        if not target_genes:
            raise ValueError('None of the requested genes were found in the dataset.')
        print('✓ Genes to analyse: ' + ', '.join(target_genes))

        # ── QC plots ───────────────────────────────────────────────────────
        umi_per_cell, genes_per_cell = pl.compute_qc(adata)
        os.chdir(out_dir)
        pl.make_qc_plots(umi_per_cell, genes_per_cell, cell_clusters)

        # ── Per-gene analysis ──────────────────────────────────────────────
        X     = adata.X
        nsim  = config.get('nsim', 1000)
        ptgt  = config.get('power_target', 0.95)

        for gene in target_genes:
            print('\n====  Analysing gene: {}  ===='.format(gene))
            gene_idx = list(adata.var_names).index(gene)

            if sp.issparse(X):
                col = X[:, gene_idx]
                cts = np.array(col.todense()).flatten().astype(int)
            else:
                cts = np.array(X[:, gene_idx]).flatten().astype(int)

            mean_umi    = float(np.mean(cts))
            dropout_pct = 100.0 * float(np.mean(cts == 0))
            disp        = pl.bio_dispersion(cts)
            print('  Mean UMI: {:.2f} | Dropout: {:.1f}% | Dispersion: {:.3f}'.format(
                  mean_umi, dropout_pct, disp))

            pl.gene_histogram(cts, gene, mean_umi, dropout_pct, disp)

            print('  Fitting ZINB distribution...')
            zinb_par = pl.fit_zinb(cts)
            print('  ZINB: mu={:.3f}, theta={:.3f}, pi={:.3f}'.format(
                  zinb_par['mu'], zinb_par['theta'], zinb_par['pi']))

            print('  Simulating power curves ({} simulations)...'.format(nsim))
            results, thresholds = pl.power_curve(zinb_par, nsim=nsim,
                                                  power_target=ptgt)
            pl.power_plot(results, thresholds, gene)

        print('\n╔══════════════════════════════════════════════════════════════╗')
        print('║  Pipeline complete. Output files saved.                      ║')
        print('╚══════════════════════════════════════════════════════════════╝')
        _stream_queue.put(json.dumps({'type': 'done'}))

    except Exception:
        err = traceback.format_exc()
        sys.stdout = orig_stdout
        for line in err.splitlines():
            _stream_queue.put(json.dumps({'type': 'log', 'cls': 'warn', 'text': line}))
        _stream_queue.put(json.dumps({'type': 'error', 'text': 'Pipeline failed — see output above'}))
    finally:
        sys.stdout  = orig_stdout
        _job_active = False


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '='*60)
    print('  scRNA-Seq Pipeline — Web Interface')
    print('  Open http://localhost:5050 in your browser')
    print('='*60 + '\n')
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
