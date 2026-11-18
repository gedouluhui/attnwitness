"""Offline evidence grid and original-coordinate witness panels; all labels escaped."""
from html import escape
import json

_CSS = """
:root{color-scheme:light;--ink:#1b263b;--muted:#607089;--line:#d5deeb;--paper:#f7f9fc;--blue:#2e55cc;--red:#b2384a;--green:#166b59}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 'Segoe UI',Arial,sans-serif}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible{outline:3px solid var(--blue);outline-offset:3px}
.shell{max-width:1220px;margin:auto;padding:30px 38px 60px}.top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:22px;gap:20px}.brand{font:700 24px Bahnschrift,'Segoe UI',sans-serif;letter-spacing:-.6px}.brand span{color:var(--blue)}.meta,.eyebrow{font:12px Consolas,monospace;letter-spacing:.1em;color:var(--muted)}.hero{padding:40px 0 30px;max-width:850px}h1{font:600 clamp(32px,4.2vw,52px)/1.1 Bahnschrift,'Segoe UI',sans-serif;letter-spacing:-1.4px;margin:16px 0}.hero p{color:var(--muted);max-width:740px}.badge{display:inline-block;font:700 12px Consolas,monospace;letter-spacing:.07em;padding:5px 10px;border:1px solid currentColor;border-radius:4px;white-space:nowrap}.bad{color:var(--red)}.good{color:var(--green)}
.layout{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(260px,1fr);gap:22px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:24px;min-width:0}.card h2{font-size:18px;margin:0 0 6px}.caption{font-size:13px;color:var(--muted)}.matrix-wrap{overflow:auto;margin:22px 0 16px;max-height:640px}.matrix{border-collapse:separate;border-spacing:5px}.matrix th{font:11px Consolas,monospace;max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);padding:4px}.matrix th[scope=row]{text-align:right}.matrix .index{color:var(--ink);font-weight:700}.cell{width:36px;height:36px;border:1px solid #d5deeb;border-radius:5px;background:#edf2fb;color:#385287;font:13px Consolas,monospace;text-align:center;vertical-align:middle;padding:0}button.cell{display:inline-flex;align-items:center;justify-content:center}.cell.blocked{background:#fafbfc;color:#758294}.cell.conflict{background:#fce9ee;border-color:#ce7b8b;color:#9f2038}.cell.free{border-style:dashed}.cell:hover{outline:2px solid var(--blue)}.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px;background:#edf2fb;border:1px solid #d5deeb}.legend .red{background:#fce9ee;border-color:#ce7b8b}.inspector{background:#eef3f9;padding:14px;border-radius:6px;min-height:70px;margin-top:18px;overflow-wrap:anywhere;font:13px/1.7 Consolas,monospace}.facts{margin:20px 0}.facts div{display:flex;justify-content:space-between;gap:15px;padding:10px 0;border-bottom:1px solid #e9edf4}.facts dt{color:var(--muted)}.facts dd{margin:0;font-family:Consolas,monospace}.note{border-left:3px solid var(--blue);padding:0 0 0 15px;margin-top:24px;font-size:13px}.note.warn{border-left-color:var(--red)}code{font-family:Consolas,monospace;font-size:.95em}
.witnesses{margin-top:24px}.witness{border-top:1px solid #e6ebf3;padding-top:16px;margin-top:18px}.witness[hidden]{display:none}.witness h3{font:600 16px/1.45 Bahnschrift,'Segoe UI',sans-serif;margin:10px 0 4px;overflow-wrap:anywhere}.witness-head{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.witness .matrix-wrap{margin:14px 0 10px;max-height:420px}.witness .matrix .cell{width:34px;height:34px}.witness .empty,.witness-empty{margin-top:14px}
.findings{margin-top:24px}.filters{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}.filter,.download{border:1px solid var(--line);background:white;border-radius:5px;padding:7px 11px;color:var(--ink)}.filter[aria-pressed=true]{background:#eaf0ff;border-color:var(--blue);color:var(--blue)}.finding{padding:14px 10px;border:1px solid transparent;border-top:1px solid #e6ebf3;border-radius:4px;display:grid;grid-template-columns:160px minmax(0,1fr) 130px;gap:12px;font-size:13px;align-items:center}.finding[hidden]{display:none}button.finding{background:white;color:inherit;width:100%;text-align:left;border-radius:0}.finding:hover{background:#f5f8ff}.finding[aria-expanded=true]{background:#eaf0ff;border-left:3px solid var(--blue)}.edge{font-family:Consolas,monospace;overflow-wrap:anywhere}.location{color:var(--muted)}.open-hint{display:block;font:11px Consolas,monospace;color:var(--blue)}.empty{padding:20px;border:1px dashed var(--line);border-radius:6px;color:var(--muted)}details{margin-top:22px}summary{cursor:pointer;font-size:13px;color:var(--muted)}pre{overflow:auto;max-height:360px;background:#edf2f8;padding:14px;border-radius:6px;font-size:12px}.footer{margin-top:30px;border-top:1px solid var(--line);padding-top:18px;font-size:12px;color:var(--muted)}@media(max-width:760px){.shell{padding:20px 16px 40px}.layout{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column;gap:8px}.hero{padding-top:26px}.card{padding:18px}.finding{grid-template-columns:1fr;gap:2px}.witness .matrix .cell{width:30px;height:30px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
"""
_SCRIPT = """
document.querySelectorAll('[data-message]').forEach(function(cell){cell.addEventListener('click',function(){document.getElementById('inspector').textContent=cell.dataset.message;});});
var witnesses=Array.prototype.slice.call(document.querySelectorAll('[data-witness]'));
function selectWitness(button){var id=button.dataset.witness;witnesses.forEach(function(b){var on=b===button;b.setAttribute('aria-expanded',String(on));});document.querySelectorAll('[data-witness-panel]').forEach(function(panel){panel.hidden=panel.dataset.witnessPanel!==id;});var empty=document.getElementById('witness-empty');if(empty){empty.hidden=true;}var panel=document.getElementById('witness-'+id);if(panel){panel.scrollIntoView({block:'nearest'});}}
witnesses.forEach(function(button){button.addEventListener('click',function(){selectWitness(button);});});
document.querySelectorAll('[data-filter]').forEach(function(button){button.addEventListener('click',function(){document.querySelectorAll('[data-filter]').forEach(function(b){b.setAttribute('aria-pressed',String(b===button));});var kind=button.dataset.filter;document.querySelectorAll('.finding').forEach(function(row){row.hidden=kind!=='all'&&row.dataset.category!==kind;});var kept=false;witnesses.forEach(function(b){if(b.hidden){b.setAttribute('aria-expanded','false');var panel=document.getElementById('witness-'+b.dataset.witness);if(panel){panel.hidden=true;}}else{kept=true;}});var empty=document.getElementById('witness-empty');if(empty){empty.hidden=kept;}var note=document.getElementById('witness-count');if(note){note.textContent=kept?'':'No retained witness is shown under this filter; the category counts beside each filter are totals from the audit, not display counts.';}});});
document.getElementById('download').addEventListener('click',function(){var text=document.getElementById('report-json').textContent;var u=URL.createObjectURL(new Blob([text],{type:'application/json'}));var a=document.createElement('a');a.href=u;a.download='attnwitness-report.json';a.click();setTimeout(function(){URL.revokeObjectURL(u);},1000);});
"""

_FILTERS = [('all', 'All findings'), ('forbidden_edge', 'Forbidden edges'),
            ('missing_edge', 'Missing edges'), ('empty_query', 'Empty queries')]


def _matrix_cell(allows, intent):
    conflict = (intent == -1 and allows) or (intent == 1 and not allows)
    classes = 'cell' + (' blocked' if not allows else '') + (' conflict' if conflict else '') + (' free' if intent == 0 else '')
    glyph = '!' if conflict else '1' if allows else '0'
    return classes, glyph


def _witness_panel(index, window, finding, total):
    b, h = window.get('batch'), window.get('head')
    qidx, kidx = window.get('query_indices', []), window.get('key_indices', [])
    qids, kids = window.get('query_ids', []), window.get('key_ids', [])
    expected, visible = window.get('expected', []), window.get('visible', [])
    category = str(finding.get('category', 'finding')).replace('_', ' ')
    q = finding.get('query_index')
    k = finding.get('key_index')
    coords = f'B{b} · H{h} · q{q} → no visible key' if k is None else f'B{b} · H{h} · q{q} → k{k}'
    head = (f'<header class="witness-head"><span class="badge bad">{escape(category.upper())}</span>'
            f'<span class="caption">finding {index + 1} of {total}</span></header><h3>{escape(coords)}</h3>')
    ids_line = (f'<p class="caption">Witness panel: original coordinates, not the preview window. '
                f'Query <code>{escape(str(finding.get("query_id")))}</code>')
    if k is not None:
        ids_line += f' · key <code>{escape(str(finding.get("key_id")))}</code>'
    ids_line += '.</p>'
    table = ['<table class="matrix witness-matrix"><thead><tr><th>query / key</th>']
    table.extend(f'<th scope="col" title="key {escape(str(ki))}"><span class="index">{escape(str(ki))}</span> '
                 f'{escape(str(name))}</th>' for ki, name in zip(kidx, kids))
    table.append('</tr></thead><tbody>')
    for qi, allows_row in enumerate(visible):
        table.append(f'<tr><th scope="row" title="query {escape(str(qidx[qi] if qi < len(qidx) else qi))}">'
                     f'<span class="index">{escape(str(qidx[qi] if qi < len(qidx) else qi))}</span> '
                     f'{escape(str(qids[qi] if qi < len(qids) else ""))}</th>')
        for kj, allows in enumerate(allows_row):
            intent = 0
            if qi < len(expected) and kj < len(expected[qi]):
                intent = expected[qi][kj]
            classes, glyph = _matrix_cell(allows, intent)
            want = {-1: 'must block', 1: 'must permit'}.get(intent, 'unconstrained')
            message = (f'{qids[qi] if qi < len(qids) else qi} → {kids[kj] if kj < len(kids) else kj}: '
                       f'{"visible" if allows else "blocked"}; declaration: {want}.')
            table.append(f'<td class="{classes}" title="{escape(message, quote=True)}" '
                         f'aria-label="{escape(message, quote=True)}">{glyph}</td>')
        table.append('</tr>')
    table.append('</tbody></table>')
    extra = ''
    if 'row_summary' in window:
        summary = window['row_summary']
        extra = (f'<p class="caption">Empty query row: all {escape(str(summary.get("keys_checked", "?")))} keys '
                 f'were checked; {escape(str(summary.get("visible_keys", 0)))} were visible. The declared contract '
                 f'requires a nonempty visible row; the window above shows only its first keys.</p>')
    semantics = ('<p class="caption">Hard visibility: boolean masks are visibility as reported by the backend; '
                 'float masks are additive scores where only <code>-inf</code> blocks. MHA attention and '
                 'key-padding masks are summed in the mask dtype before this judgement; a sum producing '
                 'NaN or +inf makes the input invalid. Finite negative scores remain visible.</p>')
    return (f'<article class="witness" id="witness-{index}" data-witness-panel="{index}" hidden>'
            f'{head}{ids_line}<div class="matrix-wrap">{"".join(table)}</div>{extra}{semantics}</article>')


def render_report(result):
    data = result.to_dict() if hasattr(result, 'to_dict') else result
    status = str(data.get('status', 'ERROR'))
    good = status == 'PASS'
    findings = data.get('findings', [])
    evidence = data.get('evidence', [])
    coverage = data.get('coverage', {})
    counts = data.get('counts') or {}
    violation_count = data.get('violation_count', 0)
    preview = data.get('preview', {})
    ids = preview.get('ids', [])
    visible, expected = preview.get('visible', []), preview.get('expected', [])
    title = 'The declared contract holds.' if good else 'A visibility contract needs attention.'
    if status == 'ERROR':
        title = 'The input could not be checked.'
    table = ['<table class="matrix"><thead><tr><th>query / key</th>']
    table.extend(f'<th scope="col" title="{escape(str(name), quote=True)}">{escape(str(name))}</th>' for name in ids)
    table.append('</tr></thead><tbody>')
    for q, name in enumerate(ids):
        table.append(f'<tr><th scope="row" title="{escape(str(name), quote=True)}">{escape(str(name))}</th>')
        for k, key in enumerate(ids):
            allows, intent = visible[q][k], expected[q][k]
            classes, glyph = _matrix_cell(allows, intent)
            want = {-1: 'must block', 0: 'unconstrained', 1: 'must permit'}[intent]
            message = f'{name} → {key}: {"visible" if allows else "blocked"}; declaration: {want}.'
            table.append(f'<td><button class="{classes}" data-message="{escape(message, quote=True)}" '
                         f'aria-label="{escape(message, quote=True)}">{glyph}</button></td>')
        table.append('</tr>')
    table.append('</tbody></table>')
    windows = {index: window for index, window in
               ((w.get('finding_index'), w) for w in evidence if isinstance(w, dict)) if isinstance(index, int)}
    rows = []
    for index, f in enumerate(findings):
        category = str(f['category'])
        edge = str(f['query_id']) + (' → ' + str(f['key_id']) if f.get('key_id') is not None else ' → no visible key')
        coords = f'B{escape(str(f["batch"]))} / H{escape(str(f["head"]))}'
        if f.get('query_index') is not None:
            coords += f' · q{escape(str(f["query_index"]))}'
            if f.get('key_index') is not None:
                coords += f' / k{escape(str(f["key_index"]))}'
        if index in windows:
            rows.append(f'<button class="finding" data-category="{escape(category, quote=True)}" data-witness="{index}" '
                        f'aria-controls="witness-{index}" aria-expanded="false"><strong>{escape(category.replace("_", " "))}</strong>'
                        f'<span class="edge">{escape(edge)}</span><span class="location">{coords}'
                        f'<span class="open-hint">open context</span></span></button>')
        else:
            rows.append(f'<div class="finding" data-category="{escape(category, quote=True)}"><strong>'
                        f'{escape(category.replace("_", " "))}</strong><span class="edge">{escape(edge)}</span>'
                        f'<span class="location">{coords} · context not retained</span></div>')
    panels = ''.join(_witness_panel(index, window, findings[index], len(findings))
                     for index, window in sorted(windows.items()) if index < len(findings))
    facts = ''.join(f'<div><dt>{label}</dt><dd>{escape(str(value))}</dd></div>' for label, value in [
        ('Mask encoding', coverage.get('backend', '—')), ('Batches', coverage.get('batch_size', '—')),
        ('Heads', coverage.get('heads', '—')), ('Relation cells', coverage.get('total_cells', '—')),
        ('Checked cells', coverage.get('checked_cells', '—')),
        ('Declared cells', coverage.get('constrained_cells', '—')),
        ('Unconstrained', coverage.get('unconstrained_cells', '—')),
        ('Largest tile cells', coverage.get('largest_tile_cells', '—')),
        ('Sampling', 'none — exact' if coverage.get('sampling') is False else '—')])
    complete = coverage.get('complete_relation_contract')
    coverage_note = ('Every relation cell in this shape carried a declaration.' if complete is True else
                     'The declaration is partial; cells without a declaration are not evidence of correctness.')
    partial = ''
    if complete is False:
        unconstrained = coverage.get('unconstrained_cells', '?')
        unit = 'cell' if unconstrained == 1 else 'cells'
        partial = (f'<p class="note warn"><strong>Partial declaration:</strong> '
                   f'{escape(str(coverage.get("constrained_cells", "?")))} of '
                   f'{escape(str(coverage.get("total_cells", "?")))} relation cells were declared; '
                   f'{escape(str(unconstrained))} unconstrained {unit} {"" if unit == "cells" else "was"}'
                   f'{"were" if unit == "cells" else ""} not asserted.</p>')
    if coverage.get('sampling') is not False and complete is not None:
        coverage_note += ' Sampling status is absent from this payload.'
    warn = ('This preview shows only the first batch/head and up to 16 tokens.'
            if preview.get('truncated') else 'Click a cell to inspect its declared and observed relation.')
    error = f'<p class="bad">{escape(str(data["error"]))}</p>' if data.get('error') else ''
    serialized = escape(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

    def count_of(kind):
        return violation_count if kind == 'all' else counts.get(kind, '—')

    filters = ''.join(f'<button class="filter" data-filter="{kind}" data-count="{escape(str(count_of(kind)), quote=True)}" '
                      f'aria-pressed="{str(kind == "all").lower()}">{label} · {escape(str(count_of(kind)))}</button>'
                      for kind, label in _FILTERS)
    truncation = '<p class="caption">Finding list truncated; the total below includes all detected violations.</p>' if data.get('findings_truncated') else ''
    evidence_note = ''
    if data.get('evidence_truncated'):
        evidence_note = (f'<p class="caption">Context windows are retained for the first {len(evidence)} of '
                         f'{len(findings)} listed findings; counts above remain global.</p>')
    empty_message = ('No declared relation violation was found. Review coverage before interpreting the result.'
                     if good else 'No finding list is available. An incomplete or rejected audit is not a PASS.')
    empty_html = '<div class="empty">' + empty_message + '</div>'
    if windows:
        witness_empty = ('<div class="empty witness-empty" id="witness-empty">Select a finding below to open its own '
                         'batch/head context with original row and column indices. Selection works with mouse, '
                         'Enter or Space.</div>')
    elif findings:
        witness_empty = ('<div class="empty witness-empty" id="witness-empty">No context window was retained for '
                         'these findings (max_evidence=0). Counts in the findings section remain complete.</div>')
    else:
        witness_empty = '<div class="empty witness-empty" id="witness-empty">No witness context to open.</div>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>AttnWitness — {escape(status)}</title><style>{_CSS}</style></head><body><main class="shell">
<header class="top"><div class="brand">Attn<span>Witness</span></div><div class="meta">ATTENTION CONTRACT / LOCAL EVIDENCE</div></header>
<section class="hero"><span class="badge {'good' if good else 'bad'}">{escape(status)}</span><h1>{title}</h1>
<p>Inspect the exact relation between an attention mask and the visibility you declared. No model download. No remote API. No hidden verdict.</p>{error}</section>
<div class="layout"><section class="card"><h2>Query → key evidence</h2><div class="caption">Preview · batch 0 / head 0</div>
<div class="matrix-wrap">{''.join(table) if ids else '<p>No matrix available.</p>'}</div>
<div class="legend"><span><i></i>1 visible / 0 blocked</span><span><i class="red"></i>! contract violation</span><span>Dashed: unconstrained</span></div>
<div class="inspector" id="inspector" aria-live="polite">{warn}</div></section>
<aside class="card"><div class="eyebrow">WHAT WAS CHECKED</div><h2>Coverage, not a blanket proof</h2><dl class="facts">{facts}</dl>
<div class="note">{coverage_note} A PASS covers only these declarations. It does not prove model quality, whole-model causality or numerical safety of empty attention rows.</div>{partial}</aside></div>
<section class="card witnesses"><h2>Witness context</h2><div class="caption">Each retained finding opens a window at its own batch, head and original indices; no preview substitution.</div>
{panels if panels else ''}{witness_empty}<p class="caption" id="witness-count"></p></section>
<section class="card findings"><h2>Concrete findings · {escape(str(violation_count))}</h2>{truncation}{evidence_note}<div class="filters">{filters}</div>
{''.join(rows) if rows else empty_html}</section>
<details><summary>Machine-readable evidence and input fingerprints</summary><pre id="report-json">{serialized}</pre></details>
<footer class="footer"><button class="download" id="download">Download report JSON</button><p>AttnWitness · offline report · synthetic examples are not model-quality benchmarks.</p></footer>
</main><script>{_SCRIPT}</script></body></html>'''
