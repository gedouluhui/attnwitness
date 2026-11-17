"""Offline evidence grid; all user-controlled labels are escaped as text."""
from html import escape
import json

_CSS = """
:root{color-scheme:light;--ink:#1b263b;--muted:#607089;--line:#d5deeb;--paper:#f7f9fc;--blue:#2e55cc;--red:#b2384a;--green:#166b59}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 'Segoe UI',Arial,sans-serif}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible{outline:3px solid var(--blue);outline-offset:3px}
.shell{max-width:1220px;margin:auto;padding:30px 38px 60px}.top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:22px;gap:20px}.brand{font:700 24px Bahnschrift,'Segoe UI',sans-serif;letter-spacing:-.6px}.brand span{color:var(--blue)}.meta,.eyebrow{font:12px Consolas,monospace;letter-spacing:.1em;color:var(--muted)}.hero{padding:40px 0 30px;max-width:850px}h1{font:600 clamp(32px,4.2vw,52px)/1.1 Bahnschrift,'Segoe UI',sans-serif;letter-spacing:-1.4px;margin:16px 0}.hero p{color:var(--muted);max-width:740px}.badge{display:inline-block;font:700 12px Consolas,monospace;letter-spacing:.07em;padding:5px 10px;border:1px solid currentColor;border-radius:4px}.bad{color:var(--red)}.good{color:var(--green)}
.layout{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(260px,1fr);gap:22px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:24px;min-width:0}.card h2{font-size:18px;margin:0 0 6px}.caption{font-size:13px;color:var(--muted)}.matrix-wrap{overflow:auto;margin:22px 0 16px;max-height:640px}.matrix{border-collapse:separate;border-spacing:5px}.matrix th{font:11px Consolas,monospace;max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);padding:4px}.matrix th[scope=row]{text-align:right}.cell{width:36px;height:36px;border:1px solid #d5deeb;border-radius:5px;background:#edf2fb;color:#385287;font:13px Consolas,monospace}.cell.blocked{background:#fafbfc;color:#758294}.cell.conflict{background:#fce9ee;border-color:#ce7b8b;color:#9f2038}.cell.free{border-style:dashed}.cell:hover{outline:2px solid var(--blue)}.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px;background:#edf2fb;border:1px solid #d5deeb}.legend .red{background:#fce9ee;border-color:#ce7b8b}.inspector{background:#eef3f9;padding:14px;border-radius:6px;min-height:70px;margin-top:18px;overflow-wrap:anywhere;font:13px/1.7 Consolas,monospace}.facts{margin:20px 0}.facts div{display:flex;justify-content:space-between;gap:15px;padding:10px 0;border-bottom:1px solid #e9edf4}.facts dt{color:var(--muted)}.facts dd{margin:0;font-family:Consolas,monospace}.note{border-left:3px solid var(--blue);padding:0 0 0 15px;margin-top:24px;font-size:13px}.findings{margin-top:24px}.filters{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}.filter,.download{border:1px solid var(--line);background:white;border-radius:5px;padding:7px 11px;color:var(--ink)}.filter[aria-pressed=true]{background:#eaf0ff;border-color:var(--blue);color:var(--blue)}.finding{padding:14px 0;border-top:1px solid #e6ebf3;display:grid;grid-template-columns:160px minmax(0,1fr) 100px;gap:12px;font-size:13px}.finding[hidden]{display:none}.edge{font-family:Consolas,monospace;overflow-wrap:anywhere}.location{color:var(--muted)}.empty{padding:20px;border:1px dashed var(--line);border-radius:6px;color:var(--muted)}details{margin-top:22px}summary{cursor:pointer;font-size:13px;color:var(--muted)}pre{overflow:auto;max-height:360px;background:#edf2f8;padding:14px;border-radius:6px;font-size:12px}.footer{margin-top:30px;border-top:1px solid var(--line);padding-top:18px;font-size:12px;color:var(--muted)}@media(max-width:760px){.shell{padding:20px 16px 40px}.layout{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column;gap:8px}.hero{padding-top:26px}.card{padding:18px}.finding{grid-template-columns:1fr;gap:2px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
"""
_SCRIPT = """
document.querySelectorAll('[data-message]').forEach(function(cell){cell.addEventListener('click',function(){document.getElementById('inspector').textContent=cell.dataset.message;});});
document.querySelectorAll('[data-filter]').forEach(function(button){button.addEventListener('click',function(){document.querySelectorAll('[data-filter]').forEach(function(b){b.setAttribute('aria-pressed',String(b===button));});document.querySelectorAll('.finding').forEach(function(row){row.hidden=button.dataset.filter!=='all'&&row.dataset.category!==button.dataset.filter;});});});
document.getElementById('download').addEventListener('click',function(){var text=document.getElementById('report-json').textContent;var u=URL.createObjectURL(new Blob([text],{type:'application/json'}));var a=document.createElement('a');a.href=u;a.download='attnwitness-report.json';a.click();setTimeout(function(){URL.revokeObjectURL(u);},1000);});
"""


def render_report(result):
    data = result.to_dict() if hasattr(result, 'to_dict') else result
    status = str(data.get('status', 'ERROR'))
    good = status == 'PASS'
    findings = data.get('findings', [])
    coverage = data.get('coverage', {})
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
            conflict = (intent == -1 and allows) or (intent == 1 and not allows)
            classes = 'cell' + (' blocked' if not allows else '') + (' conflict' if conflict else '') + (' free' if intent == 0 else '')
            want = {-1: 'must block', 0: 'unconstrained', 1: 'must permit'}[intent]
            message = f'{name} → {key}: {"visible" if allows else "blocked"}; declaration: {want}.'
            table.append(f'<td><button class="{classes}" data-message="{escape(message, quote=True)}" '
                         f'aria-label="{escape(message, quote=True)}">{"!" if conflict else "1" if allows else "0"}</button></td>')
        table.append('</tr>')
    table.append('</tbody></table>')
    rows = []
    for f in findings:
        category = str(f['category'])
        edge = str(f['query_id']) + (' → ' + str(f['key_id']) if f.get('key_id') is not None else ' → no visible key')
        rows.append(f'<div class="finding" data-category="{escape(category, quote=True)}"><strong>{escape(category.replace("_", " "))}</strong>'
                    f'<span class="edge">{escape(edge)}</span><span class="location">B{escape(str(f["batch"]))} / H{escape(str(f["head"]))}</span></div>')
    facts = ''.join(f'<div><dt>{label}</dt><dd>{escape(str(coverage.get(key, "—")))}</dd></div>' for key, label in
                    [('backend', 'Mask encoding'), ('batch_size', 'Batches'), ('heads', 'Heads'),
                     ('total_cells', 'Relation cells'), ('constrained_cells', 'Declared cells'), ('unconstrained_cells', 'Unconstrained')])
    warn = 'This preview shows only the first batch/head and up to 16 tokens.' if preview.get('truncated') else 'Click a cell to inspect its declared and observed relation.'
    error = f'<p class="bad">{escape(str(data["error"]))}</p>' if data.get('error') else ''
    serialized = escape(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    filters = ''.join(f'<button class="filter" data-filter="{kind}" aria-pressed="{str(kind == "all").lower()}">{label}</button>'
                      for kind, label in [('all', 'All findings'), ('forbidden_edge', 'Forbidden edges'),
                                          ('missing_edge', 'Missing edges'), ('empty_query', 'Empty queries')])
    truncation = '<p class="caption">Finding list truncated; the total below includes all detected violations.</p>' if data.get('findings_truncated') else ''
    empty_message = ('No declared relation violation was found. Review coverage before interpreting the result.'
                     if good else 'No finding list is available. An incomplete or rejected audit is not a PASS.')
    empty_html = '<div class="empty">' + empty_message + '</div>'
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
<div class="note">A PASS covers only these declarations. It does not prove model quality, whole-model causality or numerical safety of empty attention rows. Finite negative biases are not hard blocks.</div></aside></div>
<section class="card findings"><h2>Concrete findings · {escape(str(data.get('violation_count', 0)))}</h2>{truncation}<div class="filters">{filters}</div>
{''.join(rows) if rows else empty_html}</section>
<details><summary>Machine-readable evidence and input fingerprints</summary><pre id="report-json">{serialized}</pre></details>
<footer class="footer"><button class="download" id="download">Download report JSON</button><p>AttnWitness · offline report · synthetic examples are not model-quality benchmarks.</p></footer>
</main><script>{_SCRIPT}</script></body></html>'''
