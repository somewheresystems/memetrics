"""Render a scorecard (:func:`memetrics.scorecard.run`) as JSON, Markdown and a self-contained HTML page."""

from __future__ import annotations

import html
import json
from pathlib import Path

PROCESS_ORDER = ("replicator", "attraction", "neutral", "contagion")


def _cell(x) -> str:
    return "—" if x is None or x == "" else str(x)


def _confusion_rows(block: dict) -> list[list[str]]:
    labels = sorted({k for row in block["confusion"].values() for k in row}, key=lambda k: (k not in PROCESS_ORDER, k))
    rows = [["true \\ inferred", *labels, "accuracy"]]
    for p in PROCESS_ORDER:
        row = block["confusion"].get(p, {})
        rows.append([p, *[f"{row.get(k, 0.0):.0%}" for k in labels], f"{block['accuracy'][p]:.0%}"])
    return rows


def to_markdown(card: dict) -> str:
    ds = card["dataset"]
    out = ["# memetrics scorecard", "",
           f"{ds['n_items']:,} items, {ds['n_authors']:,} authors, {ds['n_communities']:,} communities, "
           f"{ds['start'][:10]} to {ds['end'][:10]}. {ds['n_units']:,} units cover {ds['share_in_units']:.0%} of "
           f"items; {ds['share_with_parent_in_table']:.0%} of items have a parent in the table.", "",
           "Each measure has two readings. **Literal**: the default decision rule on the raw estimate. "
           "**Calibrated**: the estimate against a simulated null of the same size and structure. When they "
           "disagree, the calibrated reading is the one that accounts for how the estimator behaves.", "",
           "## Measures", ""]
    for m in card["measures"]:
        out += [f"### {m['id']} {m['title']}", ""]
        if m["claim"]:
            out += [f"*Claim:* {m['claim']}", ""]
        if m["status"] != "ok":
            out += [f"Not run: {m['reason']}", ""]
            continue
        out += ["| data | estimate [95% CI] | literal rule | calibrated | n |", "|---|---|---|---|---|"]
        for r in m["rows"]:
            out.append(f"| {r['label']} | {r['estimate']} | {_cell(r['literal'])} | {_cell(r['calibrated'])} | "
                       f"{_cell(r['n'])} |")
        out += ["", f"*Literal rule:* {m['rule']}", "", f"*Calibration:* {m['calibration']}", ""]
        out += [f"- {n}" for n in m["notes"]] + ([""] if m["notes"] else [])
    out += ["## What this data cannot see", "",
            "Public posts and public reactions only. Every reading above inherits these gaps:", ""]
    out += [f"- **{b['what']}.** {b['consequence']}" for b in card["blind_spots"]]
    if ds.get("share_deleted_flag") is not None:
        out.append(f"- This table flags {ds['share_deleted_flag']:.1%} of items as deleted.")
    dz = card.get("distinguishability")
    out += [""]
    if dz:
        out += [f"### Can this design tell the processes apart? (simulated, {dz['p_private']:.0%} private shares)",
                "", f"{dz['reps']} simulated worlds per process with about {dz['mean_observed_items']:,.0f} observed "
                f"items each ({dz['n_agents']:,} agents), {'engagement-ranked' if dz['recommender'] else 'chronological'}"
                f" feeds, {dz['edge_mode']} parent links. Rows: the true process; columns: the label the composite "
                "decision rule assigns from the observed table.", ""]
        for col, title in (("inferred", "Literal rules"), ("inferred_calibrated", "Calibrated rules")):
            rows = _confusion_rows(dz[col])
            out += [f"**{title}** (balanced accuracy {dz[col]['balanced_accuracy']:.0%})", "",
                    "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
            out += ["| " + " | ".join(r) + " |" for r in rows[1:]] + [""]
    else:
        out += ["To see whether these readings could separate replication, attraction, drift and contagion at this "
                "dataset's size, rerun with `--p-private P` (an assumed share of private reshares).", ""]
    out += ["---", f"memetrics {card['memetrics_version']}, seed {card['seed']}, {card['created']}.", ""]
    return "\n".join(out)


CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--accent:#2f5d8a;--warn:#8a5a00;
--warnbg:#fff6e0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--fg:#ecebe6;--muted:#a3a29b;
--line:#33332f;--card:#1d1d1b;--accent:#8cb4dc;--warn:#f0c46a;--warnbg:#2a2312}}
:root[data-theme="dark"]{--bg:#151514;--fg:#ecebe6;--muted:#a3a29b;--line:#33332f;--card:#1d1d1b;--accent:#8cb4dc;
--warn:#f0c46a;--warnbg:#2a2312}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.2rem;margin:2em 0 .6em}h3{font-size:1.05rem;margin:0 0 .3em}
.muted{color:var(--muted)}section.m{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px;margin:14px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums}
.blind{background:var(--warnbg);border:1px solid var(--warn);border-radius:10px;padding:16px;margin:24px 0}
.blind h2{margin-top:0;color:var(--warn)}code{font-size:.9em}
"""


def to_html(card: dict) -> str:
    e = html.escape
    ds = card["dataset"]
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
             f"content='width=device-width,initial-scale=1'><title>memetrics scorecard</title><style>{CSS}</style>"
             "</head><body><main>", "<h1>memetrics scorecard</h1>",
             f"<p class=muted>{ds['n_items']:,} items · {ds['n_authors']:,} authors · {ds['n_communities']:,} "
             f"communities · {e(ds['start'][:10])} to {e(ds['end'][:10])} · {ds['n_units']:,} units "
             f"({ds['share_in_units']:.0%} of items) · {ds['share_with_parent_in_table']:.0%} with a parent in the "
             "table</p>",
             "<p><b>Literal</b>: the default decision rule on the raw estimate. <b>Calibrated</b>: the estimate "
             "against a simulated null of the same size and structure.</p>"]
    for m in card["measures"]:
        parts.append(f"<section class=m><h3>{e(m['id'])} {e(m['title'])}</h3>")
        if m["claim"]:
            parts.append(f"<p class=muted>{e(m['claim'])}</p>")
        if m["status"] != "ok":
            parts.append(f"<p>Not run: {e(m['reason'] or '')}</p></section>")
            continue
        parts.append("<div class=scroll><table><tr><th>data</th><th>estimate [95% CI]</th><th>literal rule</th>"
                     "<th>calibrated</th><th>n</th></tr>")
        for r in m["rows"]:
            parts.append(f"<tr><td>{e(r['label'])}</td><td class=num>{e(r['estimate'])}</td>"
                         f"<td>{e(_cell(r['literal']))}</td><td>{e(_cell(r['calibrated']))}</td>"
                         f"<td class=num>{e(_cell(r['n']))}</td></tr>")
        parts.append("</table></div>")
        parts.append(f"<p class=muted><b>Literal rule:</b> {e(m['rule'])}<br><b>Calibration:</b> "
                     f"{e(m['calibration'])}</p>")
        if m["notes"]:
            parts.append("<ul class=muted>" + "".join(f"<li>{e(n)}</li>" for n in m["notes"]) + "</ul>")
        parts.append("</section>")
    parts.append("<div class=blind><h2>What this data cannot see</h2><p>Public posts and public reactions only. "
                 "Every reading above inherits these gaps.</p><ul>")
    parts += [f"<li><b>{e(b['what'])}.</b> {e(b['consequence'])}</li>" for b in card["blind_spots"]]
    if ds.get("share_deleted_flag") is not None:
        parts.append(f"<li>This table flags {ds['share_deleted_flag']:.1%} of items as deleted.</li>")
    parts.append("</ul>")
    dz = card.get("distinguishability")
    if dz:
        parts.append(f"<h3>Can this design tell the processes apart? ({dz['p_private']:.0%} private shares, "
                     "simulated)</h3>"
                     f"<p>{dz['reps']} worlds per process, about {dz['mean_observed_items']:,.0f} observed items each. "
                     "Rows: true process; columns: label assigned by the composite rule.</p>")
        for col, title in (("inferred", "Literal rules"), ("inferred_calibrated", "Calibrated rules")):
            rows = _confusion_rows(dz[col])
            parts.append(f"<p><b>{title}</b> · balanced accuracy {dz[col]['balanced_accuracy']:.0%}</p>"
                         "<div class=scroll><table>")
            parts.append("<tr>" + "".join(f"<th>{e(c)}</th>" for c in rows[0]) + "</tr>")
            parts += ["<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows[1:]]
            parts.append("</table></div>")
    else:
        parts.append("<p>Rerun with <code>--p-private P</code> to simulate whether this design can separate "
                     "replication, attraction, drift and contagion.</p>")
    parts.append(f"</div><p class=muted>memetrics {e(card['memetrics_version'])} · seed {card['seed']} · "
                 f"{e(card['created'])}</p></main></body></html>")
    return "\n".join(parts)


def write(card: dict, out_dir: str | Path) -> dict[str, Path]:
    """Write ``scorecard.json``, ``report.md`` and ``report.html`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "scorecard.json", "md": out / "report.md", "html": out / "report.html"}
    paths["json"].write_text(json.dumps(card, indent=1) + "\n")
    paths["md"].write_text(to_markdown(card))
    paths["html"].write_text(to_html(card))
    return paths
