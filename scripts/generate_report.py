import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def _bootstrap_env_from_argv() -> None:
  if "--env" not in sys.argv:
    return
  idx = sys.argv.index("--env")
  if idx + 1 >= len(sys.argv):
    return
  load_dotenv(sys.argv[idx + 1], override=True)


_bootstrap_env_from_argv()

from config.settings import SQLITE_DB_PATH


def _query_all(db_path: str, query: str, params=()):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _query_one(db_path: str, query: str, params=()):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    return row


def build_report_data(db_path: str) -> dict:
    total_papers = _query_one(db_path, "SELECT COUNT(*) FROM papers")[0]
    total_clusters = _query_one(db_path, "SELECT COUNT(*) FROM cluster_progress")[0]
    total_pairs = _query_one(db_path, "SELECT COUNT(*) FROM candidate_pairs")[0]
    total_contradictions = _query_one(db_path, "SELECT COUNT(*) FROM contradictions")[0]
    total_flagged = _query_one(db_path, "SELECT COUNT(*) FROM nli_results WHERE ensemble_label = 'flagged'")[0]
    total_skipped = _query_one(db_path, "SELECT COUNT(*) FROM candidate_pairs WHERE status = 'skipped'")[0]

    error_clusters = _query_all(
        db_path,
        "SELECT cluster_id FROM cluster_progress WHERE status = 'error'",
    )
    error_cluster_ids = [row["cluster_id"] for row in error_clusters]

    latest_update = _query_one(
        db_path,
        "SELECT MAX(updated_at) FROM cluster_progress",
    )[0]

    type_counts = _query_all(
        db_path,
        "SELECT contradiction_type, COUNT(*) as count FROM contradictions GROUP BY contradiction_type",
    )

    cluster_summary = _query_all(
        db_path,
        """
        SELECT cp.cluster_id, cp.paper_count, cp.pair_count, cp.status,
               (SELECT COUNT(*) FROM contradictions c JOIN candidate_pairs p ON c.pair_id = p.pair_id WHERE p.cluster_id = cp.cluster_id) AS contradiction_count,
               (SELECT COUNT(*) FROM nli_results n JOIN candidate_pairs p ON n.pair_id = p.pair_id WHERE p.cluster_id = cp.cluster_id AND n.ensemble_label = 'flagged') AS flagged_count
        FROM cluster_progress cp
        """,
    )

    contradictions = _query_all(
        db_path,
        """
        SELECT c.contradiction_type, c.explanation, p.cluster_id,
               a.title AS anchor_title, a.paper_id AS anchor_id,
               b.title AS challenger_title, b.paper_id AS challenger_id,
               cl.claim_text, e.evidence_text, n.llama_confidence, n.bloomz_label, n.llama_label
        FROM contradictions c
        JOIN candidate_pairs p ON c.pair_id = p.pair_id
        JOIN papers a ON p.anchor_id = a.paper_id
        JOIN papers b ON p.challenger_id = b.paper_id
        JOIN claims cl ON c.claim_id = cl.claim_id
        JOIN evidence e ON c.evidence_id = e.evidence_id
        JOIN nli_results n ON n.claim_id = c.claim_id AND n.evidence_id = c.evidence_id
        """,
    )

    flagged_pairs = _query_all(
        db_path,
        """
        SELECT p.cluster_id,
               a.title AS anchor_title, a.paper_id AS anchor_id,
               b.title AS challenger_title, b.paper_id AS challenger_id,
               cl.claim_text, e.evidence_text, n.bloomz_label, n.llama_label
        FROM nli_results n
        JOIN candidate_pairs p ON n.pair_id = p.pair_id
        JOIN papers a ON p.anchor_id = a.paper_id
        JOIN papers b ON p.challenger_id = b.paper_id
        JOIN claims cl ON n.claim_id = cl.claim_id
        JOIN evidence e ON n.evidence_id = e.evidence_id
        WHERE n.ensemble_label = 'flagged'
        """,
    )

    neutral_support_samples = _query_all(
        db_path,
        """
        SELECT n.ensemble_label, a.title AS anchor_title, a.paper_id AS anchor_id,
               b.title AS challenger_title, b.paper_id AS challenger_id,
               cl.claim_text, e.evidence_text
        FROM nli_results n
        JOIN candidate_pairs p ON n.pair_id = p.pair_id
        JOIN papers a ON p.anchor_id = a.paper_id
        JOIN papers b ON p.challenger_id = b.paper_id
        JOIN claims cl ON n.claim_id = cl.claim_id
        JOIN evidence e ON n.evidence_id = e.evidence_id
        WHERE n.ensemble_label IN ('neutral', 'support')
        ORDER BY RANDOM()
        LIMIT 20
        """,
    )

    return {
        "summary": {
            "total_papers": total_papers,
            "total_clusters": total_clusters,
            "total_pairs": total_pairs,
            "total_contradictions": total_contradictions,
            "total_flagged": total_flagged,
            "total_skipped": total_skipped,
            "error_clusters": error_cluster_ids,
            "latest_update": latest_update,
        },
        "type_counts": type_counts,
        "clusters": cluster_summary,
        "contradictions": contradictions,
        "flagged": flagged_pairs,
        "neutral_support_samples": neutral_support_samples,
    }


def render_html(report_data: dict) -> str:
    date_str = report_data["summary"]["latest_update"] or datetime.utcnow().isoformat()
    data_json = json.dumps(report_data)

    return f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>Contradiction Pipeline Report</title>
<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
<style>
body {{ font-family: Arial, sans-serif; color: #111; background: #fff; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
.badge {{ padding: 4px 8px; border-radius: 6px; color: #fff; font-size: 12px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
.table {{ width: 100%; border-collapse: collapse; }}
.table th, .table td {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; }}
.note {{ background: #f5f5f5; padding: 12px; border-radius: 8px; margin-bottom: 20px; }}
.pager {{ display: flex; gap: 8px; margin: 12px 0; }}
</style>
</head>
<body>
<div class=\"container\">
  <div class=\"note\">Generated from pipeline_state.db on {date_str}. Flagged pairs have not been human-reviewed.</div>

  <h2>Pipeline Summary</h2>
  <table class=\"table\">
    <tr><th>Total papers</th><td id=\"totalPapers\"></td></tr>
    <tr><th>Total clusters</th><td id=\"totalClusters\"></td></tr>
    <tr><th>Total pairs</th><td id=\"totalPairs\"></td></tr>
    <tr><th>Total contradictions</th><td id=\"totalContradictions\"></td></tr>
    <tr><th>Total flagged</th><td id=\"totalFlagged\"></td></tr>
    <tr><th>Total skipped</th><td id=\"totalSkipped\"></td></tr>
    <tr><th>Error clusters</th><td id=\"errorClusters\"></td></tr>
  </table>

  <h2>Contradiction Type Breakdown</h2>
  <canvas id=\"typeChart\" height=\"120\"></canvas>

  <h2>Per-Cluster Summary</h2>
  <table class=\"table\" id=\"clusterTable\">
    <thead>
      <tr>
        <th data-key=\"cluster_id\">Cluster ID</th>
        <th data-key=\"paper_count\">Paper count</th>
        <th data-key=\"pair_count\">Pair count</th>
        <th data-key=\"contradiction_count\">Contradictions</th>
        <th data-key=\"flagged_count\">Flagged</th>
        <th data-key=\"status\">Status</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <h2>Contradiction Browser</h2>
  <div class=\"pager\">
    <button id=\"prevContradictions\">Previous</button>
    <button id=\"nextContradictions\">Next</button>
  </div>
  <div id=\"contradictionCards\"></div>

  <h2>Flagged Pairs Browser</h2>
  <div class=\"pager\">
    <button id=\"prevFlagged\">Previous</button>
    <button id=\"nextFlagged\">Next</button>
  </div>
  <div id=\"flaggedCards\"></div>

  <h2>Sample Neutral/Support Pairs</h2>
  <details>
    <summary>Show samples</summary>
    <div id=\"sampleCards\"></div>
  </details>
</div>

<script>
const REPORT_DATA = {data_json};
const typeColors = {{
  direct_factual: '#d32f2f',
  methodological: '#7b1fa2',
  conditional: '#1976d2',
  interpretive: '#388e3c',
  ontological: '#f57c00'
}};

function initSummary() {{
  const s = REPORT_DATA.summary;
  document.getElementById('totalPapers').textContent = s.total_papers;
  document.getElementById('totalClusters').textContent = s.total_clusters;
  document.getElementById('totalPairs').textContent = s.total_pairs;
  document.getElementById('totalContradictions').textContent = s.total_contradictions;
  document.getElementById('totalFlagged').textContent = s.total_flagged;
  document.getElementById('totalSkipped').textContent = s.total_skipped;
  document.getElementById('errorClusters').textContent = s.error_clusters.join(', ');
}}

function initChart() {{
  const labels = REPORT_DATA.type_counts.map(t => t.contradiction_type || 'unknown');
  const values = REPORT_DATA.type_counts.map(t => t.count);
  const colors = labels.map(l => typeColors[l] || '#999');
  new Chart(document.getElementById('typeChart'), {{
    type: 'bar',
    data: {{ labels, datasets: [{{ data: values, backgroundColor: colors }}] }},
    options: {{ indexAxis: 'y' }}
  }});
}}

function initClusterTable() {{
  const tbody = document.querySelector('#clusterTable tbody');
  const rows = REPORT_DATA.clusters;
  let sortKey = 'cluster_id';
  let ascending = true;

  function render() {{
    tbody.innerHTML = '';
    const sorted = [...rows].sort((a, b) => {{
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === 'number' && typeof bv === 'number') {{
        return ascending ? av - bv : bv - av;
      }}
      return ascending ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    }});
    for (const row of sorted) {{
      const tr = document.createElement('tr');
      const status = row.status || 'pending';
      const color = status === 'done' ? '#2e7d32' : status === 'error' ? '#c62828' : '#f9a825';
      tr.innerHTML = `
        <td>${{row.cluster_id}}</td>
        <td>${{row.paper_count || 0}}</td>
        <td>${{row.pair_count || 0}}</td>
        <td>${{row.contradiction_count || 0}}</td>
        <td>${{row.flagged_count || 0}}</td>
        <td style="color:${{color}}">${{status}}</td>
      `;
      tbody.appendChild(tr);
    }}
  }}

  document.querySelectorAll('#clusterTable th').forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.getAttribute('data-key');
      if (key === sortKey) {{
        ascending = !ascending;
      }} else {{
        sortKey = key;
        ascending = true;
      }}
      render();
    }});
  }});

  render();
}}

function paginate(items, containerId, prevId, nextId) {{
  let index = 0;
  const pageSize = 20;
  const container = document.getElementById(containerId);
  const prev = document.getElementById(prevId);
  const next = document.getElementById(nextId);

  function render() {{
    container.innerHTML = '';
    const page = items.slice(index, index + pageSize);
    for (const item of page) {{
      const card = document.createElement('div');
      card.className = 'card';
      const badgeColor = typeColors[item.contradiction_type] || '#777';
      const badge = item.contradiction_type ? `<span class="badge" style="background:${{badgeColor}}">${{item.contradiction_type}}</span>` : '';
      card.innerHTML = `
        <div>${{badge}}</div>
        <div><strong>Cluster:</strong> ${{item.cluster_id || ''}}</div>
        <div><strong>Anchor:</strong> <a target="_blank" href="https://arxiv.org/abs/${{item.anchor_id}}">${{item.anchor_title}} (${{item.anchor_id}})</a></div>
        <div><strong>Challenger:</strong> <a target="_blank" href="https://arxiv.org/abs/${{item.challenger_id}}">${{item.challenger_title}} (${{item.challenger_id}})</a></div>
        <div><strong>Claim:</strong> ${{item.claim_text}}</div>
        <div><strong>Evidence:</strong> ${{item.evidence_text}}</div>
        ${item.explanation ? `<div><strong>Explanation:</strong> ${{item.explanation}}</div>` : ''}
        ${item.llama_confidence ? `<div><strong>Confidence:</strong> ${{Math.round(item.llama_confidence * 100)}}%</div>` : ''}
        ${item.bloomz_label ? `<div><strong>Bloomz:</strong> ${{item.bloomz_label}} | <strong>Llama:</strong> ${{item.llama_label}}</div>` : ''}
      `;
      container.appendChild(card);
    }}
  }}

  prev.addEventListener('click', () => {{
    index = Math.max(0, index - pageSize);
    render();
  }});
  next.addEventListener('click', () => {{
    index = Math.min(items.length - pageSize, index + pageSize);
    render();
  }});

  render();
}}

function renderSamples() {{
  const container = document.getElementById('sampleCards');
  for (const item of REPORT_DATA.neutral_support_samples) {{
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div><strong>Label:</strong> ${{item.ensemble_label}}</div>
      <div><strong>Anchor:</strong> <a target="_blank" href="https://arxiv.org/abs/${{item.anchor_id}}">${{item.anchor_title}} (${{item.anchor_id}})</a></div>
      <div><strong>Challenger:</strong> <a target="_blank" href="https://arxiv.org/abs/${{item.challenger_id}}">${{item.challenger_title}} (${{item.challenger_id}})</a></div>
      <div><strong>Claim:</strong> ${{item.claim_text}}</div>
      <div><strong>Evidence:</strong> ${{item.evidence_text}}</div>
    `;
    container.appendChild(card);
  }}
}}

initSummary();
initChart();
initClusterTable();
paginate(REPORT_DATA.contradictions, 'contradictionCards', 'prevContradictions', 'nextContradictions');
paginate(REPORT_DATA.flagged, 'flaggedCards', 'prevFlagged', 'nextFlagged');
renderSamples();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    load_dotenv(args.env, override=True)

    data = build_report_data(SQLITE_DB_PATH)
    html = render_html(data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(html)

    print("Report written to", args.output)


if __name__ == "__main__":
    main()
