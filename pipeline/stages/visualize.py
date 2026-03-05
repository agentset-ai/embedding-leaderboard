"""
Visualization stage: Generate plots and HTML report
"""

import json
from pathlib import Path
from typing import Dict
import logging

from ..config import Config
from ..paths import RunPaths


def visualize_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    Generate visualizations and HTML report

    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance

    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting visualization stage...")

    results = {
        'status': 'completed',
    }

    print("\nVisualization stage")
    print("   Generating HTML report...")

    # Generate HTML report
    report_file = paths.get_report_file()
    html = generate_simple_report(config, paths)

    with open(report_file, 'w') as f:
        f.write(html)

    print(f"   HTML report saved to {report_file}")
    logger.info(f"HTML report saved to {report_file}")

    results['report_file'] = str(report_file)

    logger.info("Visualization stage completed")
    return results


def generate_simple_report(config: Config, paths: RunPaths) -> str:
    """Generate a simple HTML report"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Embedder Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { font-weight: bold; }
    </style>
</head>
<body>
    <h1>Embedder Evaluation Report</h1>
    <p>Run: """ + paths.timestamp + """</p>

    <h2>Global Leaderboard</h2>
"""

    # Load and display global leaderboard
    leaderboard_file = paths.get_global_leaderboard_file()
    if leaderboard_file.exists():
        with open(leaderboard_file, 'r') as f:
            leaderboard = json.load(f)

        html += "<table><tr><th>Rank</th><th>Model</th>"
        for k in config.evaluation.k_values:
            html += f"<th>NDCG@{k}</th><th>Recall@{k}</th>"
        html += "<th>Avg Latency</th><th>Datasets</th></tr>"

        for rank, entry in enumerate(leaderboard, 1):
            html += f"<tr><td>{rank}</td><td>{entry['model']}</td>"
            for k in config.evaluation.k_values:
                html += f"<td>{entry.get(f'avg_ndcg@{k}', 0):.4f}</td>"
                html += f"<td>{entry.get(f'avg_recall@{k}', 0):.4f}</td>"
            html += f"<td>{entry['avg_latency']:.4f}s</td>"
            html += f"<td>{entry['num_datasets']}</td></tr>"

        html += "</table>"

    # Per-dataset results
    html += "<h2>Per-Dataset Results</h2>"
    for dataset in config.datasets:
        comparison_file = paths.get_comparison_file(dataset.name)
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                comparison = json.load(f)

            html += f"<h3>{dataset.name}</h3>"
            html += "<table><tr><th>Model</th>"
            for k in config.evaluation.k_values:
                html += f"<th>NDCG@{k}</th><th>Recall@{k}</th>"
            html += "<th>Latency</th><th>Elo</th></tr>"

            for result in comparison['results']:
                html += f"<tr><td>{result['model']}</td>"
                for k in config.evaluation.k_values:
                    html += f"<td>{result.get(f'ndcg@{k}', 0):.4f}</td>"
                    html += f"<td>{result.get(f'recall@{k}', 0):.4f}</td>"
                html += f"<td>{result['avg_query_latency']:.4f}s</td>"
                html += f"<td>{comparison['elo_scores'].get(result['model'], 1500):.0f}</td></tr>"

            html += "</table>"

    html += """
</body>
</html>
"""

    return html
