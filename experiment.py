"""Offline artifact commands. Never starts a benchmark or downloads data."""
import argparse
from experiments.registry import import_legacy, manifests
from experiments.reports import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="Index historical JSON and related saved files")
    imp.add_argument("--evidence", help="Optional content-hash keyed historical disclosure JSON")
    sub.add_parser("list", help="List registered runs")
    report = sub.add_parser("report", help="Generate deterministic Markdown from saved runs")
    report.add_argument("--readme", help="Update only the marked generated results block")
    args = parser.parse_args()
    if args.command == "import":
        print(f"Imported {len(import_legacy(args.results_dir, args.evidence))} new runs")
    elif args.command == "report":
        print(f"Generated {generate(args.results_dir, args.readme)} reports")
    else:
        for run in manifests(args.results_dir):
            print(run["run_id"], run.get("model"), run["environment"].get("label"))


if __name__ == "__main__":
    main()
