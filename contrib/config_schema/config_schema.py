#!/usr/bin/env python3
"""Export the GuideLLM benchmark configuration schema and optional HTML docs."""

import argparse
import json
from pathlib import Path

from json_schema_for_humans.generate import generate_from_filename
from json_schema_for_humans.generation_configuration import GenerationConfiguration

from guidellm.schemas.benchmark import BenchmarkScenario

BASE_NAME = "benchmark_scenario"


def main() -> None:
    """Write the JSON schema and render it as HTML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./schema"),
        help="Output directory for the schema",
    )
    parser.add_argument(
        "--html",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate HTML documentation at output path",
    )
    args = parser.parse_args()

    schema = BenchmarkScenario.model_json_schema()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_schema_path = args.output_dir / f"{BASE_NAME}.schema.json"

    with json_schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"JSON schema written to {json_schema_path}")

    if args.html:
        html_path = args.output_dir / f"{BASE_NAME}.html"
        generate_from_filename(
            json_schema_path,
            str(html_path),
            config=GenerationConfiguration(
                template_name="js",
            ),
        )
        print(f"HTML documentation written to {html_path}")


if __name__ == "__main__":
    main()
