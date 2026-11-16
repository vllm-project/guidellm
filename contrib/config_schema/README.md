# Configuration schema

Export the JSON schema of GuideLLM's `BenchmarkScenario` Pydantic model. Optionally render it as browsable HTML documentation using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans).

## Setup

Create a environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate

# Optionally install the local version of GuideLLM
# pip install -e ../..
pip install -r requirements.txt
```

The schema reflects the installed version of GuideLLM.

## Usage

```bash
# Write benchmark_scenario.schema.json in the current directory
python config_schema.py

# Choose output paths and also generate HTML documentation
python config_schema.py \
    --output schema/benchmark_scenario.schema.json \
    --html schema/benchmark_scenario.html
```

Output directories are created automatically.
