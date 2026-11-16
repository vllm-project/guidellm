# Extract conversations

Pretty-print conversation history and responses from the first benchmark in a GuideLLM JSON report. Tool calls are displayed as YAML, and multi-agent benchmarks include agent and conversation graph context.

## Usage

Run either from within the GuideLLM Python environment or using the included `requirements.txt`

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python ./extract_conversation.py benchmarks.json
```

The input path defaults to `benchmarks.json` in the current directory. By default, the script prints up to 15 successful requests, sorted by message count.

- `--limit` / `-n`: maximum number of requests to print.
- `--turn` / `-t`: print only the request at the given 1-based index.
- `--detail` / `-d`: choose `request` (metadata only), `roles` (metadata and role markers), or `full` (default, including message content and responses).

```bash
python ./extract_conversation.py benchmarks.json --limit 5
python ./extract_conversation.py benchmarks.json --turn 3 --detail roles
```
