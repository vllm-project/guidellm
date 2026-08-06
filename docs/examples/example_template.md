# [Title: What the user is trying to accomplish]

[One or two sentences describing the scenario and why someone would need this.]

## Prerequisites

- A running OpenAI-compatible server ([setup guide](../getting-started/server.md))
- GuideLLM installed ([install guide](../getting-started/install.md))
- [Any additional prerequisites specific to this example]

## Step 1: [Setup / Configure]

[Brief explanation of what this step does and why.]

```bash
guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --data kind=synthetic_text,prompt_tokens=___,output_tokens=___ \
  --profile kind=___,___=___ \
  --constraint kind=max_duration,seconds=___ \
  --seed kind=static,value=42 \
  --output kind=json,path=___.json
```

[Explain what each non-obvious flag does in the context of this example. Don't repeat what the getting-started docs already cover. **Focus on why these specific values were chosen** for this use case.]

## Step 2: [Run / Execute]

[If the example requires multiple runs (e.g. comparing two configs), show the second command here. If it's a single run, this step is about what happens during execution — what to watch for in the console output.]

## Step 3: [Interpret the Results]

[Explain how to interpret the metrics/results, highlighting what to take away (good and the bad)]

| Metric | What to look for |
|--------|-----------------|
| `metric_name` | [What a good/bad value means for this use case] |
| `metric_name` | [What a good/bad value means for this use case] |

## Step 4: [Make a Decision / Take Action]

[How to go from the numbers to a concrete deployment decision. This separates an example from a reference doc.]

## Example Output

[A realistic (ideally real) table or snippet showing what the results look like, with annotations pointing out the key takeaway.]

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| ... | ... | ... |

[One or two sentences summarizing the conclusion from this example data.]

## Next Steps
[The user finds other helpful guides]
- [Link to relevant guide](../guides/relevant_guide.md) for deeper coverage of a feature used here
- [Link to another example](another_example.md) for a related workflow