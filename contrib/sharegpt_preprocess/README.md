# ShareGPT Datasets

You can use ShareGPT_V3_unfiltered_cleaned_split.json as benchmark datasets.

## Prerequisites
Before you begin, ensure you have the following installed:

* Python 3.9 or higher
* pip (Python package manager)

## Example Commands

Download and prepare the ShareGPT dataset; You can specify the proportion of data to process by providing a number between 0 and 1 as an argument to the script.

```bash
cd contrib/sharegpt_preprocess
pip install -r requirements.txt
bash prepare_sharegpt_data.sh 1

```

In this example, 1 indicates processing 100% of the dataset. You can adjust this value as needed. Conda env is Recommanded to install libs.

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --rate-type "throughput" \
  --data-args '{"prompt_column": "value", "split": "train"}' \
  --max-requests 10 \
  --data "/${local_path}/ShareGPT.json"
```
