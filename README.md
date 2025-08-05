# CSLParser: A Collaborative Framework Using Small and Large Language Models for Log Parsing

## Quick Start

### Step 1: Download Data and Models
```bash
cd datasets/loghub-2.0-full
./download.sh
cd ../../code/models
./clone_roberta.sh
```

### Step 2: Install Dependencies
Install python >= 3.11
```bash
pip install -r requirements.txt
```

### Step 3: Configure and Run
1. Edit `code/llm_query.py` to set your `MODEL`, `API_KEY`, and `BASE_URL`
2. Run the framework:
```bash
cd code
./run.sh
``` 