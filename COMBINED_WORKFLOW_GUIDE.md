# Combined Workflow Guide: From Start to Finish

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Training/Optimization Phase](#trainingoptimization-phase)
4. [Testing Phase](#testing-phase)
5. [How Rounds Work](#how-rounds-work)
6. [File Structure and Evolution](#file-structure-and-evolution)
7. [Key Concepts](#key-concepts)
8. [Troubleshooting](#troubleshooting)

## Overview

The Combined workflow is an AI system that learns to solve three types of problems:
- **Math problems** (mathematical reasoning)
- **MMLU problems** (multiple choice questions)
- **HumanEval problems** (coding challenges)

The system works in two main phases:
1. **Training/Optimization**: The AI learns and improves over multiple rounds
2. **Testing**: The best performing version is tested on a separate dataset

Think of it like training a student - first they practice and improve (optimization), then they take a final exam (testing).

## Prerequisites

Before you start, make sure you have:

1. **Python 3.8+** installed
2. **Required packages** installed (run `pip install -r requirements.txt`)
3. **API keys** configured in `config/config2.yaml`:
   ```yaml
   gemini-2.0-flash:
     api_type: "gemini"
     base_url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
     api_key: "your_api_key_here"
   ```
4. **Dataset files** in the `data/` directory:
   - `combined_train.jsonl` (for training)
   - `combined_validate.jsonl` (for testing)

## Training/Optimization Phase

### Step 1: Start the Optimization Process

Open the `run.py` file and modify the bottom section to start training:

```python
# At the bottom of run.py, around line 140-151, change it to:

    # Optimize workflow via setting the optimizer's mode to 'Graph'
    optimizer.optimize("Train")

    # Test workflow via setting the optimizer's mode to 'Test'
    # optimizer.optimize("Test")  # Comment this out during training
```

Then run:
```bash
python run.py --dataset Combined --sample 1
```

**What this does:**
- `--dataset Combined`: Tells the system to work with the Combined dataset
- `--sample 2`: Uses 2 samples per round (you can increase this for better results but it takes longer)
- The `optimizer.optimize("Train")` line starts the training/optimization process

### Step 2: Understanding the Optimization Process

The system will:

1. **Create the initial workflow** in `workspace/Combined/workflows/round_1/`
2. **Run the first round** and get a score
3. **Analyze the results** and generate improvements
4. **Create round_2** with the improved workflow
5. **Repeat** until scores stop improving significantly

### Step 3: Monitor Progress

Watch the console output for:
- Round scores (higher is better)
- Cost information (how much money was spent on API calls)
- Convergence messages

Example output:
```
Round 1 Score: 0.65
Round 2 Score: 0.72
Round 3 Score: 0.74
Round 4 Score: 0.75
Convergence detected - stopping optimization
```

## Testing Phase

### Step 1: Identify the Best Round

After optimization completes, check the scores in the logs or look at the CSV files in `workspace/Combined/logs/`. Find the round with the highest score.

### Step 2: Copy Best Round to Test Directory

```bash
# Example: if round_3 had the best score
cp -r workspace/Combined/workflows/round_3 workspace/Combined/workflows_test/
```

### Step 3: Run Testing

Open the `run.py` file and modify the bottom section to run testing:

```python
# At the bottom of run.py, around line 140-151, change it to:

    # Optimize workflow via setting the optimizer's mode to 'Graph'
    # optimizer.optimize("Train")  # Comment this out during testing

    # Test workflow via setting the optimizer's mode to 'Test'
    optimizer.optimize("Test")
```

Then run:
```bash
python run.py --dataset Combined --sample 400
```

**What this does:**
- `--sample 400`: Tests on all 400 questions in the validation dataset
- The `optimizer.optimize("Test")` line runs the testing phase

## How Rounds Work

### Round Structure

Each round is a folder containing:
```
workspace/Combined/workflows/round_X/
├── graph.py          # The workflow logic
├── prompt.py         # Custom prompts for this round
└── operator.py       # Operator definitions (copied from template)
```

### Round Evolution Process

1. **Round 1**: Starts with a basic Chain-of-Thought (CoT) approach
2. **Round 2+**: The AI analyzes the previous round's performance and makes improvements
3. **Convergence**: When scores stop improving significantly, optimization stops

### Important Rules

⚠️ **NEVER DELETE ROUND_1** - It's the foundation that all other rounds build upon.

⚠️ **Don't manually edit round folders** during optimization - the AI manages them automatically.

⚠️ **If retrying optimization**: Delete all rounds except round_1 to start fresh:
   ```bash
   # Keep round_1, delete all others
   cd workspace/Combined/workflows/
   rm -rf round_2 round_3 round_4 round_5 round_6 round_7 round_8 round_9 round_10
   # Or more safely, move them to a backup folder
   mkdir backup_old_rounds
   mv round_2 round_3 round_4 round_5 round_6 round_7 round_8 round_9 round_10 backup_old_rounds/ 2>/dev/null || true
   
   # Also delete processed experience and results files
   rm -f processed_experience.json
   rm -f results.json
   ```

## File Structure and Evolution

### Initial Setup (Round 1)

**graph.py** (Chain-of-Thought only):
```python
async def __call__(self, problem: str, entry_point: str = ""):
    # HumanEval path (coding problems)
    if entry_point:
        code = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        return code.get('response', ''), self.llm.get_usage_summary()["total_cost"]

    # Non-code path (Math/MMLU): use Custom with per-round prompt_custom
    resp = await self.custom(input=problem, instruction="")
    return resp.get('response', ''), self.llm.get_usage_summary()["total_cost"]
```

**prompt.py** (Empty with comment):
```python
# Initial prompt - will be populated by the model during optimization
```

### How the AI Improves Each Round

After each round, the AI:

1. **Analyzes performance** on each problem type (Math, MMLU, HumanEval)
2. **Identifies weaknesses** (e.g., "Math problems need better step-by-step reasoning")
3. **Generates improvements** to `graph.py` and `prompt.py`
4. **Creates the next round** with these improvements

### Example Evolution

**Round 1**: Basic CoT approach
**Round 2**: Adds specialized prompts for Math problems
**Round 3**: Adds ensemble voting for MMLU problems
**Round 4**: Refines the ensemble approach
**Round 5**: Scores converge - optimization complete

## Key Concepts

### Chain-of-Thought (CoT)
A reasoning approach where the AI shows its work step-by-step, just like a student solving a math problem.

### Ensemble Methods
Using multiple approaches and combining their results for better accuracy.

### Convergence
When the improvement between rounds becomes very small, indicating the system has learned as much as it can.

### Cost Tracking
The system tracks how much money is spent on API calls, helping you monitor expenses.

## File Locations

### Important Directories
- `workspace/Combined/workflows/` - Training rounds (round_1, round_2, etc.)
- `workspace/Combined/workflows_test/` - Testing rounds (copied from best training round)
- `workspace/Combined/logs/` - Results and performance data
- `data/` - Dataset files

### Configuration Files
- `config/config2.yaml` - API keys and model settings
- `run.py` - Main execution script

## Troubleshooting

### Common Issues

1. **"API key not found"**
   - Check `config/config2.yaml` has your API key
   - Ensure the API key is valid and has sufficient credits

2. **"Dataset file not found"**
   - Make sure `data/combined_train.jsonl` and `data/combined_validate.jsonl` exist
   - Check file permissions

3. **"Round 1 not found"**
   - Never delete `workspace/Combined/workflows/round_1/`
   - If accidentally deleted, you'll need to restart the entire process
   - If retrying optimization, delete all other rounds but keep round_1
   - Also delete `processed_experience.json` and `results.json` when retrying

4. **Low scores**
   - Try increasing the `--sample` parameter (e.g., `--sample 5` instead of `--sample 2`)
   - Check that your API key has access to the required model
   - Ensure the dataset files are properly formatted

### Performance Tips

1. **Start small**: Use `--sample 2` for initial testing
2. **Scale up**: Use `--sample 5` or higher for better results
3. **Monitor costs**: Watch the cost output to avoid unexpected expenses
4. **Backup**: Copy your best round before testing
5. **Switch modes**: Remember to comment/uncomment the appropriate lines in `run.py` when switching between training and testing

### Expected Timeline

- **Training (sample=2)**: 30-60 minutes
- **Training (sample=5)**: 2-4 hours
- **Testing**: 10-30 minutes

## Advanced Topics

### Customizing the Process

You can modify the initial `round_1/graph.py` to start with a different approach, but remember:
- Keep it simple initially
- Let the AI do the optimization
- Don't change it after round 1 is created

### Understanding the Results

The system generates several types of output:
- **Console logs**: Real-time progress and scores
- **CSV files**: Detailed results for each round
- **JSON logs**: Detailed analysis of failures

### Model Selection

You can change the model in `config/config2.yaml`:
```yaml
# Current default
gemini-2.0-flash:
  api_type: "gemini"
  base_url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
  api_key: "your_key"

# Alternative models
gpt-4o:
  api_type: "openai"
  base_url: "https://api.openai.com/v1"
  api_key: "your_key"
```

## Summary

The Combined workflow is a sophisticated AI training system that:

1. **Starts simple** with basic Chain-of-Thought reasoning
2. **Learns iteratively** by analyzing performance and making improvements
3. **Converges automatically** when no more improvements are possible
4. **Tests thoroughly** on a separate dataset to ensure real-world performance

The key is to let the AI do the heavy lifting - you just need to:
- Set up the initial configuration
- Modify `run.py` to use `optimizer.optimize("Train")` and run the command
- Copy the best round for testing
- Modify `run.py` to use `optimizer.optimize("Test")` and run the command

The system handles all the complex optimization automatically, making it accessible even for beginners while producing sophisticated, high-performing AI workflows.
