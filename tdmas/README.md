# Task Decomposition based Multi-Agent System (TDMAS)

## Project Introduction

This project implements an automated design method for Multi-Agent Systems (MAS) based on task decomposition. The method has the following characteristics:

- **Multi-dataset transfer**: Supports transfer across different datasets
- **Independent of training sample size**: Generates rich training samples through slight perturbations of agents
- **Simple design**: Simple and general framework
- **Independent of labeled data**: Evaluates internal consistency rather than overly relying on ground-truth

## Core Method

This MAS construction method adopts a recursive design, consisting of 3 main elements:

1. **Question**: Pass a question to the leader agent l
2. **Decomposition**: The leader agent l decomposes the question into n sub-questions and passes them to employee agents {e1, e2, …, en}
3. **Reply**: After receiving answers from employee agents, the leader agent l returns the final answer

The method evaluates MAS internal consistency through a bidirectional scoring mechanism between "leader" and "employees", rather than overly relying on labeled data.

## File Structure

```
tdmas/
├── __init__.py           # Module initialization
├── prompts.py            # Prompt template definitions
├── parser.py             # Parse agent outputs
├── mas.py                # Multi-agent system core implementation
├── loss.py               # Loss calculation module
├── data_collector.py     # Data collection module
├── preference_data.py    # Preference data generation
├── trainer.py            # DPO training module
├── evaluator.py          # Evaluation module
└── main.py               # Main program entry
```

## Usage

### 1. Configuration File Preparation

Ensure the following configuration files exist:

- `config/running_config.yaml`: Runtime configuration
- `config/local_llm.yaml`: LLM model configuration
- `config/optimize_config.yaml`: Optimization configuration (copy from optimize_config.yaml.example and modify)

### 2. Run Training

```bash
python -m tdmas.main --config config/running_config.yaml
```

### 3. Configuration Description

Main parameters in `config/running_config.yaml`:

```yaml
dataset: GSM8K              # Dataset name
task_type: full_pipeline    # full_pipeline, infer_only, train_only
start_epoch: 0              # Start epoch
end_epoch: 3                # End epoch
model_name: Qwen3-8B        # Model name
limit: 16                   # Limit number of samples (for rapid testing)
max_depth: 5                # Maximum recursion depth
max_concurrent: 10          # Maximum concurrency
```

## Workflow

Each epoch consists of the following steps:

1. **Training Data Collection**: Use MAS system to collect data on the training set and calculate loss (correctness + scores + token consumption)
2. **Generate Preference Data**: Generate preference pairs required for DPO training from collected data
3. **Model Optimization**: Optimize the model using DPO method
4. **Test Set Validation**: Validate model performance on the test set

## Loss Design

The loss function consists of three parts:

- **Correctness Loss**: The degree of conformity between the final MAS answer and ground-truth
- **Consistency Loss**: Sum of all scores (self-consistency)
- **Token Loss**: Token consumption in Q&A (cost control)

Total Loss = Correctness Loss × Weight1 + Consistency Loss × Weight2 + Token Loss

## Notes

1. Ensure dataset files exist in the `data/` directory
2. Ensure model path configuration is correct
3. First run requires creating the `config/optimize_config.yaml` file
4. Training process generates many intermediate files, pay attention to disk space
