# Sharif ML-Lab Data Generation Toolkit

This toolkit provides a comprehensive CLI for generating and evaluating images and texts using various metrics and generation methods.

## Installation

To install and run the requirments like Ollama, use Docker for easy setup:

```bash
docker compose up -d
```

## Usage

The toolkit supports various operations divided into four main spaces: metric evaluations, AI-based generation, experiments, and pipelines. Below is a guide on how to use each functionality.

### General CLI Format

```bash
./main.py --space <space_name> --method <method_name> --data <data_type> [other_options]
```

- `<space_name>`: The operational space (`metric`, `genai`, `experiment`, or `pipeline`).
- `<method_name>`: The specific method to use within the chosen space.
- `<data_type>`: The type of data to work with (`image` or `text`).

### Experiment CLI

For running experiments with image data:

```bash
./main.py --space experiment --data image --method tendency|noise --gpath <generated_path> --prompt <true_caption> --neg-prompt <false_caption>
```

### Metric CLI

Evaluate various metrics for image and text data:

```bash
./main.py --space metric --data image|text --method <metric_type> --task <metric_name> --gpath <generated_path> [--rpath <real_path>] [--cpath <caption_path>] [--model <model_name>]
```

Examples of metric evaluations include:

- **Inception Score:** `--method quality --task inception`
- **Frechet Score:** `--method quality --task frechet`
- **Clip Score:** `--method alignment --task clip`
- **VQA Responses:** `--method alignment --task vqa --model <model_name>`
- **Perceptual Score:** `--method diversity --task perceptual`

### GenAI CLI

Generate images or texts using AI models:

```bash
./main.py --space genai --data image|text --method <generation_method> --task <generation_task> [options]
```

### Pipeline CLI

Execute full pipelines for image generation:

```bash
./main.py --space pipeline --data image --method full --cpath <caption_path> --opath <output_path>
```

### Additional Options

Depending on the operation, you can specify paths for generated, real, caption, or output data, as well as model names, prompts, and the number of images to generate.

### Commit Guide

For contributors, please format your code using `black` before pushing:

```bash
black .
```
