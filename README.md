# Image Generation

## Installation

### Packages
- `pip install -r requirements.txt`

### Ollama 
1. Luanch Ollama
```bash
docker run -d --gpus=all \
    -v ollama:/root/.ollama \
    -p 11434:11434 \ 
    --name ollama ollama/ollama
```
2. Run Lamma2 Model
```bash
docker exec -it ollama ollama run llama2:70b
```

### Juggernaut 
```bash
docker run -d --gpus=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -p 8886:8888 konieshadow/fooocus-api
```

## Metric CLI
`./main.py --space <metric_type> --task <metric_name> --gpath <generated_path> [--rpath <real_path>]`


Inception Score
- `--space quality --task inception`

Frechet Score
- `--space quality --task frechet`

Clip Score
- `--space alignment --task clip`

VQA Responses
- `--space alignemnt --task vqa --model <model_name>`


Perceptual Score
- `--space diversity --task perceptual`

-------

### Commit Guide
Please Use `black .` Before Pushing 