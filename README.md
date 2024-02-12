# Image Generation

## Installation

### Packages
- `pip install -r requirements.txt`

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