# Image Generation

## Installation
- `docker compose up -d`


## Experiment CLI
`./main.py --space experiment --image data --method tendecny|noise [--gpath <generated_path>] --prompt <true_caption> --neg-prompt <false_caption>


## Metric CLI
`./main.py --space metric --image data --method <metric_type> --task <metric_name> --gpath <generated_path> [--rpath <real_path>]`


Inception Score
- `--method quality --task inception`

Frechet Score
- `--method quality --task frechet`

Clip Score
- `--method alignment --task clip`

VQA Responses
- `--method alignemnt --task vqa --model <model_name>`


Perceptual Score
- `--method diversity --task perceptual`

-------

### Commit Guide
Please Use `black .` Before Pushing 