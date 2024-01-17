# Image Generation


### Metric CLI
`./main.py --space <metric_type> --task <metric_name> --gpath <generated_path> [--rpath <real_path>]`


Inception Score
- `--space quality --task inception`

Frechet Score
- `--space quality --task frechet`

Clip Score
- `--space alignment --task clip`


Perceptual Score
- `--space diversity --task perceptual`

-------

### Commit Guide
Please Use `black .` Before Pushing 