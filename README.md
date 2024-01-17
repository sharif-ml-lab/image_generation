# Image Generation

### Metric CLI
Inception Score
- `python main.py --space metric --task inception --gpath utils/test/generated`

Frechet Score
- `python main.py --space metric --task frechet --gpath utils/test/generated --rpath utils/test/generated`

Clip Score
- `python main.py --space alignment --task clip --gpath utils/test/generated`

Perceptual Score
- `python main.py --space diversity --task perceptual --gpath utils/test/generated`


### Commit Guide
- Please Use `black .` Before Pushing 