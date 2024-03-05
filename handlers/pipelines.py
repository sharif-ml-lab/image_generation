from utils.load import Loader
from pipelines.fullgen import fire as full_fire


def tendency_handler(cpath, opath):
    pipeline_captions = Loader.load_pipeline(cpath, batch_size=1)
    full_fire(pipeline_captions, opath)
