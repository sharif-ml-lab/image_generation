from utils.load import Loader

from generators.sdm.trigger import generate_image_with_sdm
from generators.juggernaut.trigger import generate_image_with_juggernaut
from generators.bing.trigger import generate_image_with_bing

from generators.text.trigger import generate_text


def sdm_handler(opath, model, prompt, count):
    generate_image_with_sdm(opath, model, prompt, count)


def juggernaut_handler(opath, prompt, count):
    generate_image_with_juggernaut(opath, prompt, count)


def prompts_llm_handler(opath, model, prompt, count):
    generate_text(opath, model, prompt, count)


def bing_handler(cpath, opath):
    loader = Loader.load_texts(cpath, batch_size=1)
    generate_image_with_bing(loader, opath)
