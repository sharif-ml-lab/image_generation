from utils.load import Loader

from generators.sdm.trigger import generate_image_with_sdm
from generators.juggernaut.trigger import generate_image_with_juggernaut
from generators.bing.trigger import generate_image_with_bing

from generators.text.trigger import generate_text, summarize_text
from generators.text.diversity import save_cluster_data


def sdm_handler(opath, model, prompt, count):
    generate_image_with_sdm(opath, model, prompt, count)


def juggernaut_handler(opath, prompt, count):
    generate_image_with_juggernaut(opath, prompt, count)


def prompts_llm_handler(opath, model, prompt, count):
    generate_text(opath, model, prompt, count)


def summarize_llm_handler(cpath, opath, model, prompt):
    text_dataset = Loader.load_texts(cpath, batch_size=1)
    summarize_text(text_dataset, opath, model, prompt)


def llm_diversity_handler():
    save_cluster_data("utils/data/text/cluster.pkl")


def bing_handler(cpath, opath):
    loader = Loader.load_texts(cpath, batch_size=1)
    generate_image_with_bing(loader, opath)
