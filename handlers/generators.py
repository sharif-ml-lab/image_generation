from generators.sdm.trigger import generate_image_with_sdm
from generators.juggernaut.trigger import generate_image_with_juggernaut
from generators.text.trigger import generate_text


def sdm_handler(opath, model, prompt, count):
    generate_image_with_sdm(opath, model, prompt, count)


def juggernaut_handler(opath, prompt, count):
    generate_image_with_juggernaut(opath, prompt, count)


def prompts_llm_handler(opath, model, prompt, count):
    generate_text(opath, model, prompt, count)
