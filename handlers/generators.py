from generators.sdm.trigger import generate_image_with_sdm


def sdm_handler(opath, model, prompt, count):
    generate_image_with_sdm(opath, model, prompt, count)
