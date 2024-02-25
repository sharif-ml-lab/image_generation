from utils.load import Loader
from experiment.tendency import tendency_experiment
from experiment.noise import noise_experiment


def tendency_handler(gpath, prompt, neg_prompt):
    generated_dataset = Loader.load(gpath, batch_size=1)
    tendency_experiment(generated_dataset, prompt, neg_prompt)


def noise_handler(prompt, neg_prompt):
    noise_experiment(prompt, neg_prompt)
