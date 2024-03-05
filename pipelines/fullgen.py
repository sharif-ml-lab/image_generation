import json
import pandas as pd
from generators.text.diversity import sample_from_clusters
from generators.text.api import generate as generate_prompt
from generators.bing.downloader import process
from tqdm import tqdm
from metrics.diversity.text import diversity_matrix


PROMPT_PER_SCENARIO = 10
with open('utils/data/text/neutral_activity.json') as io:
    NEUTRALS = json.load(io)


def filter_prompt(prompts):
    diversity = diversity_matrix(prompts)
    scored_prompts = [(prompt, diverse) for prompt, diverse in zip(prompts, diversity)]
    scored_prompts.sort(key=lambda item:-item[1])
    result = []
    for prompt, _ in scored_prompts[:10]:
        result.append(prompt)
    return result


def fire(loader, opath):
    for subj, objc, activity, areas in loader:
        subj = subj[0]
        objc = objc[0]
        activity = activity[0]
        areas = areas[0].split('|')

        prompts_expected_double = []
        prompts_unexpected_double = []
        prompts_unexpected_single = []

        for area in areas:
            for neutral_activity in NEUTRALS[area]:
                # Expected Double
                base_prompt = f'a {subj} is {activity} and a {objc} is {neutral_activity} in a {area}'
                for comb in tqdm(sample_from_clusters(PROMPT_PER_SCENARIO), desc='Expected Double'):
                    prompts_expected_double.append(generate_prompt(base_prompt, prompts_expected_double, comb))
                # UnExpected Double
                base_prompt = f'a {objc} is {activity} and a {subj} is {neutral_activity} in a {area}'
                for comb in tqdm(sample_from_clusters(PROMPT_PER_SCENARIO), desc='UnExpected Double'):
                    prompts_unexpected_double.append(generate_prompt(base_prompt, prompts_unexpected_double, comb))
            # UnExpected Single
            base_prompt = f'a {objc} is {activity} in a {area} [Single {objc.lower().capitalize()}]'
            for comb in tqdm(sample_from_clusters(PROMPT_PER_SCENARIO), desc='UnExpected Single'):
                prompts_unexpected_single.append(generate_prompt(base_prompt, prompts_unexpected_single, comb))

        image_output_path = opath + f'/{"_".join(activity.strip().split())}/'
        
        print("Expected Double")
        prompts_expected_double = filter_prompt(prompts_expected_double)
        process(prompts_expected_double, image_output_path + 'expected_double/', image_output_path + 'expected_double.csv')

        print("UnExpected Double")
        prompts_unexpected_double = filter_prompt(prompts_unexpected_double)
        process(prompts_unexpected_double, image_output_path + 'unexpected_double/', image_output_path + 'unexpected_double.csv')

        print("UnExpected Single")
        prompts_unexpected_single = filter_prompt(prompts_unexpected_single)
        process(prompts_unexpected_single, image_output_path + 'unexpected_single/', image_output_path + 'unexpected_single.csv')