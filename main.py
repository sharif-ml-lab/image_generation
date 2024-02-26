import argparse
import logging
import os


def main(
    space,
    method,
    data,
    task,
    gpath,
    rpath,
    cpath,
    opath,
    model,
    prompt,
    neg_prompt,
    count,
):
    all_task = task == "report"
    output = []
    
    if space == "metric":
        if data == "image":
            if method == "quality":
                if all_task or task == "inception":
                    output.append(metric_handlers.inception_handler(gpath))
                if all_task or task == "frechet":
                    output.append(metric_handlers.frechet_handler(gpath, rpath))
                if all_task or task == "realism":
                    output.append(metric_handlers.realism_handler(gpath))

            elif method == "diversity":
                if all_task or task == "perceptual":
                    output.append(metric_handlers.perceptual_handler(gpath))
                if all_task or task == "simemb":
                    output.append(metric_handlers.sentence_image_handler(gpath))
                if all_task or task == "ssim":
                    output.append(metric_handlers.ssim_image_handler(gpath))
                if all_task or task == "psnr":
                    output.append(metric_handlers.psnr_handler(gpath))

            elif method == "alignment":
                if all_task or task == "clip":
                    output.append(metric_handlers.clip_handler(gpath, cpath))
                if all_task or task == "vqa":
                    output.append(metric_handlers.vqa_handler(gpath, cpath, model))
                if all_task or task == "captioning":
                    output.append(metric_handlers.captioning_handler(gpath, cpath))

        if data == "text":
            if method == "diversity":
                if all_task or task == "censor":
                    output.append(metric_handlers.simemb_text_handler(cpath, prompt))
                if all_task or task == "bert":
                    output.append(metric_handlers.bert_diversity_handler(cpath))
            elif method == "alignment":
                if all_task or task == "sentence":
                    output.append(metric_handlers.sentence_text_handler(cpath, prompt))
                if all_task or task == "classic":
                    output.append(metric_handlers.classic_handler(cpath, prompt))

    elif space == "genai":
        if data == "image":
            if method == "sdm":
                if task == "xlarge":
                    generator_handlers.sdm_handler(opath, model, prompt, count)
                if task == "juggernaut":
                    generator_handlers.juggernaut_handler(opath, prompt, count)
            if method == "bing":
                if task == "dalle3":
                    generator_handlers.bing_handler(cpath, opath)

        elif data == "text":
            if method == "llm":
                generator_handlers.prompts_llm_handler(opath, model, prompt, count)
            if method == "config":
                if task == "llm-diversity":
                    generator_handlers.llm_diversity_handler()
                
    elif space == "experiment":
        if data == "image":
            if method == "tendency":
                experiment_handlers.tendency_handler(gpath, prompt, neg_prompt)
            if method == "noise":
                experiment_handlers.noise_handler(prompt, neg_prompt)

    print("\n".join(output))


if __name__ == "__main__":
    logging.getLogger("torch").setLevel(logging.CRITICAL)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    import handlers.generators as generator_handlers
    import handlers.metrics as metric_handlers
    import handlers.experiments as experiment_handlers

    parser = argparse.ArgumentParser(
        description="Sharif ML-Lab Data Generation ToolKit"
    )
    parser.add_argument(
        "-s", "--space", type=str, required=True, help="Space Name (e.g. metric, genai)"
    )
    parser.add_argument(
        "-mt",
        "--method",
        type=str,
        required=True,
        help="Method Name (e.g. quality, diversity, sdm)",
    )
    parser.add_argument(
        "-d", "--data", type=str, required=True, help="Kind of Data (e.g. image, text)"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=False,
        help="Task Name (e.g. inception, xlarge)",
    )
    parser.add_argument(
        "-gp", "--gpath", type=str, required=False, help="Generated Data Path"
    )
    parser.add_argument(
        "-cp", "--cpath", type=str, required=False, help="Caption Data Path"
    )
    parser.add_argument(
        "-rp", "--rpath", type=str, required=False, help="Real Data Path"
    )
    parser.add_argument(
        "-op", "--opath", type=str, required=False, help="Output Data Path"
    )
    parser.add_argument("-m", "--model", type=str, required=False, help="Model Name")
    parser.add_argument("-p", "--prompt", type=str, required=False, help="Prompt")
    parser.add_argument(
        "-np", "--neg-prompt", type=str, required=False, help="Negative Prompt"
    )
    parser.add_argument(
        "-n", "--count", type=int, required=False, help="Number of Images To Generate"
    )
    args = parser.parse_args()
    main(**vars(args))
