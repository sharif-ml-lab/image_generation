import argparse
import handlers.generators as generator_handlers
import handlers.metrics as metric_handlers


def main(space, task, gpath, rpath, model, cpath):
    if space == "quality":
        if task == "inception":
            metric_handlers.inception_handler(gpath)
        elif task == "frechet":
            metric_handlers.frechet_handler(gpath, rpath)
        elif task == "realism":
            metric_handlers.realism_handler(gpath)
            
    elif space == "diversity":
        if task == "perceptual":
            metric_handlers.perceptual_handler(gpath)
        if task == "simemb":
            metric_handlers.simemb_handler(gpath)
        if task == "ssim":
            metric_handlers.ssim_handler(gpath)
        if task == "psnr":
            metric_handlers.psnr_handler(gpath)

    elif space == "alignment":
        if task == "clip":
            metric_handlers.clip_handler(gpath)
        elif task == "vqa":
            metric_handlers.vqa_handler(gpath, model)
        elif task == "captioning":
            metric_handlers.captioning_handler(gpath, cpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sharif ML-Lab Data Generation ToolKit"
    )
    parser.add_argument(
        "-s", "--space", type=str, required=True, help="Space Name (e.g. metric, crawl)"
    )
    parser.add_argument(
        "-t", "--task", type=str, required=True, help="Task Name (e.g. inception, knn)"
    )
    parser.add_argument(
        "-gp", "--gpath", type=str, required=True, help="Generated Data Path"
    )
    parser.add_argument(
        "-cp", "--cpath", type=str, required=False, help="Caption Data Path"
    )
    parser.add_argument(
        "-rp", "--rpath", type=str, required=False, help="Real Data Path"
    )
    parser.add_argument("-m", "--model", type=str, required=False, help="Model Name")
    args = parser.parse_args()
    print(args)
    main(**vars(args))
