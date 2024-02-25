"""Infer using Fast Personalized Inpainting Pipeline."""
import argparse, os, torch, json, sys, glob
from accelerate import Accelerator
sys.path.insert(0, ".")
from lib.pva import PVADiffusion
from lib.dataset import CelebAHQIDIDataset, PROMPT_TEMPLATES_SMALL, PROMPT_TEMPLATES_CONTROL


def parse_args():
    parser = argparse.ArgumentParser(description="Fast personalization for inpainting.")
    parser.add_argument("--resume", type=str, default="",
        help="The path to the model directory.")
    parser.add_argument("--config", type=str, default="",
        help="The path to the config file. Can be left empty if using args.resume.")
    parser.add_argument("--ft_dir", type=str, default="",
        help="The directory to finetuned models.")
    parser.add_argument("--out_prefix", type=str, default="",
        help="The prefix of output results.")
    parser.add_argument("--ref_num", type=int, default=-1,
        help="Control the number of reference images.")
    parser.add_argument("--disp_batches", type=int, default=-1,
        help="Number of batches shown as HTML.")
    parser.add_argument("--dataset", type=str,
        default="data/celebahq",
        help="Path to dataset.")
    parser.add_argument("--prompt_set", type=str,
        default="small",
        help="small / control. Small is for inpainting only.")
    parser.add_argument("--guidance_scale", type=float,
        default=1.0,
        help="Classifier-free guidance scale.")
    parser.add_argument("--num_infer_steps", type=int,
        default=100,
        help="Default: 100 steps DDIM with eta=1.")
    parser.add_argument("--seed", type=int,
        default=2023,
        help="For reproducible results.")
    parser.add_argument("--local_rank", type=int,
        help="needed for accelerator.")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    accelerator = Accelerator()
    image_dir = f"{args.out_prefix}_{args.prompt_set}_{args.guidance_scale}"
    html_dir = f"{image_dir}_html/"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    if args.resume == "":
        if args.config == "":
            args.config = "config/SDI2_TI.json"
        config = json.load(open(args.config))
        fpi_pipe = PVADiffusion(config, accelerator.device)
    else:
        resume_fpath = os.path.join(args.resume, "config.json")
        config = json.load(open(resume_fpath))
        fpi_pipe = PVADiffusion(config, accelerator.device)
        data_path = os.path.join(args.resume, "FPIP.bin")
        fpi_pipe.restore(torch.load(data_path, map_location="cpu"))

    prompt_temps = PROMPT_TEMPLATES_SMALL
    use_random_mask = True
    extra_feat_neg = False
    inpaint_regions = ["lowerface", "eyebrow", "wholeface"]
    if args.prompt_set == "control":
        prompt_temps = PROMPT_TEMPLATES_CONTROL
        inpaint_regions = ["wholeface"]
        use_random_mask = False
        extra_feat_neg = True
    test_ds = CelebAHQIDIDataset(
        data_dir=args.dataset,
        split="test",
        use_caption=False,
        inpaint_region=inpaint_regions,
        seed=args.seed)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    test_dl = accelerator.prepare(test_dl)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    fpi_pipe.eval()
    fpi_pipe.inpaint_dl(
        dl=test_dl,
        html_dir=html_dir,
        html_num=args.disp_batches,
        image_dir=image_dir,
        prompt_temps=prompt_temps,
        extra_feat_neg=extra_feat_neg,
        use_random_mask=use_random_mask,
        accelerator=accelerator,
        ft_dir=args.ft_dir if len(args.ft_dir) > 0 else None,
        max_batches=-1,
        max_infer_samples=-1,
        ref_num=args.ref_num,
        guidance_scale=args.guidance_scale,
        num_infer_steps=args.num_infer_steps)

if __name__ == "__main__":
    main()