
"""Finetune PVA pathway for a specific identity.
"""
import argparse, logging, os, torch, sys, json
from collections import defaultdict
from accelerate import Accelerator
from accelerate.logging import get_logger
sys.path.insert(0, ".")
from lib.pva import PVADiffusion
from lib.dataset import CelebAHQIDIDataset, PROMPT_TEMPLATES_SMALL

  
logger = get_logger(__name__)
logging.basicConfig(
    format="%(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Learn a fast personalization network for diffusion models.")
    parser.add_argument("--seed", type=int, default=2022,
        help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1,
        help="For distributed training: local_rank. Needed for accelerate.")
    parser.add_argument("--manual_rank", type=str, default="0/1",
        help="Submit parallel jobs for different ids.")
    parser.add_argument("--config", type=str, default="config/SDI2_TI.json",
        help="The path to the config.json file.")
    parser.add_argument("--resume", type=str,
        default="",
        help="Resume from previous training.")
    parser.add_argument("--ref_num", type=int, default=5,
        help="The number of reference images.")
    parser.add_argument("--output_dir", type=str,
        default="",
        help="The output directory. If not specified, use the path in config.")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


if __name__ == "__main__":
    args = parse_args()
    config = json.load(open(args.config))
    if len(args.output_dir) > 1:
        config["training"]["output_dir"] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"])
    total_batch_size = config["training"]["batch_size"] * accelerator.num_processes * \
        config["training"]["gradient_accumulation_steps"]
    config["optimizer"]["lr"] *= total_batch_size
    fpi_pipe = PVADiffusion(config, accelerator.device)
    restore_sd = {}
    if len(args.resume) > 0:
        print(f"=> Loading base model from {args.resume}/FPIP.bin")
        restore_sd = torch.load(f"{args.resume}/FPIP.bin", map_location="cpu")
        fpi_pipe.restore(restore_sd)
    else:
        print("=> Using the pretrained model as base model.")
        restore_sd["modifier_vec"] = fpi_pipe.tuned_embedding.modifier_vec.clone().detach().cpu()
        restore_sd["unet"] = {k: v.cpu() for k, v in fpi_pipe.unet.state_dict().items()}

    optimizer = torch.optim.AdamW(
        fpi_pipe.setup_parameters(), **config["optimizer"])
    fpi_pipe.prepare_accelerator(accelerator)
    optimizer = accelerator.prepare(optimizer)
    test_ids = CelebAHQIDIDataset(
        data_dir=config["data"]["data_dir"],
        use_caption=False,
        inpaint_region=["wholeface"], split="test").ids
    
    prompt_temp = list(PROMPT_TEMPLATES_SMALL.values())[0]
    prompts = [fpi_pipe.personalized_prompt(prompt_temp)]
    input_ids = fpi_pipe.tokenizer(prompts,
        padding="max_length",
        truncation=True,
        max_length=fpi_pipe.tokenizer.model_max_length,
        return_tensors="pt").input_ids.to(accelerator.device)
    with torch.no_grad():
        text_feat_orig = fpi_pipe._pre_visual_feature(None, input_ids)[0]

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    last_res = None
    rank, n_rank = [int(i) for i in args.manual_rank.split("/")]
    for i, id_name in enumerate(test_ids):
        if i % n_rank != rank:
            continue
        print(f"=> Finetuning for {id_name}")
        expr_dir = os.path.join(config["training"]["output_dir"], id_name)
        if accelerator.is_main_process:
            os.makedirs(expr_dir, exist_ok=True)
        if os.path.exists(os.path.join(expr_dir, "FPIP.bin")):
            print(f"=> {expr_dir}/FPIP.bin exists, skip!")
            continue
        orig_rv_imgs = CelebAHQIDIDataset(
            data_dir=config["data"]["data_dir"],
            loop_data="identity", use_caption=False,
            flip_p=0, single_id=id_name)[0]["ref_image"]

        # restore base model
        net = fpi_pipe.unet.module if hasattr(fpi_pipe.unet, "module") else fpi_pipe.unet
        net.load_state_dict(restore_sd["unet"], strict=False)
        fpi_pipe.tuned_embedding.load_state_dict(restore_sd, strict=False)

        # restore optimizer
        optimizer.state = defaultdict(dict)

        # visualize
        val_train_ds = CelebAHQIDIDataset(
            data_dir=config["data"]["data_dir"],
            seed=args.seed,
            loop_data="image-all", use_caption=False,
            inpaint_region=["wholeface"],
            flip_p=0, single_id=id_name)
        val_train_dl = torch.utils.data.DataLoader(val_train_ds,
            batch_size=1, shuffle=False)

        #fpi_pipe.inpaint_dl(dl=accelerator.prepare(val_train_dl),
        #                html_dir=f"{expr_dir}/base_",
        #                accelerator=accelerator,
        #                num_infer_steps=50,
        #                ref_num=args.ref_num)

        # finetuning is here
        fpi_pipe.finetune(orig_rv_imgs, accelerator)

        # save the model
        if accelerator.is_main_process:
            sd = fpi_pipe.simple_state_dict(accelerator)
            torch.save(sd, os.path.join(expr_dir, "FPIP.bin"))

        #fpi_pipe.inpaint_dl(dl=accelerator.prepare(val_train_dl),
        #                    html_dir=f"{expr_dir}/finetune_",
        #                    accelerator=accelerator,
        #                    ref_num=args.ref_num)
        accelerator.wait_for_everyone()