
"""Train PVA pathway for Stable Diffusion.
"""
import argparse, logging, os, torch, json, sys
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
sys.path.insert(0, ".")
from lib.lion_pytorch import Lion
from lib.diffusers import get_scheduler
from lib.pva import PVADiffusion
from lib.visualizer import ref_inpaint_visualize_html
from lib.dataset import CelebAHQIDIDataset


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
    parser.add_argument("--config", type=str, default="config/SDI2_PVA_stage1.json",
        help="The path to the config.json file.")
    parser.add_argument("--expr_name", type=str, default="stage1",
        help="The name for experiment")
    parser.add_argument("--resume", type=str, default="",
        help="Resume from previous training.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


if __name__ == "__main__":
    args = parse_args()
    config = json.load(open(args.config))
    expr_dir = os.path.join(config["training"]["output_dir"], args.expr_name)

    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_with="tensorboard",
        logging_dir=expr_dir)

    fpi_pipe = PVADiffusion(config, accelerator.device)

    # create dataloaders
    size = config["data"]["resolution"]
    B = config['training']['batch_size']
    common_dict = {"data_dir": config["data"]["data_dir"],
                   "size": (size, size),
                   "seed": args.seed}
    train_ds = CelebAHQIDIDataset(**common_dict,
        loop_data="image-all", split="train", flip_p=0.5)
    val_ds = CelebAHQIDIDataset(**common_dict,
        inpaint_region=["wholeface"], split="val")
    val_train_ds = CelebAHQIDIDataset(**common_dict,
        inpaint_region=["wholeface"], split="train")
    train_dl = torch.utils.data.DataLoader(train_ds,
        batch_size=B, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds,
        batch_size=1, shuffle=False)
    val_train_dl = torch.utils.data.DataLoader(val_train_ds,
        batch_size=1, shuffle=False)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(expr_dir, exist_ok=True)
        os.makedirs(f"{expr_dir}/val", exist_ok=True)
        os.makedirs(f"{expr_dir}/train", exist_ok=True)
        # save the current config
        json.dump(config, open(f"{expr_dir}/config.json", "w"), indent=2)

    # scale the learning rate with total batch size
    total_batch_size = config["training"]["batch_size"] * accelerator.num_processes * \
        config["training"]["gradient_accumulation_steps"]
    config["optimizer"]["lr"] *= total_batch_size

    if len(args.resume) > 0:
        sd = torch.load(f"{args.resume}/FPIP.bin", map_location="cpu")
        logger.info(f"Loaded components: {sd.keys()}")
        fpi_pipe.restore(sd)

    optimizer_fn = {
        "adam": torch.optim.AdamW,
        "lion": Lion,
    }[config["optimizer"]["name"]]
    optimizer = optimizer_fn(
        fpi_pipe.setup_parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"])

    config["lr_scheduler"]["num_warmup_steps"] *= total_batch_size
    lr_scheduler = get_scheduler(
        optimizer=optimizer, **config["lr_scheduler"])
    
    # Prepare for DDP
    fpi_pipe.prepare_accelerator(accelerator)
    optimizer, train_dl, val_dl, val_train_dl, lr_scheduler = \
        accelerator.prepare(optimizer, train_dl, val_dl, val_train_dl, lr_scheduler)
    face_net_train = fpi_pipe.face_net_train
    trans_net = fpi_pipe.trans_net
    vae, unet = fpi_pipe.vae, fpi_pipe.unet
    tokenizer = fpi_pipe.tokenizer
    clip_text_enc = fpi_pipe.clip_text_enc
    scheduler = fpi_pipe.scheduler

    if accelerator.is_main_process:
        accelerator.init_trackers("tensorboard")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(config["training"]["max_train_steps"]),
        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    T = fpi_pipe.scheduler.config.num_train_timesteps
    bin_size = T // config["training"]["num_time_resample"]

    # did not use seed, otherwise different local processes would have the same randomization
    #if args.seed is not None:
    #    set_seed(args.seed)

    while global_step < config["training"]["max_train_steps"]:
        for step, batch in enumerate(train_dl):
            mask = batch["random_mask"][:, 0, :1] # (N, 1, H, W)
            smask = F.interpolate(mask, scale_factor=1/8)
            iv_img = batch["infer_image"][:, 0] * 2 - 1 # (N, C, H, W)
            masked_image = iv_img * (1 - mask)
            rv_imgs = batch["ref_image"]
            rand_n_ref = torch.randint(1, rv_imgs.shape[1] + 1, (1,)).item()
            rv_imgs = rv_imgs[:, :rand_n_ref].contiguous()
            if torch.rand(1).item() < 0.2:
                rv_imgs = torch.cat([rv_imgs, rv_imgs[:, -1:].flip(4)], 1)
            N, N_REF = rv_imgs.shape[:2]

            with torch.no_grad():
                vae_input = torch.cat([iv_img, masked_image])
                vae_output = 0.18215 * vae.encode(vae_input).latent_dist.sample()
                gt_latents, mask_latents = vae_output[:B], vae_output[B:]
                # In image-all, results are [(B_0, B_1), (B_0, B_1), ...]
                prompts = [fpi_pipe.personalized_prompt(s)
                           for s in batch["prompt_template"][0]]
                input_ids = tokenizer(prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt").input_ids.to(accelerator.device)
                text_feat = None
                if config["text_in"] == "None":
                    text_feat = clip_text_enc(input_ids)[0].requires_grad_(True)
                if config["loss"]["ID"] > 0:
                    infer_id_feats = fpi_pipe.face_net_eval(iv_img)
                    infer_id_feats /= infer_id_feats.norm(p=2, dim=1, keepdim=True)

            # stratified sampling of time
            for repeat in range(config["training"]["num_time_resample"]):
                noise = torch.randn_like(mask_latents)
                bin_st, bin_ed = bin_size * repeat, bin_size * (repeat + 1)
                t = torch.randint(bin_st, bin_ed, (B,), device=noise.device)
                noisy_latents = scheduler.add_noise(gt_latents, noise, t)
                z_t = torch.cat([noisy_latents, smask, mask_latents], 1)
                z_t = z_t.requires_grad_(True)
                with accelerator.accumulate(fpi_pipe):
                    context_feat, visual_feat = fpi_pipe.calc_visual_feature(
                        x=rv_imgs.requires_grad_(True),
                        text_feat=text_feat, text_id=input_ids)
                    pred_eps = unet(z_t, t, context_feat).sample
                    dsm_loss = torch.square(pred_eps - noise).mean([1, 2, 3])
                    accelerator.backward(dsm_loss.mean())
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                logs = {"lr": min(lr_scheduler.get_last_lr())}
                if config["text_in"] == "Token":
                    logs["modifier_token"] = fpi_pipe.tuned_embedding.modifier_vec.norm().item()
                if config["unet"]["tune_params"] == "PVA":
                    pva_params = [v.norm().item() for k, v in \
                        fpi_pipe.unet.named_parameters() if "to_visual" in k]
                    logs["PVA_norm"] = sum(pva_params) / len(pva_params)

                if accelerator.sync_gradients and global_step % config["training"]["save_steps"] == 0:
                    fpi_pipe.eval()
                    if accelerator.is_main_process:
                        torch.save(fpi_pipe.simple_state_dict(accelerator), f"{expr_dir}/FPIP.bin")
                        torch.save(optimizer.state_dict(), f"{expr_dir}/optimizer.bin")
                    print(f"=> Visualizing at step {global_step}")
                    with torch.no_grad():
                        val_ds.rng = np.random.RandomState(args.seed)
                        val_train_ds.rng = np.random.RandomState(args.seed)
                        common_dict = dict(
                            use_random_mask=True,
                            accelerator=accelerator,
                            max_batches=20 // accelerator.num_processes,
                            single_file=True,
                            max_infer_samples=1)
                        cosim_val_1 = fpi_pipe.inpaint_dl(
                            dl=val_dl,
                            html_dir=os.path.join(expr_dir, "val", f"{global_step}_1"),
                            **common_dict)
                        cosim_val_6 = fpi_pipe.inpaint_dl(
                            dl=val_dl,
                            html_dir=os.path.join(expr_dir, "val", f"{global_step}_6"),
                            guidance_scale=6.0,
                            **common_dict)
                        cosim_train_1 = fpi_pipe.inpaint_dl(
                            dl=val_train_dl,
                            html_dir=os.path.join(expr_dir, "train", f"{global_step}_1"),
                            **common_dict)
                    log_dic = {
                        "cosim_val_1": cosim_val_1.mean().item(),
                        "cosim_train_1": cosim_train_1.mean().item(),
                        "cosim_val_6": cosim_val_6.mean().item()}
                    fpi_pipe.train()
                    fpi_pipe.face_net_train.eval()
                    logs.update(log_dic)
                    torch.cuda.empty_cache()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                # record DSM loss in different time bins
                for i, t_ in enumerate(t):
                    dsm_loss_bin = int(t_ / (T // 5))
                    logs[f"dsm_loss_{dsm_loss_bin}"] = dsm_loss[i].detach().item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                accelerator.wait_for_everyone()

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        torch.save(fpi_pipe.simple_state_dict(accelerator), f"{expr_dir}/FPIP.bin")
    accelerator.end_training()
