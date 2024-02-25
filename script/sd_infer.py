"""Stable Diffusion Inpainting. Including Textual Inversion and Custom Diffusion.
"""
import torch, sys, os, argparse
from tqdm import tqdm
sys.path.insert(0, ".")
from lib.diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
# import diffusers must happen before vutils, otherwise munmap_chunk error
from lib.dataset import CelebAHQIDIDataset, PROMPT_TEMPLATES_CONTROL, PROMPT_TEMPLATES_SMALL
from lora_diffusion import patch_pipe, tune_lora_scale


def load_custom_diffusion_delta(text_encoder, tokenizer, unet, save_path,
                                compress=False, freeze_model='crossattn_kv'):
    """Load the custom diffusion delta file.
    """
    st = torch.load(save_path)
    if 'text_encoder' in st:
        text_encoder.load_state_dict(st['text_encoder'])
    if 'modifier_token' in st:
        modifier_tokens = list(st['modifier_token'].keys())
        modifier_token_id = []
        for modifier_token in modifier_tokens:
            _ = tokenizer.add_tokens(modifier_token)
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for i, id_ in enumerate(modifier_token_id):
            token_embeds[id_] = st['modifier_token'][modifier_tokens[i]]

    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                if compress and ('to_k' in name or 'to_v' in name):
                    params.data += st['unet'][name]['u']@st['unet'][name]['v']
                else:
                    params.data.copy_(st['unet'][f'{name}'])
        else:
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                if compress:
                    params.data += st['unet'][name]['u']@st['unet'][name]['v']
                else:
                    params.data.copy_(st['unet'][f'{name}'])


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting. Including Textual Inversion and Custom Diffusion.")
    parser.add_argument("--model_path", type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="The name of stable diffusion model.")
    parser.add_argument("--cache_dir", type=str,
        default="pretrained",
        help="The path to diffuser cache dir.")
    parser.add_argument("--seed", type=int,
        default=2022,
        help="Random seed.")
    parser.add_argument("--rank", type=str, default="0/1",
        help="Simple parallelization. format: {local_rank}/{global_rank}")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
        help="Guidance scale.")
    parser.add_argument("--prompt_set", type=str, default="small",
        help="The prompt template set. Choices: [small, control]")
    parser.add_argument("--personalization", type=str, default="SDI",
        help="The personalization method. Choices: [SDI, TI, CD, LORA]")
    parser.add_argument("--expr_dir", type=str, default="expr/celebahq/SDI2",
        help="The path to personalized models.")
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = args.expr_dir.replace("/SDI2", "/Inpaint_SDI2")
    args.result_dir = f"{result_dir}_{args.prompt_set}_{args.guidance_scale}"
    os.makedirs(args.result_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    scheduler = DDIMScheduler.from_config(
        args.model_path, subfolder="scheduler", cache_dir=args.cache_dir)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path, safety_checker=None,
        scheduler=scheduler, cache_dir=args.cache_dir)
    unet_sd = {k: v.cpu() for k, v in pipe.unet.state_dict().items()}
    pipe = pipe.to("cuda")

    personal_words = "a person"
    if args.personalization == "TI":
        special_token = "<person>"
        personal_words = "<person>"
        pipe.tokenizer.add_tokens(special_token)
        special_token_id = pipe.tokenizer.convert_tokens_to_ids(special_token)
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        embed_module = pipe.text_encoder.get_input_embeddings()
    elif args.personalization == "CD":
        special_token = "<new1>"
        personal_words = "a <new1> person"
    elif args.personalization == "LORA":
        special_token = "<person>" #"<s1><s2>"
        personal_words = "<person>" #"<s1><s2>"

    prompt_temps = PROMPT_TEMPLATES_SMALL
    use_random_mask = True
    inpaint_regions = ["lowerface", "eyebrow", "wholeface"]
    if args.prompt_set == "control":
        prompt_temps = PROMPT_TEMPLATES_CONTROL
        inpaint_regions = ["wholeface"]
        use_random_mask = False

    test_ds = CelebAHQIDIDataset(size=(512, 512),
        split="test", loop_data="identity",
        inpaint_region=inpaint_regions,
        seed=args.seed)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    local_rank, n_rank = [int(i) for i in args.rank.split("/")]
    for batch_idx, batch in enumerate(tqdm(test_dl)):
        if batch_idx % n_rank != local_rank:
            continue
        person_id = batch["id"][0]
        
        if args.personalization == "TI":
            sd = torch.load(f"{args.expr_dir}/{person_id}/FPIP.bin")
            embed_module.weight.data[special_token_id].copy_(list(sd.values())[0])
        elif args.personalization == "CD":
            delta_path = f"{args.expr_dir}/{person_id}/delta.bin"
            load_custom_diffusion_delta(pipe.text_encoder, pipe.tokenizer, pipe.unet,
            delta_path, False, "crossattn")
        elif args.personalization == "LORA":
            # patch_pipe will handle replacement
            #pipe.unet.load_state_dict(unet_sd)
            patch_pipe(pipe, f"{args.expr_dir}/{person_id}/final_lora.safetensors",
                       patch_unet=True,
                       patch_ti=True,
                       patch_text=True)
            tune_lora_scale(pipe.unet, 0.5)
            tune_lora_scale(pipe.text_encoder, 0.5)

        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

        for p_i, (p_name, prompt_temp) in enumerate(prompt_temps.items()):
            prompt = prompt_temp.format(personal_words)
            print(f"=> ({p_i}/{len(prompt_temps)}) {prompt}")
            id_name = batch["id"][0]
            for i, image in enumerate(batch["infer_image"][0]):
                image_name = batch["all_indice"][i].item()
                image = image.unsqueeze(0).cuda()
                iv_masks = batch["infer_mask"][0, :, :, :1].cuda()
                if use_random_mask:
                    random_masks = batch["random_mask"][0, :iv_masks.shape[0], None, :1]
                    iv_masks = torch.cat([iv_masks, random_masks.cuda()], 1)
                for j, mask in enumerate(iv_masks[i]):
                    name = f"{id_name}_{image_name}_{j}_{p_name}.png"
                    if os.path.exists(f"{args.result_dir}/{name}"):
                        continue
                    mask = mask[None, :1].cuda()
                    masked_image = (image * 2 - 1).cuda() * (1 - mask)

                    n_img = pipe(prompt, masked_image, mask,
                        num_inference_steps=100,
                        guidance_scale=args.guidance_scale,
                        negative_prompt=None,
                        eta=0.5).images[0]
                    n_img.save(f"{args.result_dir}/{name}")


if __name__ == "__main__":
    main()