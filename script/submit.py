"""Submit mutilple jobs."""
import os, argparse, glob, json

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="-1")
parser.add_argument("--func", default="all")
args = parser.parse_args()


def tii_infer():
    cmds = []
    cmd = "python3 script/pva_infer.py --ft_dir expr/celebahq/SDI2_{personalization} --disp_batches 20 --out_prefix expr/celebahq/Inpaint_SDI2_{personalization} --guidance_scale {scale} --prompt_set {prompt_set} --config config/SDI2_{personalization}.json"

    for personalization in ["TI"]: # "TI", 
        for prompt_set in ["control"]: # ["small", "control"]
            for scale in [6]:
                cmds.append(cmd.format(scale=scale, prompt_set=prompt_set, personalization=personalization))

    return cmds


def sd_infer():
    cmds = []
    cmd = "python3 script/sd_infer.py --personalization {personalization} --guidance_scale {scale} --prompt_set {prompt_set} --rank {rank}/{n_rank} --expr_dir expr/celebahq/SDI2_{personalization}"

    n_rank = 1
    for personalization in ["LORA"]: # "TI", "CD"
        for prompt_set in ["control"]:
            for scale in [1, 6]:
                for rank in range(n_rank):
                    cmds.append(cmd.format(scale=scale, prompt_set=prompt_set, rank=rank, n_rank=n_rank, personalization=personalization))

    return cmds


def tii_train():
    cmds = []
    cmd = "python3 script/finetune.py --manual_rank {rank}/{n_rank}"

    n_rank = 4
    for rank in range(n_rank):
        cmds.append(cmd.format(rank=rank, n_rank=n_rank))

    return cmds


def cdi_train():
    cmds = []
    model_name = "stabilityai/stable-diffusion-2-inpainting"
    data_dir = "data/celebahq"
    expr_dir = "expr/celebahq/SDI2_CD"
    fp = f"{data_dir}/annotation/idi-5.json"
    test_ids = json.load(open(fp, "r"))["test_ids"]

    cmd_format = """accelerate launch --multi_gpu --gpu_ids 2,3 --main_process_port 29500 --num_processes=2 \
    script/cd_train.py \
    --pretrained_model_name_or_path={model_name}  \
    --instance_data_dir={data_dir}/IDI/5/{id_name}/ref_image  \
    --class_data_dir={data_dir}/image  \
    --output_dir={expr_dir}/{id_name}  \
    --with_prior_preservation --real_prior --prior_loss_weight=1.0  \
    --flip_p 0.5 \
    --instance_prompt="photo of a <new1> person"  \
    --class_data_dir={data_dir}/annotation/dialog/class_images.txt  \
    --class_prompt={data_dir}/annotation/dialog/class_prompts.txt  \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps 4 \
    --learning_rate=1e-6 \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --num_class_images=10000 \
    --scale_lr \
    --modifier_token "<new1>" \
    --freeze_model crossattn \
    --gradient_checkpointing \
    --save_steps 100000
    """

    for i, id_name in enumerate(test_ids):
        out_dir = f"{expr_dir}/{id_name}"
        if os.path.exists(out_dir):
            print(f"!> Skip {out_dir}")
            continue
        cmd = cmd_format.format(data_dir=data_dir, expr_dir=expr_dir,
            id_name=id_name, model_name=model_name)
        cmds.append(cmd)
    return cmds


def pva_ft_infer():
    cmds = []
    cmd = "python3 script/pva_infer.py --guidance_scale {scale} --prompt_set {prompt_set} --disp_batches 10 --out_prefix expr/celebahq/Inpaint_CrossQKV_FT40_R{num_ref} --resume expr/celebahq/PVA/stage2 --ref_num {num_ref} --ft_dir expr/celebahq/CrossQKV_FT40_R{num_ref}"

    for prompt_set in ["small"]: # "control"
        scales = [1, 6] if prompt_set == "small" else [15]
        for num_ref in [1, 5]:
            for scale in scales:
                cmds.append(cmd.format(scale=scale, prompt_set=prompt_set, num_ref=num_ref))
    return cmds


def pva_infer():
    cmds = []
    cmd = "python3 script/pva_infer.py --guidance_scale {scale} --prompt_set {prompt_set} --disp_batches 10 --out_prefix expr/celebahq/Inpaint_PVA_R{num_ref}_repeat --resume expr/celebahq/PVA/stage2 --ref_num {num_ref}"

    for prompt_set in ["small"]:
        scales = [1, 6] if prompt_set == "small" else [1, 30]
        for scale in scales:
            for num_ref in [5]: #1, 2, 3, 4, 
                cmds.append(cmd.format(scale=scale, prompt_set=prompt_set, num_ref=num_ref))
    return cmds


func = {
    "sd_infer": sd_infer,
    "tii_infer": tii_infer,
    "tii_train": tii_train,
    "cdi_train": cdi_train,
    "pva_ft_infer": pva_ft_infer,
    "pva_infer": pva_infer,
}[args.func]

gpus = args.gpu.split("/")
slots = [[] for _ in gpus]
for i, cmd in enumerate(func()):
    gpu_cmd = cmd
    if gpus[0] != "-1":
        gpu_cmd = f"CUDA_VISIBLE_DEVICES={gpus[i % len(gpus)]} {cmd}"
    slots[i % len(gpus)].append(gpu_cmd)

if len(slots) > 1:
    for slot_cmds in slots:
        slot_cmd = " && ".join(slot_cmds) + " &"
        print(slot_cmd)
        os.system(slot_cmd)
else:
    try:
        for cmd in slots[0]:
            os.system(cmd)
    except:
        print("!> Interrupted!")

