import json, os
import concurrent.futures
import subprocess
import threading

test_ids = json.load(open("data/celebahq/annotation/idi-5.json", "r"))["test_ids"]
instance_dir = "data/celebahq/IDI/idi_5/{}/ref_image"
output_dir = "expr/celebahq/SDI2_LORA/{}"

DEVICES = [4, 5, 6, 7]
device_slots = [[] for _ in DEVICES]
for idx, test_id in enumerate(test_ids):
    cmd_format = "python lora_diffusion/cli_lora_pti.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting  --instance_data_dir={instance_dir} --output_dir={output_dir} --train_inpainting --cached_latents=False --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=2 --gradient_checkpointing --scale_lr --learning_rate_unet=2e-4 --learning_rate_text=1e-6 --learning_rate_ti=5e-4 --color_jitter --lr_scheduler='linear' --lr_warmup_steps=0 --lr_scheduler_lora='constant' --lr_warmup_steps_lora=100 --placeholder_tokens='<person>' --save_steps=10000 --max_train_steps_ti=1000 --max_train_steps_tuning=3000 --perform_inversion=True --clip_ti_decay --weight_decay_ti=0.000 --weight_decay_lora=0.000 --device='cuda:{device_id}' --lora_rank=8 --lora_dropout_p=0.1 --lora_scale=2.0"
    # original version
    #cmd_format = "lora_pti --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting  --instance_data_dir={instance_dir} --output_dir={output_dir} --cached_latents=False --train_text_encoder --train_inpainting --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=2 --gradient_checkpointing --scale_lr --learning_rate_unet=2e-4 --learning_rate_text=1e-6 --learning_rate_ti=5e-4 --color_jitter --lr_scheduler='linear' --lr_warmup_steps=0 --lr_scheduler_lora='constant' --lr_warmup_steps_lora=100 --placeholder_tokens='<s1>|<s2>' --save_steps=10000 --max_train_steps_ti=3000 --max_train_steps_tuning=3000 --perform_inversion=True --clip_ti_decay --weight_decay_ti=0.000 --weight_decay_lora=0.000 --device='cuda:{device_id}' --lora_rank=8 --use_face_segmentation_condition --lora_dropout_p=0.1 --lora_scale=2.0"
    device_slots[idx % len(DEVICES)].append(cmd_format.format(
        instance_dir=instance_dir.format(test_id),
        output_dir=output_dir.format(test_id),
        device_id=DEVICES[idx % len(DEVICES)]))



def worker(idx):
    for cmd in device_slots[idx]:
        print(f"Running {idx}: {idx}/{len(device_slots[idx])}")
        print(cmd)
        subprocess.run(cmd, shell=True)

with concurrent.futures.ThreadPoolExecutor(len(DEVICES)) as executor:
    # Submit each command for execution
    futures = [executor.submit(worker, job_idx) for job_idx in range(len(DEVICES))]
    
    # Wait for all tasks to complete
    concurrent.futures.wait(futures)
    