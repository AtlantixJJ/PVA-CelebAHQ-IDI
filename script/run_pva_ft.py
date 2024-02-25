import os, argparse

parser = argparse.ArgumentParser(description="Run finetuning for PVA.")
parser.add_argument("--ref_num", type=int, default=5,
    help="The number of reference images.")
parser.add_argument("--gpu", type=str, default="0",
    help="The number of reference images.")
args = parser.parse_args()

# Workaround for GPU memory leakage
for i in range(40):
    os.system(f"CUDA_VISIBLE_DEVICES={args.gpu} python script/finetune.py --config config/SDI2_CrossQKV_FT40.json --resume expr/celebahq/PVA/stage2 --ref_num {args.ref_num} --output_dir expr/celebahq/CrossQKV_FT40_R{args.ref_num}")