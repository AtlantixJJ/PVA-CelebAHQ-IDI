"""Calculate the FID and KID."""
import glob, os, json, argparse
import numpy as np
from cleanfid.fid import frechet_distance, get_files_features, kernel_distance
from cleanfid.features import build_feature_extractor

def statis_from_feats(feats):
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)

def fp2indice(fp):
    return int(fp[fp.rfind("/"):].split("_")[1])

model = build_feature_extractor(mode="clean", device="cuda", use_dataparallel=False)

data_dir = "data/celebahq/IDI"
gt_dir = f"{data_dir}/5/"

gt_feat_path = f"{data_dir}/5_infer_test_feat.npz"
if os.path.exists(gt_feat_path):
    print(f"=> Loading GT features from {gt_feat_path}")
    res = np.load(gt_feat_path)
    gt_feats = res["gt_feats"]
    gt_indices = res["gt_indices"].tolist()
else:
    files = glob.glob(f"{gt_dir}/*")
    files.sort()
    gt_indices = [int(f[f.rfind("/")+1:-4]) for f in files]
    gt_feats = get_files_features(files, model)
    np.savez_compressed(gt_feat_path, gt_indices=gt_indices, gt_feats=gt_feats)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/celebahq")
parser.add_argument("--expr_dir", default="expr/celebahq")
parser.add_argument("--result_dir", default="expr/celebahq/evaluate")
parser.add_argument("--prompt", default="small",
                    help="small or control.")
args = parser.parse_args()

methods = [
    #"SDI2",
    #"MyStyle",
    #"Paint by Example",
    "LORA-1", "LORA-6"
    #"CD-1", "CD-6",
    #"TI-1", "TI-6",

    #"PVA-FT20-1", "PVA-FT20-6",
    #"PVA-FT100-1", "PVA-FT100-6",
    #"PVA-FT40-1", "PVA-FT40-6",
    #"PVA/CrossQKV-FT40-1-R1", 
    #"PVA/CrossQKV-FT40-1-R2", 
    #"PVA/CrossQKV-FT40-1-R3", 
    #"PVA/CrossQKV-FT40-1-R4", 
    #"PVA/CrossQKV-FT40-1-R5", 
    #"PVA/CrossQKV-FT40-6-R1", 
    #"PVA/CrossQKV-FT40-6-R2", 
    #"PVA/CrossQKV-FT40-6-R3", 
    #"PVA/CrossQKV-FT40-6-R4", 
    #"PVA/CrossQKV-FT40-6-R5", 
    #"PVA-FT20-orig",
    #"PVA-1", "PVA-2", "PVA-4", "PVA-6",
    #"PVA-1-Repeat", "PVA-6-Repeat",
    #"Edit-PVA-15", "Edit-PVA-30",
    #"Edit-CD-1", "Edit-CD-6",
    #"Edit-TI-1", "Edit-TI-6",
    #"PVA-R1-1", "PVA-R2-1", "PVA-R3-1", "PVA-R4-1", "PVA-R5-1",
    #"PVA-R1-6", "PVA-R2-6", "PVA-R3-6", "PVA-R4-6", "PVA-R5-6",
    #"PVA-FT20-R1-1", "PVA-FT20-R2-1", "PVA-FT20-R3-1", "PVA-FT20-R4-1", "PVA-FT20-R5-1"
]
methods_folder = [
    #"SDI2",
    #"MyStyle_IDI_5/output",
    #"PbE",

    f"Inpaint_SDI2_LORA_{args.prompt}_1.0",
    f"Inpaint_SDI2_LORA_{args.prompt}_6.0",
    #f"Inpaint_SDI2_CD_{args.prompt}_1.0",
    #f"Inpaint_SDI2_CD_{args.prompt}_6.0",
    #f"Inpaint_SDI2_TI_{args.prompt}_1.0",
    #f"Inpaint_SDI2_TI_{args.prompt}_6.0",

    #f"Inpaint_CrossQKV_FT40_R1_{args.prompt}_1.0",
    #f"Inpaint_CrossQKV_FT40_R2_{args.prompt}_1.0",
    #f"Inpaint_CrossQKV_FT40_R3_{args.prompt}_1.0",
    #f"Inpaint_CrossQKV_FT40_R4_{args.prompt}_1.0",
    #f"Inpaint_CrossQKV_FT40_R5_{args.prompt}_1.0",
    #f"Inpaint_CrossQKV_FT40_R1_{args.prompt}_6.0",
    #f"Inpaint_CrossQKV_FT40_R2_{args.prompt}_6.0",
    #f"Inpaint_CrossQKV_FT40_R3_{args.prompt}_6.0",
    #f"Inpaint_CrossQKV_FT40_R4_{args.prompt}_6.0",
    #f"Inpaint_CrossQKV_FT40_R5_{args.prompt}_6.0",

    #"SDI2_FP_FT20_control_15.0", "SDI2_FP_FT20_control_30.0",
    #"SDI_CD_1_control", "SDI_CD_6_control",
    #"SDI_TI_1_control", "SDI_TI_6_control"
    #"Inpaint_PVA_R1_small_1.0", "Inpaint_PVA_R2_small_1.0", "Inpaint_PVA_R3_small_1.0", "Inpaint_PVA_R4_small_1.0", "Inpaint_PVA_R5_small_1.0",
    #"Inpaint_PVA_R5_repeat_small_1.0", "Inpaint_PVA_R5_repeat_small_6.0"
]

total_masks = 4 if args.prompt == "small" else 1

dic = {"fid": {}, "kid": {}}
for method, method_folder in zip(methods, methods_folder):
    res_dir = f"{args.expr_dir}/{method_folder}"
    dic["fid"][method] = {}
    dic["kid"][method] = {}
    for mask_idx in range(total_masks):
        print(f"=> Computing statistics for {res_dir} mask_idx={mask_idx}")
        files = glob.glob(f"{res_dir}/*_{mask_idx}[._]*")
        files.sort()
        indices = [gt_indices.index(fp2indice(f)) for f in files]
        feats = get_files_features(files, model)
        mu, cov = statis_from_feats(feats)
        this_gt_feats = gt_feats[indices]
        gt_mu, gt_cov = statis_from_feats(this_gt_feats)
        fid = frechet_distance(mu, cov, gt_mu, gt_cov)
        kid = kernel_distance(feats, gt_feats)
        dic["fid"][method][mask_idx] = float(fid)
        dic["kid"][method][mask_idx] = float(kid) * 1000
    json.dump(dic, open(f"{args.result_dir}/fid_kid.json", "w"), indent=2)