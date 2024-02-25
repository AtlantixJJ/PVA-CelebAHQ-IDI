"""Evaluate the identity similarity on generated images.
"""
import torch, glob, argparse, sys, os
import numpy as np
from tqdm import tqdm
sys.path.insert(0, ".")
from lib.misc import imread_pil, dict_append
from lib.face_net import IDSimilarity


def str_latex_table(strs):
    """Format a string table to a latex table.
    
    Args:
        strs : A 2D string table. Each item is a cell.
    Returns:
        A single string for the latex table.
    """
    for i in range(len(strs)):
        for j in range(len(strs[i])):
            if "_" in strs[i][j]:
                strs[i][j] = strs[i][j].replace("_", "-")

        ncols = len(strs[0])
        seps = "".join(["c" for i in range(ncols)])
        s = []
        s.append("\\begin{table}")
        s.append("\\centering")
        s.append("\\begin{tabular}{%s}" % seps)
        header = " & ".join(strs[0]) + " \\\\\\hline"
        s.append("title & " + header)
        for line in strs[1:]:
            s.append(" & ".join(line) + " \\\\")
        s.append("\\end{tabular}")
        s.append("\\end{table}")

        for i in range(len(strs)):
            for j in range(len(strs[i])):
                if "_" in strs[i][j]:
                    strs[i][j] = strs[i][j].replace("\\_", "_")

    return "\n".join(s)


def str_table_single(dic, percentage=False):
    """Convert a 2-level dictionary to a str table."""
    strs = []
    for row_name in dic.keys():
        if len(strs) == 0: # table header
            strs.append([] + list(dic[row_name].keys()))
        s = [row_name]
        for col_name in dic[row_name].keys():
            if percentage:
                item = f"{dic[row_name][col_name]*100:.1f}"
            else:
                obj = dic[row_name][col_name]
                if isinstance(obj, dict):
                    mean, std = obj["mean"], obj["std"]
                    item = f"{mean:.3f} $\\pm$ {std:.3f}"
                else:
                    item = f"{obj:.3f}"
            s.append(item)
        strs.append(s)
    return strs


def get_fname(path):
    """Return the file image in the path (CelebAHQ format)."""
    return path.split("/")[-1].split("_")[1]


def batch_inference(func, data, batch_size=16):
    """Inference in batch."""
    res = []
    for i in tqdm(range(len(data) // batch_size + 1)):
        st, ed = i * batch_size, (i + 1) * batch_size
        st, ed = min(st, len(data)), min(ed, len(data))
        if st == ed: # no data
            break
        res.append(func(data[st:ed]))
    return torch.cat(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/celebahq")
    parser.add_argument("--expr_dir", default="expr/celebahq")
    parser.add_argument("--result_dir", default="expr/celebahq/evaluate")
    parser.add_argument("--prompt", default="small",
                        help="small or control.")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    id_crit = IDSimilarity(model_type="glint").cuda()

    if args.prompt == "small":
        inpaint_regions = ["lowerface", "eyebrow", "wholeface", "random"]
    else:
        inpaint_regions = ["wholeface"] # for editing

    methods = [
        #"SDI2",
        #"MyStyle",
        #"Paint by Example",

        #"CD-1", "CD-6",
        #"TI-1", "TI-6",

        "PVA/CrossQKV-FT40-1-R1", 
        "PVA/CrossQKV-FT40-1-R2", 
        "PVA/CrossQKV-FT40-1-R3", 
        "PVA/CrossQKV-FT40-1-R4", 
        "PVA/CrossQKV-FT40-1-R5", 
        "PVA/CrossQKV-FT40-6-R1", 
        "PVA/CrossQKV-FT40-6-R2", 
        "PVA/CrossQKV-FT40-6-R3", 
        "PVA/CrossQKV-FT40-6-R4", 
        "PVA/CrossQKV-FT40-6-R5", 

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

        #f"Inpaint_SDI2_CD_{args.prompt}_1.0",
        #f"Inpaint_SDI2_CD_{args.prompt}_6.0",
        #f"Inpaint_SDI2_TI_{args.prompt}_1.0",
        #f"Inpaint_SDI2_TI_{args.prompt}_6.0",

        f"Inpaint_CrossQKV_FT40_R1_{args.prompt}_1.0",
        f"Inpaint_CrossQKV_FT40_R2_{args.prompt}_1.0",
        f"Inpaint_CrossQKV_FT40_R3_{args.prompt}_1.0",
        f"Inpaint_CrossQKV_FT40_R4_{args.prompt}_1.0",
        f"Inpaint_CrossQKV_FT40_R5_{args.prompt}_1.0",
        f"Inpaint_CrossQKV_FT40_R1_{args.prompt}_6.0",
        f"Inpaint_CrossQKV_FT40_R2_{args.prompt}_6.0",
        f"Inpaint_CrossQKV_FT40_R3_{args.prompt}_6.0",
        f"Inpaint_CrossQKV_FT40_R4_{args.prompt}_6.0",
        f"Inpaint_CrossQKV_FT40_R5_{args.prompt}_6.0",

        #"SDI2_FP_FT20_control_15.0", "SDI2_FP_FT20_control_30.0",
        #"SDI_CD_1_control", "SDI_CD_6_control",
        #"SDI_TI_1_control", "SDI_TI_6_control"
        #"Inpaint_PVA_R1_small_1.0", "Inpaint_PVA_R2_small_1.0", "Inpaint_PVA_R3_small_1.0", "Inpaint_PVA_R4_small_1.0", "Inpaint_PVA_R5_small_1.0",
        #"Inpaint_PVA_R5_repeat_small_1.0", "Inpaint_PVA_R5_repeat_small_6.0"
    ]
    file_lists = [glob.glob(f"{args.expr_dir}/{folder}/*")
        for folder in methods_folder]
    id_feat_gt = torch.load(f"{args.data_dir}/annotation/id_feat_gt.pth").cuda()
    id_feat_gt /= id_feat_gt.norm(p=2, dim=1, keepdim=True)
    
    pil2tensor_fn = lambda pil_img: torch.from_numpy(np.asarray(
        pil_img).copy()).permute(2, 0, 1).float().cuda() / 127.5 - 1
    read_fn = lambda fname: pil2tensor_fn(imread_pil(fname, (256, 256)))
    infer_fn = lambda xl: id_crit.extract_feats(
        torch.stack([read_fn(x) for x in xl]))

    dic = {}
    for method_name, methold_folder in zip(methods, methods_folder):
        dic[method_name] = {}
        for idx, subfix in enumerate(inpaint_regions):
            folder = f"{args.expr_dir}/{methold_folder}"
            if len(inpaint_regions) == 1:
                image_paths = glob.glob(f"{folder}/*")
            else:
                # mask indice. Select a specific mask index for comparison.
                image_paths = glob.glob(f"{folder}/*_{idx}[._]*")

            inp_id = np.array([int(get_fname(x)) for x in image_paths])
            _sorted_idx = np.argsort(inp_id)
            image_paths = np.array(image_paths)[_sorted_idx]
            inp_id = inp_id[_sorted_idx]
            id_feat_inp = batch_inference(infer_fn, image_paths)
            cosims = (id_feat_gt[inp_id] * id_feat_inp).sum(1).cpu().numpy()
            dic[method_name][subfix] = cosims

    res_dic = {}
    for key1 in dic:
        all_regions = []
        for key2, value in dic[key1].items():
            mean, std = value.mean(), value.std()
            dict_append(res_dic, mean, key1, key2)
            res_dic[key1][key2] = {"mean": mean, "std": std}
            all_regions.append(mean)
        res_dic[key1]["total"] = np.array(all_regions).mean()

    os.makedirs(f"{args.result_dir}", exist_ok=True)
    with open(f"{args.result_dir}/identity_similarity.tex", "w") as f:
        strs = str_table_single(res_dic, False)
        f.writelines(str_latex_table(strs))
