"""Export the CelebAHQ-IDI dataset into images organized in folders."""
import argparse, os, sys
import torchvision.utils as vutils
from tqdm import tqdm
sys.path.insert(0, ".")
from lib.dataset import CelebAHQIDIDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/celebahq",
        help="The root directory of the dataset.")
    parser.add_argument("--mask-dir", type=str, default="data/celebahq/mask",
        help="The root directory of random masks.")
    parser.add_argument("--out-dir", type=str, default="data/celebahq/IDI",
        help="The output directory.")
    parser.add_argument("--num-ref", type=int, default=5,
        help="The number of reference images available per each identity.")
    parser.add_argument("--split", type=str, default="test",
        help="Which split to export. Choices [train, val, test, all].")
    parser.add_argument("--seed", type=int, default=2023,
        help="The seed for randomization.")
    args = parser.parse_args()

    inpaint_region = ["lowerface", "eyebrow", "wholeface"]
    ds = CelebAHQIDIDataset(
        data_dir=args.data_dir,
        random_mask_dir=args.mask_dir,
        num_ref=args.num_ref,
        size=(1024, 1024),
        use_caption=False,
        split=args.split,
        inpaint_region=inpaint_region,
        seed=args.seed)
    
    for i, batch in enumerate(tqdm(ds)):
        id_name = batch["id"]
        if type(batch["id"]) is int:
            id_name = f"{batch['id']:04d}"
        prefix = f"{args.out_dir}/{id_name}"
        os.makedirs(f"{prefix}/infer_image", exist_ok=True)
        os.makedirs(f"{prefix}/infer_mask", exist_ok=True)
        os.makedirs(f"{prefix}/ref_image", exist_ok=True)
        indice = batch["all_indice"]
        iv_imgs = batch["infer_image"]
        for j, x in enumerate(iv_imgs):
            fname = f"{prefix}/infer_image/{indice[j]}.jpg"
            vutils.save_image(x.unsqueeze(0), fname)
            for k, m in enumerate(batch["infer_mask"][j]): # (num_masks, 3, H, W)
                mask_name = inpaint_region[k]
                fname = f"{prefix}/infer_mask/{indice[j]}_{mask_name}.png"
                vutils.save_image(m.unsqueeze(0), fname)
            fname = f"{prefix}/infer_mask/{indice[j]}_random.png"
            vutils.save_image(batch["random_mask"][j:j+1], fname)
        for j, x in enumerate(batch["ref_image"]):
            fname = f"{prefix}/ref_image/{indice[j + iv_imgs.shape[0]]}.jpg"
            vutils.save_image(x.unsqueeze(0), fname)
