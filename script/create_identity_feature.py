"""Calculate the feature bank using FFHQ images."""
import sys, torch, argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, ".")
from lib.dataset import SimpleDataset
from lib.face_net import IDSimilarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        default="data/celebahq/image",
        help="The path to the images.")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    B = 64
    id_crit = IDSimilarity()
    id_crit_gt = IDSimilarity(model_type="glint")
    ds = SimpleDataset(args.dataset, size=(256, 256))
    fname2idx_fn = lambda fp: int(fp[fp.rfind("/")+1:fp.rfind(".")])
    _sorted_indices = np.argsort([fname2idx_fn(fp) for fp in ds.files])
    ds.files = list(np.array(ds.files)[_sorted_indices])
    dl = DataLoader(ds, B, shuffle=False, num_workers=16, drop_last=False)
    feats = torch.zeros(len(ds), 512)
    gt_feats = torch.zeros(len(ds), 512)
    count = 0
    for idx, x in enumerate(tqdm(dl)):
        B = x.shape[0]
        x = x.cuda() * 2 - 1
        feats[count:count+B].copy_(id_crit.extract_feats(x, False))
        gt_feats[count:count+B].copy_(id_crit_gt.extract_feats(x, False))
        count += B
    torch.save(feats, f"{args.dataset}/../annotation/id_feat.pth")
    torch.save(gt_feats, f"{args.dataset}/../annotation/id_feat_gt.pth")