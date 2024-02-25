"""Create re-organized CelebAHQ-ID inpainting dataset.
For creating the MyStyle dataset:
python script/create_id_inpaint_dataset.py --root-dir data/MyStyle --test-ratio 1 --val-ratio 0 --num-refs 5,10,15,20,30,40,50
"""
import sys, os, torch, argparse, json, glob, pprint
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from datasets import SimpleDataset, IDInpaintDataset, name2idx


REGION_NAMES = ["leye", "reye", "eye", "eyebrow", "lowerface", "wholeface", "mouth", "nose"]


def dlib_landmarks(ROOT_DIR, dataset):
    """Detect the face landmark using Dlib.
    Args:
        ROOT_DIR: The root directory of the dataset.
        dataset: The name of image folder of the dataset.
    """
    import dlib, cv2

    os.makedirs(f"{ROOT_DIR}/landmark_visualization", exist_ok=True)
    os.makedirs(f"{ROOT_DIR}/failed_detection", exist_ok=True)
    os.makedirs(f"{ROOT_DIR}/annotation", exist_ok=True)
    image_paths = glob.glob(f"{ROOT_DIR}/{dataset}/*")
    # The sorting is not numerical, but it does not matter
    # Latter, we are going to use file name as index.
    image_paths.sort()

    dlib.cuda.set_device(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("pretrained/shape_predictor_68_face_landmarks.dat")

    dic = {}
    for i, fp in enumerate(tqdm(image_paths)):
        fname = fp[fp.rfind("/") + 1:]
        image = dlib.load_rgb_image(fp)
        bboxes = detector(image, 1)
        if len(bboxes) == 0:
            cv2.imwrite(f"{ROOT_DIR}/failed_detection/{fname}", image[..., ::-1])
            continue
        bbox = bboxes[0]
        points = predictor(image, bbox).parts()
        landmarks = [[p.x, p.y] for p in points]
        dic[fname] = {"image": {"face_landmarks": landmarks}}

        if i < 100:
            for x, y in landmarks:
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            cv2.imwrite(f"{ROOT_DIR}/landmark_visualization/{fname}", image[..., ::-1])
        if i % 1000 == 0:
            with open(f"{ROOT_DIR}/annotation/landmarks.json", "w") as f:
                json.dump(dic, f)
    with open(f"{ROOT_DIR}/annotation/landmarks.json", "w") as f:
        json.dump(dic, f)
    return dic


def bbox_from_points(points, S, r_h=1.2, r_w=1.2, min_height=0.2):
    """Create a box bounding all the points with margin.
    Returns:
        A Tensor of [upper left x, y, lower right x, y]
    """
    x_min, y_min = points.min(0)
    x_max, y_max = points.max(0)
    c_x, c_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    h, w = x_max - x_min, y_max - y_min
    w = int(max(r_w * w, min_height * S))
    h = int(max(r_h * h, min_height * S))
    nx_min = max(int(c_x - h / 2), 0)
    nx_max = min(int(c_x + h / 2), S)
    ny_min = max(int(c_y - w / 2), 0)
    ny_max = min(int(c_y + w / 2), S)
    return torch.Tensor([ny_min, nx_min, ny_max, nx_max])


def create_bbox(data_dir, landmark_fpath, resolution, min_height=0.2, expand_ratio=0.2):
    """Create bounding boxes from the landmark file.
    Args:
        landmark_fpath: The path to the landmark file.
        resolution: The resolution of the images when landmarks were detected.
    """
    r_h, r_w = 1 + expand_ratio, 1 + expand_ratio
    with open(landmark_fpath, "r", encoding="ascii") as f:
        dict = json.load(f)
    # use the same folder as the landmark file as output directory
    out_dir = landmark_fpath[:landmark_fpath.rfind("/")]

    length = len(os.listdir(os.path.join(data_dir, "image")))
    data = {
        "landmarks": torch.zeros(length, 68, 2).long(),
        "bboxes": {}
    }
    for name in REGION_NAMES:
        data["bboxes"][name] = torch.zeros(length, 4).long()

    for fname, info in dict.items():
        lm_points = np.array(info["image"]["face_landmarks"])
        # we assume the image format to be 00000.jpg or 00000_0000.jpg
        idx = int(fname[:-4].split("_")[0])
        data["landmarks"][idx] = torch.from_numpy(lm_points).flip(1)
        eye = lm_points[36:48]
        eye_brow = np.concatenate([lm_points[17:27], eye])
        l_eye = lm_points[36:42]
        r_eye = lm_points[43:48]
        lower_face = lm_points[1:16]
        whole_face = lm_points
        mouth = lm_points[48:68]
        nose = lm_points[27:36]
        point_sets = [l_eye, r_eye, eye, eye_brow, lower_face, whole_face, mouth, nose]
        # coordinate of left-upper and right-lower
        # note that dlib use (x, y) for column, row
        for name, points in zip(REGION_NAMES, point_sets):
            data["bboxes"][name][idx] = bbox_from_points(
                points, resolution, r_h, r_w, min_height)
    torch.save(data, f"{out_dir}/region_bbox.pth")
    return data


def block_cdist(A, B, bs=1000):
    """Calculate pairwise distance matrix using blocks to save GPU memory."""
    dist = torch.zeros(A.shape[0], B.shape[0]).to(A)
    bs = min(A.shape[0], bs)
    for i in range(A.shape[0] // bs + 1):
        st1, ed1 = bs * i, min(A.shape[0], bs * (i + 1))
        if st1 >= ed1:
            break
        for j in range(B.shape[0] // bs + 1):
            st2, ed2 = bs * j, min(B.shape[0], bs * (j + 1))
            if st2 >= ed2:
                break
            r = torch.cdist(A[st1:ed1].unsqueeze(0), B[st2:ed2].unsqueeze(0))
            dist[st1:ed1, st2:ed2].copy_(r[0])
    return dist


def merge_duplicate_group(dg_from, dg_to):
    """Merge two lists of sets.
    """
    for this_set in dg_from:
        new = True
        for i in range(len(dg_to)):
            if len(dg_to[i].intersection(this_set)) > 0:
                dg_to[i] = dg_to[i].union(this_set)
                new = False
        if new:
            dg_to.append(this_set)


def update_duplicate_group(dup_groups, dist, threshold=10):
    """Update a duplicate group sets using the pairwise distance."""
    for i in tqdm(range(dist.shape[0])):
        dist[i, i] = 1e10
        invalid_indice = list(torch.where(dist[i] < threshold)[0].cpu().numpy())
        if len(invalid_indice) > 0:
            this_set = set([i] + invalid_indice)
            merge_duplicate_group([this_set], dup_groups)


def find_duplicate(data_dir):
    """Find the duplicate images in the dataset."""

    os.makedirs(f"{data_dir}/duplicate_images", exist_ok=True)
    S = 128
    ds = SimpleDataset(f"{data_dir}/image", size=(S, S))
    ds.files.sort(key=lambda x: int(x[:-4].split("_")[0]))
    dl = DataLoader(ds, batch_size=10, shuffle=False, num_workers=4)
    images = torch.zeros(len(ds), 3, S, S).cuda()
    count = 0
    for x in tqdm(dl):
        images[count:count+x.shape[0]].copy_(x)
        count += x.shape[0]
    v = images.view(images.shape[0], -1)

    dup_groups = []
    dist = block_cdist(v, v)
    update_duplicate_group(dup_groups, dist)
    del dist
    torch.cuda.empty_cache()
    rv = images.flip(3).view(len(ds), -1)
    rdist = block_cdist(v, rv)
    update_duplicate_group(dup_groups, rdist)
    for i, g in enumerate(dup_groups):
        l = list(g)
        l.sort()
        name = str(l)[1:-1].replace(", ", "_")
        vutils.save_image(images[l],
            f"{data_dir}/duplicate_images/dup{i:02d}_{name}.png")
    del rdist
    torch.cuda.empty_cache()
    with open(f"{data_dir}/annotation/duplicate.txt", "w") as f:
        for g in dup_groups:
            l = list(g)
            l.sort()
            f.write(" ".join([str(x) for x in l]) + "\n")
    return dup_groups


def create_id_annotation(data_dir, dup_groups, test_ratio=0.3, val_ratio=0.1, num_ref=5):
    ds = IDInpaintDataset(data_dir, (1024, 1024),
        num_ref=num_ref, inpaint_region=["lowerface", "eyebrow"])
    n_image = len(ds.image_paths)
    id_names = ds.id_names
    id_names.sort()
    n_ids = len(id_names)
    test_sep = int(test_ratio * n_ids)
    val_sep = int(val_ratio * n_ids)
    np.random.RandomState(2022).shuffle(id_names)
    
    test_ids = [n for n in id_names[:test_sep]]
    if val_ratio > 0:
        train_ids = [n for n in id_names[test_sep:-val_sep]]
        val_ids = [n for n in id_names[-val_sep:]]
    else:
        train_ids = [n for n in id_names[test_sep:]]
        val_ids = []
    id_names.sort()
    test_ids.sort()
    train_ids.sort()
    val_ids.sort()
    split_names = ["train", "val", "test"]
    dic = {
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
        "val_ids": list(val_ids),
        "duplicate_images": dup_groups,
        "discard_ids": ds.discard_id_names,
        "id2image": {}}
    for key in split_names:
        dic[f"{key}_images"] = {"all": [], "infer": [], "ref": []}
    image2id = [""] * n_image
    n_infer = 0
    for id_idx, id_name in enumerate(id_names):
        infer_images, ref_images = ds._fetch_id(id_idx)
        all_images = infer_images + ref_images
        for image_idx in name2idx(all_images):
            image2id[image_idx] = id_name
        infer_images.sort()
        ref_images.sort()
        if id_name in train_ids:
            key = "train_images"
        elif id_name in val_ids:
            key = "val_images"
        else:
            key = "test_images"
        dic[key]["all"].extend(all_images)
        dic[key]["infer"].extend(infer_images)
        dic[key]["ref"].extend(ref_images)
        dic["id2image"][id_name] = {
            "infer": infer_images,
            "ref": ref_images}
        n_infer += len(infer_images)
    dic["image2id"] = image2id
    n_dic = {k: len(dic[f"{k}_images"]["all"]) for k in split_names}
    n_valid_total = sum([v for v in n_dic.values()])
    dic["meta"] = {
        "n_ref": num_ref,
        "n_images": n_valid_total,
        "n_infer_images": n_infer,
        "n_ids": len(train_ids) + len(val_ids) + len(test_ids),
        "n_discard_ids": len(ds.discard_id_names),
        "n_discard_images": n_image - n_valid_total,
        "n_train_ids": len(train_ids),
        "n_train_images": n_dic["train"],
        "n_val_ids": len(val_ids),
        "n_val_images": n_dic["val"],
        "n_test_ids": len(test_ids),
        "n_test_images": n_dic["test"],
        "resolution": 1024}
    ann_fp = os.path.join(data_dir, "annotation", f"idi-{num_ref}.json")
    json.dump(dic, open(ann_fp, "w"), indent=2)
    return dic


def create_celebaid_inpaint(data_dir, test_ratio=0.3, val_ratio=0.1):
    """Create CelebAHQ-IDI dataset.
    """

    if not os.path.exists(f"{data_dir}/annotation/landmarks.json"):
        print(f"=> Detecting landmarks of {data_dir} now...")
        dlib_landmarks(data_dir, "image")
    #print(f"=> Loading landmark file from {data_dir}/annotation/landmarks.json")
    #landmark_dict = json.load(open(f"{data_dir}/annotation/landmarks.json", "r"))
    
    if not os.path.exists(f"{data_dir}/annotation/region_bbox.pth"):
        print(f"=> Region bounding box file is missing. Creating now...")
        create_bbox(data_dir, f"{data_dir}/annotation/landmarks.json", 1024)
    #print(f"=> Load bounding box file from {data_dir}/annotation/region_bbox.pth")
    #bbox_dict = torch.load(f"{data_dir}/annotation/region_bbox.pth")
        
    if not os.path.exists(f"{data_dir}/annotation/duplicate.txt"):
        print(f"=> Checking duplicate images...")
        find_duplicate(data_dir)
    print(f"=> Loading duplicate files from {data_dir}/annotation/duplicate.txt")
    dup_groups = []
    with open(f"{data_dir}/annotation/duplicate.txt", "r") as f:
        for l in f.readlines():
            if len(l) < 2: continue
            dup_groups.append([int(i) for i in l.split(" ")])

    print(f"=> Creating annotation file...")
    num_refs = [int(i) for i in args.num_refs.split(",")]
    for num_ref in num_refs:
        dict = create_id_annotation(data_dir, dup_groups,
            test_ratio, val_ratio, num_ref)
        pprint.pprint(dict["meta"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="data/celebahq",
        help="The root directory of the dataset.")
    parser.add_argument("--test-ratio", type=float, default=0.3,
        help="The ratio of test identities.")
    parser.add_argument("--val-ratio", type=float, default=0.1,
        help="The ratio of validation identities.")
    parser.add_argument("--num-refs", type=str, default="1,2,3,4,5,10,15,20",
        help="The number of reference images.")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    create_celebaid_inpaint(args.root_dir, args.test_ratio, args.val_ratio)


