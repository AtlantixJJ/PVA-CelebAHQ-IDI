"""Datasets."""
import torch, glob, json, os, dlib, cv2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
import torch.nn.functional as F
import numpy as np
from PIL import Image


# The prompt templates used for language-controlled inpainting
PROMPT_TEMPLATES_CONTROL = {
    "laughing": "photo of {}, laughing",
    "serious": "photo of {}, serious",
    "smile": "photo of {}, smiling",
    "sad": "photo of {}, looking sad",
    "angry": "photo of {}, angry",
    "surprised": "photo of {}, surprised",
    "beard": "photo of {}, has beard",
    
    "makeup": "photo of {}, with heavy makeup",
    "lipstick": "photo of {}, wearing lipstick",
    "funny": "photo of {}, making a funny face",
    "tongue": "photo of {}, putting the tongue out",

    "singing": "photo of {}, singing with a microphone",
    "cigarette": "photo of {}, smoking, has a cigarette",

    "eyeglass": "photo of {}, wearing eyeglasses",
    "sunglasses": "photo of {}, wearing sunglasses",
}


PROMPT_TEMPLATES_SMALL = {
    "person": "photo of {}"
}


def name2idx(names):
    """Possible names: <num>_*[.jpg][.png] or <num>[.jpg][.png]."""
    return [int(n[:-4].split("_")[0]) for n in names]


def predict_face_landmark(images):
    """Predicting face landmarks using dlib.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("pretrained/shape_predictor_68_face_landmarks.dat")

    image_landmarks = []
    for image in images:
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        bboxes = detector(image_np, 1)
        if len(bboxes) == 0:
            image_landmarks.append(None)
            continue
        bbox = bboxes[0]
        points = predictor(image_np, bbox).parts()
        image_landmarks.append(np.array([[p.x, p.y] for p in points]))
    return image_landmarks


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


def predict_landmark_bbox(images, min_height=0.2, expand_ratio=0.2, region_names=["lowerface", "eyebrow", "wholeface"]):
    """Create bounding boxes from the landmark file.
    Args:
        landmark_fpath: The path to the landmark file.
        resolution: The resolution of the images when landmarks were detected.
    """
    r_h, r_w = 1 + expand_ratio, 1 + expand_ratio
    resolution = images.shape[2]
    image_landmarks = predict_face_landmark(images)
    bbox_dict= {}
    for name in region_names:
        bbox_dict[name] = torch.zeros(images.shape[0], 4).long()

    valid = None
    for idx, lm_points in enumerate(image_landmarks):
        if lm_points is None:
            continue
        eye_brow = np.concatenate([lm_points[17:27], lm_points[36:48]])
        lower_face = lm_points[1:16]
        whole_face = lm_points
        point_sets = [lower_face, eye_brow, whole_face]
        # coordinate of left-upper and right-lower
        # note that dlib use (x, y) for column, row
        for name, points in zip(region_names, point_sets):
            bbox_dict[name][idx] = bbox_from_points(
                points, resolution, r_h, r_w, min_height)
            ny_min, nx_min, ny_max, nx_max = bbox_dict[name][idx]
            if (ny_max - ny_min) * (nx_max - nx_min) <= 10: # invalid mask
                image_landmarks[idx] = None
        valid = idx
    # when detected failed, fill with the last valid one
    for idx, lm_points in enumerate(image_landmarks):
        if lm_points is None:
            for name in region_names:
                bbox_dict[name][idx].copy_(bbox_dict[name][valid])
    return bbox_dict


class RotatingTensorDataset(torch.utils.data.Dataset):
    """Sample one image and left others as reference images."""
    def __init__(self, images,
                inpaint_region=["eyebrow", "lowerface", "wholeface"],
                use_random_mask=False):
        self.images = images
        self.inpaint_region = inpaint_region
        self.use_random_mask = use_random_mask
        self.bbox_dict = predict_landmark_bbox(images)
        self.rng = np.random.RandomState(1)
        if self.use_random_mask:
            self.mask_ds = RandomMaskDataset()

    def get_mask(self, image_path, orig_size):
        random_mask = self.ds.sample()[None, ..., 0]
        rname = np.random.choice(self.inpaint_region, (1,))[0]
        image_path = str(image_path)
        gidx = int(image_path[image_path.rfind("/") + 1 : -4])
        self.bboxes[rname][gidx]
        iv_mask = torch.zeros(1, self.size, self.size)
        scale = float(self.size) / orig_size 
        x_min, y_min, x_max, y_max = (self.bboxes[rname][gidx] * scale).long()
        iv_mask[..., x_min:x_max, y_min:y_max].fill_(1)
        iv_mask = (iv_mask + random_mask).clamp(min=0, max=1)
        return iv_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        indice = [i for i in range(len(self))]
        del indice[idx]
        infer_image = self.images[idx:idx+1].clone().detach()
        #random_mask = self.ds.sample()[None, ..., 0]
        rname = np.random.choice(self.inpaint_region, (1,))[0]
        ind = self.bbox_dict[rname][idx].long()
        mask = torch.zeros_like(infer_image).unsqueeze(0)
        mask[..., ind[0]:ind[2], ind[1]:ind[3]].fill_(1)
        if self.rng.randint(0, 2) == 0:
            infer_image = torch.flip(infer_image, (3,))
            mask = torch.flip(mask, (4,))
        dic = {
            "infer_image": infer_image, # (1, 3, H, W)
            "infer_mask": mask, # (1, 3, H, W)
            "prompt_template": ["photo of {}"] #["photo of {}"]
            }
        if len(indice) == 0:
            return dic

        self.rng.shuffle(indice)
        ref_images = self.images[indice].clone().detach()
        for i in range(len(indice)):
            if self.rng.randint(0, 2) == 0:
                ref_images[i] = torch.flip(ref_images[i], (2,))

        dic["ref_image"] = ref_images
        return dic


class MyStyleDataset(torch.utils.data.Dataset):
    """Load the images of MyStyle dataset"""

    def __init__(self, data_dir="data/MyStyle", size=(512, 512),
                 flip_p=0, num_ref=5, seed=None,
                 infer_folder="train", ref_folder="test", mask_folder="test_mask"):
        self.data_dir = data_dir
        self.num_ref = num_ref
        self.size = size
        self.flip_p = flip_p
        self.infer_folder = infer_folder
        self.ref_folder = ref_folder
        self.mask_folder = mask_folder
        self.transform = Compose([Resize(size), ToTensor()])
        self.mask_ds = RandomMaskDataset(size=self.size)
        self.ids = [int(i) for i in os.listdir(data_dir)]
        self.ids.sort()
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, id_idx):
        read_pil = lambda fp: Image.open(open(fp, "rb"))

        id_dir = os.path.join(self.data_dir, f"{id_idx:04d}")

        infer_paths = glob.glob(f"{id_dir}/{self.infer_folder}/*")
        infer_paths.sort()
        #infer_mask_paths = glob.glob(f"{id_dir}/{self.mask_folder}/*.png")
        #infer_mask_paths.sort()
        ref_file_path = os.path.join(id_dir, "ref.txt")
        if os.path.exists(ref_file_path):
            with open(ref_file_path, "r") as f:
                ref_paths = [os.path.join(id_dir, self.ref_folder, f.strip())
                             for f in f.readlines()][:self.num_ref]
        else:
            ref_paths = glob.glob(f"{id_dir}/ref_image/p*.jpg")
            ref_paths.sort()
            ref_paths = ref_paths[:self.num_ref]
        random_masks = torch.stack([self.mask_ds.sample(self.rng)
                                    for _ in infer_paths])
        # load and preprocess the image
        iv_imgs = torch.stack([self.transform(read_pil(fp))
                               for fp in infer_paths])
        rv_imgs = torch.stack([self.transform(read_pil(fp))
                               for fp in ref_paths])
        #mask = torch.stack([self.transform(read_pil(fp))[:1]
        #                    for fp in infer_mask_paths])
        #mask = (mask > 0.5).float()
        num = iv_imgs.shape[0] + rv_imgs.shape[0]
        temps = ["photo of {}" for _ in range(num)]
        return {"infer_image": iv_imgs,
                "ref_image": rv_imgs,
                #"infer_mask": mask.unsqueeze(1),
                "random_mask": random_masks,
                "all_indice": list(range(num)),
                "prompt_template": temps,
                "id": id_idx}


class RandomMaskDataset(torch.utils.data.Dataset):
    """Load a random mask from a folder."""

    def __init__(self, data_dir="data/celebahq/mask", size=(512, 512)):
        self.data_dir = data_dir
        self.size = size
        # mask should be stored in PNG format
        self.mask_paths = glob.glob(f"{data_dir}/*.png")
        self.mask_paths.sort()
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.mask_paths)
    
    def sample(self, rng=None):
        if rng is None:
            idx = np.random.randint(0, len(self))
        else:
            idx = rng.randint(0, len(self))
        return self[idx]

    def __getitem__(self, i):
        img = Image.open(self.mask_paths[i])
        if img.size[0] != self.size[0]:
            img = img.resize(self.size, resample=Image.Resampling.NEAREST)
        return self.to_tensor(img)


class CelebAHQIDIDataset(torch.utils.data.Dataset):
    """CelebAHQ IDentity Inpainting dataset.

    Args:
        num_ref: The number of reference images.
        split: train, test, val.
        loop_data: Whether to loop over identity or images in each batch.
            identity: all images of one identity, inference images are masked;
            image-ref: all reference images are masked once, reference selected from ref;
            image-all: all images are masked once, reference selected from both infer and ref.
        single_id: only return data of a single id. For compatibility with Textual Inversion.
    """
    def __init__(self, data_dir="data/celebahq", split="train",
                 image_folder="image", ann_folder="annotation",
                 random_mask_dir="data/celebahq/mask",
                 num_ref=5, size=(512, 512), flip_p=0, use_caption=True,
                 inpaint_region=["lowerface", "eyebrow", "wholeface"], 
                 loop_data="identity", single_id=None, seed=None):
        self.data_dir = data_dir
        self.split = split
        self.image_folder = image_folder
        self.ann_folder = ann_folder
        self.random_mask_dir = random_mask_dir
        self.num_ref = num_ref
        self.size = size
        self.flip_p = flip_p
        self.use_caption = use_caption
        self.loop_data = loop_data
        self.single_id = single_id
        self.inpaint_regions = inpaint_region
        if type(inpaint_region) is str:
            self.inpaint_regions = [inpaint_region]
        self.num_mask = len(self.inpaint_regions)
        self.rng = np.random.RandomState(seed)
        
        self.transform = Compose([Resize(size), ToTensor()])
        self.bboxes = torch.load(f"{data_dir}/{ann_folder}/region_bbox.pth")["bboxes"]
        n_images = self.bboxes[inpaint_region[0]].shape[0]
        ann_path = f"{data_dir}/{ann_folder}/idi-{num_ref}.json"
        self.ann = json.load(open(ann_path, "r"))

        if os.path.exists(random_mask_dir):
            self.mask_ds = RandomMaskDataset(
                data_dir=random_mask_dir, size=self.size)
        else:
            self.mask_ds = None
        
        self.prompt_templates = PROMPT_TEMPLATES_SMALL #PROMPT_TEMPLATES_LARGE
        self.prepend_text = "photo of {}. " # "photo of {}. "
        
        if use_caption:
            caption_path = f"{data_dir}/{ann_folder}/dialog/captions_hq.json"
            caption_dict = json.load(open(caption_path, "r"))
            self.captions = [] # The caption misses on 5380.jpg
            for n in [f"{i}.jpg" for i in range(n_images)]:
                text = self.prepend_text
                if n in caption_dict:
                    text = text + caption_dict[n]["overall_caption"]
                self.captions.append(text)
        else:
            self.captions = [self.prepend_text] * n_images
        self._create_loop_list()

    def _create_loop_list(self):
        split_names = ["train", "test", "val"]
        self.ann["all_ids"] = []
        for k in split_names:
            self.ann["all_ids"] += self.ann[f"{k}_ids"]
        self.ann["all_ids"].sort()
        self.ids = self.ann[f"{self.split}_ids"]
        if self.loop_data == "identity":
            if self.single_id is not None:
                #self.ids = [self.ids[self.single_id]]
                self.ids = [self.single_id]
        elif "image" in self.loop_data:
            key = self.loop_data.split("-")[1] # total, infer, ref
            self.ann["all_images"] = {key: []}
            for k in split_names:
                self.ann["all_images"][key] += self.ann[f"{k}_images"][key]
            self.image_indices = self.ann[f"{self.split}_images"][key]
            if self.single_id is not None:
                self.this_id = self.single_id
                #self.this_id = self.ids[self.single_id]
                m = self.ann["id2image"][self.this_id]
                all_indices = m["infer"] + m["ref"]
                self.image_indices = all_indices if key == "all" else m[key]

    def __len__(self):
        if self.loop_data == "identity":
            if hasattr(self, "start_indices"):
                return len(self.start_indices)
            return len(self.ids)
        elif "image" in self.loop_data:
            return len(self.image_indices)

    def _read_pil(self, fp):
        fpath = os.path.join(self.data_dir, self.image_folder, fp)
        return Image.open(open(fpath, "rb")).convert("RGB")

    def _fetch_id(self, index):
        if self.loop_data == "identity":
            if hasattr(self, "start_indices"):
                index = self.start_indices[index]
            id_name = self.ids[index]
            id_ann = self.ann["id2image"][self.ids[index]]
            iv_files, rv_files = id_ann["infer"], id_ann["ref"]
        elif "image" in self.loop_data:
            image_name = self.image_indices[index]
            id_name = self.ann["image2id"][name2idx([image_name])[0]]
            id_ann = self.ann["id2image"][id_name]
            iv_files = [image_name]
            if self.loop_data == "image-ref":
                rv_files = [f for f in id_ann["ref"] if f != image_name]
            elif self.loop_data == "image-infer":
                rv_files = id_ann["ref"]
            elif self.loop_data == "image-all":
                all_files = id_ann["infer"] + id_ann["ref"]
                other_files = [f for f in all_files if f != image_name]
                rv_files = list(np.random.choice(
                    other_files, (self.num_ref,), replace=False))
            else:
                raise NotImplementedError
        return iv_files, rv_files, id_name

    def _sample_prompt_temp(self, i):
        """Sample prompt template"""
        rand_temp = np.random.choice(list(self.prompt_templates.values()))
        if not self.use_caption or np.random.rand() < 0.5:
            return rand_temp
        return self.captions[i]

    def set_start_id(self, start_id):
        self.start_indices = np.arange(self.ids.index(start_id), len(self.ids))
        print(f"=> Reset dataset index starting from {self.start_indices[0]}")

    def __getitem__(self, index):
        iv_files, rv_files, id_idx = self._fetch_id(index)
        # sometimes training pipeline samples according to the order of
        # rv_files, so shuffle here
        self.rng.shuffle(rv_files)
        iv_indices = name2idx(iv_files)
        rv_indices = name2idx(rv_files)
        all_files = iv_files + rv_files
        all_indices = iv_indices + rv_indices
        #print(all_files, all_indices)

        # load and preprocess the image
        iv_imgs = [self._read_pil(fp) for fp in iv_files]
        rv_imgs = [self._read_pil(fp) for fp in rv_files]
        orig_size = iv_imgs[0].size[0]
        scale = float(self.size[0]) / orig_size
        iv_imgs = torch.stack([self.transform(img) for img in iv_imgs])
        rv_imgs = torch.stack([self.transform(img) for img in rv_imgs])
        mask = torch.zeros(len(iv_files), self.num_mask, *iv_imgs.shape[1:])
        for i, gidx in enumerate(iv_indices):
            # obtain and scale the bbox
            for j, rname in enumerate(self.inpaint_regions):
                x_min, y_min, x_max, y_max = (self.bboxes[rname][gidx] * scale).long()
                mask[i, j, :, x_min:x_max, y_min:y_max].fill_(1)
        for i in range(len(iv_imgs)):
            if self.rng.rand() < self.flip_p:
                mask[i] = torch.flip(mask[i], (3,))
                iv_imgs[i] = torch.flip(iv_imgs[i], (2,))
        for i in range(len(rv_imgs)):
            if self.rng.rand() < self.flip_p:
                rv_imgs[i] = torch.flip(rv_imgs[i], (2,))

        indices = self.rng.randint(0, self.num_mask, (iv_imgs.shape[0],))
        selected_mask = torch.stack([mask[i, indices[i]]
                                  for i in range(iv_imgs.shape[0])])
        if self.mask_ds is not None:
            random_masks = torch.stack([self.mask_ds.sample(self.rng)
                for _ in range(iv_imgs.shape[0])])
        else:
            random_masks = selected_mask
        temps = [self._sample_prompt_temp(i) for i in all_indices]
        return {"infer_image": iv_imgs,
                "ref_image": rv_imgs,
                "infer_mask": mask,
                "random_mask": (random_masks + selected_mask).clamp(max=1),
                "all_indice": all_indices,
                "all_file": all_files,
                "prompt_template": temps,
                "id": id_idx}


class SimpleDataset(torch.utils.data.Dataset):
    """
    Image-only datasets.
    """
    def __init__(self, data_path, size=None, transform=ToTensor()):
        self.size = size
        self.data_path = data_path
        self.transform = transform

        self.files = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(data_path) if files], [])
        self.files.sort()

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(os.path.join(self.data_path, fpath), "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.size:
                img = img.resize(self.size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)

