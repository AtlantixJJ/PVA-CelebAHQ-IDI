"""Calculate the CLIP score."""
import torch, os, glob, json, clip, sys
import numpy as np
from transformers import CLIPModel, CLIPTextModel, CLIPVisionModel, CLIPFeatureExtractor, CLIPTokenizer
import torchvision.transforms.functional as ttf
from torchvision import transforms
import torch.nn.functional as F
from cleanfid.fid import get_files_features
sys.path.insert(0, ".")
from lib.misc import imread_to_tensor


PROMPT_TEMPLATES_CONTROL = {
    "person": "photo of {}",

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
normal_word = "a person"

data_dir = "data/celebahq"
cache_dir = "pretrained"
#inpaint_dir = f"{data_dir}/SDI2_FP_FT20_small_1.0"
output_dir = f"{data_dir}/result/clip_score"

clip_name = "openai/clip-vit-large-patch14"
sd_name = "stabilityai/stable-diffusion-2-inpainting"
clip_size = (224, 224)
device = "cuda"

def get_clip():
    return CLIPModel.from_pretrained(clip_name, cache_dir=cache_dir).cuda().eval()

def get_text_feats(clip):
    tokenizer = CLIPTokenizer.from_pretrained(sd_name, subfolder="tokenizer")
    long_lf_dic, short_lf_dic = {}, {}
    for p_name, prompt_temp in PROMPT_TEMPLATES_CONTROL.items():
        prompt = prompt_temp.format(normal_word)
        sent_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        clip_feat = clip.text_model(sent_ids).pooler_output
        clip_feat = clip.text_projection(clip_feat)[0]
        long_lf_dic[p_name] = clip_feat.clone().detach().cpu()
        sent_ids = tokenizer(p_name, return_tensors="pt").input_ids.to(device)
        clip_feat = clip.text_model(sent_ids).pooler_output
        clip_feat = clip.text_projection(clip_feat)[0]
        short_lf_dic[p_name] = clip_feat.clone().detach().cpu()
    return short_lf_dic, long_lf_dic

def get_vision_func(clip):
    _c = CLIPFeatureExtractor.from_pretrained(clip_name, cache_dir=cache_dir)
    return lambda x: clip.visual_projection(clip.vision_model(
                ttf.normalize(x / 255., _c.image_mean, _c.image_std)).pooler_output)


class CLIPScoreMetric(object):
    def __init__(self):
        self.cache = {}
        self.model, clip_preprocess = clip.load("ViT-B/32", device="cpu")  # cpu allows for fp32 loading.
        self.model = self.model.to('cuda')
        self.model.eval()

        self.clip_preprocess = transforms.Compose(  # Already un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            clip_preprocess.transforms[:2] +  # Skip ToRGB and ToTensor
            clip_preprocess.transforms[4:]
        )

    def __call__(self, img, text):
        assert len(text) == img.shape[0]

        with torch.no_grad():
            if text[0] in self.cache: # single prompt case
                text_feature = self.cache[text[0]]
            else:
                text_feature = self.model.encode_text(clip.tokenize(text).to('cuda'))
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
                self.cache[text[0]] = text_feature
            x = self.clip_preprocess(img)
            img_feature = self.model.encode_image(x.to('cuda'))
            img_feature /= img_feature.norm(dim=-1, keepdim=True)

            clip_score = torch.einsum('bz,bz->b', img_feature, text_feature)

        return clip_score


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    clip_score_metric = CLIPScoreMetric()
    methods = [
        #"SDI2",
        #"MyStyle",
        #"Paint by Example",
        #"LORA-1", "LORA-6"
        
        "CD-1", "CD-6",
        "TI-1", "TI-6",

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

        #"Inpaint_SDI2_LORA_control_1.0",
        #"Inpaint_SDI2_LORA_control_6.0",
        
        "Inpaint_SDI2_CD_control_1.0",
        "Inpaint_SDI2_CD_control_6.0",
        "Inpaint_SDI2_TI_control_1.0",
        "Inpaint_SDI2_TI_control_6.0",

        #"Inpaint_CrossQKV_FT40_R1_control_1.0",
        #"Inpaint_CrossQKV_FT40_R2_control_1.0",
        #"Inpaint_CrossQKV_FT40_R3_control_1.0",
        #"Inpaint_CrossQKV_FT40_R4_control_1.0",
        #"Inpaint_CrossQKV_FT40_R5_control_1.0",
        #"Inpaint_CrossQKV_FT40_R1_control_6.0",
        #"Inpaint_CrossQKV_FT40_R2_control_6.0",
        #"Inpaint_CrossQKV_FT40_R3_control_6.0",
        #"Inpaint_CrossQKV_FT40_R4_control_6.0",
        #"Inpaint_CrossQKV_FT40_R5_control_6.0",
    ]
    res_dic = {}
    for method in methods:
        edit_dir = f"{data_dir}/{method}"
        res_dic[method] = {}
        for p_name, prompt_temp in PROMPT_TEMPLATES_CONTROL.items():
            if p_name == "person":
                continue
            print(method, p_name)
            files = glob.glob(f"{edit_dir}/*{p_name}.*")
            files.sort()
            prompt = prompt_temp.format("a person")
            res = []
            images = [imread_to_tensor(fp).unsqueeze(0) for fp in files]
            for image in images:
                clip_score = clip_score_metric(image, [prompt])
                res.append(clip_score)
            res = torch.cat(res)
            res_dic[method][p_name] = (
                float(res.mean().cpu().item()),
                float(res.std().cpu().item()))
            fp = os.path.join(output_dir, "clip_score.json")
            json.dump(res_dic, open(fp, "w"), indent=2)