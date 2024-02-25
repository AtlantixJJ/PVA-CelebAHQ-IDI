"""Defines the PVA pathway."""
import torch, os, glob
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPAttention
from transformers import CLIPModel, CLIPTextModel, CLIPVisionModel, CLIPFeatureExtractor, CLIPTokenizer
from tqdm import tqdm

from lib.diffusers import PVAInpaintPipeline, AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, DDIMScheduler, BasicTransformerBlock
from lib.dataset import PROMPT_TEMPLATES_SMALL, RotatingTensorDataset
from lib.face_net import FaceNet
from lib.lion_pytorch import Lion
from lib.visualizer import ref_inpaint_visualize_html
from lib.misc import pil2torch
import torchvision.transforms.functional as ttf
import torchvision.utils as vutils


def pva_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif self.cross_attention_norm:
        encoder_hidden_states = self.norm_cross(encoder_hidden_states)

    text_query = self.to_q(hidden_states)
    text_query = self.head_to_batch_dim(text_query)

    if hasattr(self, "to_visual_q"):
        text_length = 77 # hard coded the length of text
        text_feats = encoder_hidden_states[:, :text_length]
        text_key = self.to_k(text_feats)
        text_value = self.to_v(text_feats)
        text_key = self.head_to_batch_dim(text_key)
        text_value = self.head_to_batch_dim(text_value)
        
        # Parallel Visual Attention
        visual_query = self.to_visual_q(hidden_states)
        visual_query = self.head_to_batch_dim(visual_query)

        visual_feats = encoder_hidden_states[:, text_length:]
        visual_key = self.to_visual_k(visual_feats)
        visual_value = self.to_visual_v(visual_feats)
        visual_key = self.head_to_batch_dim(visual_key)
        visual_value = self.head_to_batch_dim(visual_value)

        text_scores = torch.baddbmm(
            torch.empty(text_query.shape[0], text_query.shape[1], text_key.shape[1],
                dtype=text_query.dtype, device=text_query.device),
            text_query,
            text_key.transpose(-1, -2),
            beta=0, alpha=self.scale)
        visual_scores = torch.baddbmm(
            torch.empty(visual_query.shape[0], visual_query.shape[1], visual_key.shape[1],
                dtype=visual_query.dtype, device=visual_query.device),
            visual_query,
            visual_key.transpose(-1, -2),
            beta=0, alpha=self.scale)
        attention_scores = torch.cat([text_scores, visual_scores], 2)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(text_query.dtype)
        value = torch.cat([text_value, visual_value], 1)
        hidden_states = torch.bmm(attention_probs, value)
    else:
        query = text_query
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        attention_probs = self.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)

    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states


def change_forward(unet):
    """Change the forward function and add a new attention for visual features."""
    for m in unet.children():
        if isinstance(m, BasicTransformerBlock):
            bound_method = pva_forward.__get__(m.attn2, m.attn2.__class__)
            setattr(m.attn2, 'forward', bound_method)

            query_dim = m.attn2.to_q.in_features
            inner_dim = m.attn2.to_q.out_features
            cross_attention_dim = m.attn2.to_k.in_features
            m.attn2.to_visual_q = torch.nn.Linear(query_dim, inner_dim,
                bias=(m.attn2.to_q.bias is not None))
            m.attn2.to_visual_k = torch.nn.Linear(cross_attention_dim, inner_dim,
                bias=(m.attn2.to_k.bias is not None))
            m.attn2.to_visual_v = torch.nn.Linear(cross_attention_dim, inner_dim,
                bias=(m.attn2.to_v.bias is not None))
            m.attn2.to_visual_q.load_state_dict(m.attn2.to_q.state_dict())
            m.attn2.to_visual_k.load_state_dict(m.attn2.to_k.state_dict())
            m.attn2.to_visual_v.load_state_dict(m.attn2.to_v.state_dict())
            m.attn2.to(next(m.parameters()).device)

        else:
            change_forward(m)


class PVADiffusion(torch.nn.Module):
    """Parallel Visual Attention Pathway for Stable Diffusion."""

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.cfg = config
        self.tune_clip = config["clip_image_enc"]["tune_params"]
        self.tune_unet = config["unet"]["tune_params"]
        self.unet_attention = config["unet"]["attention"]
        self.tune_facenet = config["face_net"]["tune_params"]
        self.face_net_train = FaceNet().to(device).eval()
        self.face_net_eval = FaceNet().to(device).eval()
        self._load_clip(config)
        self._load_diffusion(config)

        if config["text_in"] != "None":
            # The input to Text encoder is modified
            self._resize_text_embedding(config)

        self.trans_net = None
        if "ID" in [config["text_in"], config["text_out"]]:
            self.trans_net = TransformNetwork(**config["id_trans_net"]["arch"])
            self.trans_net.to(self.device).train()

        self.inpaint_pipe = PVAInpaintPipeline(self.vae, self.clip_text_enc,
            self.tokenizer, self.unet, self.scheduler,
            safety_checker=None, feature_extractor=None)

    def _load_diffusion(self, config):
        """Load the diffusion modules."""
        dic = {k: config[k] for k in ["pretrained_model_name_or_path", "cache_dir"]}
        self.vae = AutoencoderKL.from_pretrained(
            subfolder="vae", **dic).to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(
            subfolder="unet", **dic).to(self.device).eval()
        if self.unet_attention == "PVA":
            change_forward(self.unet)
        class_func = DDIMScheduler
        if config["sampling"]["sampler"] == "dpmsolver":
            class_func = DPMSolverMultistepScheduler
        self.scheduler = class_func.from_config(
            subfolder="scheduler", **dic)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            subfolder="tokenizer", **dic)
        self.clip_text_enc = CLIPTextModel.from_pretrained(
            subfolder="text_encoder", **dic).to(self.device).eval()

    def _load_clip(self, config):
        """Load the CLIP modules."""
        # SD2 uses a new CLIP model
        if "stable-diffusion-2" in config["pretrained_model_name_or_path"]:
            clip_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        else:
            clip_name = "openai/clip-vit-large-patch14"

        #self.clip_text_enc = CLIPTextModel(clip.config.text_config)
        #self.clip_text_enc.load_state_dict(sd, strict=False)
        #self.clip_text_enc.to(self.device).eval()

        # The CLIP vision model is only needed in some cases
        if "CLIP" in config["text_in"] or "CLIP" in config["text_out"]:
            clip = CLIPModel.from_pretrained(
                clip_name, cache_dir=config["cache_dir"])
            sd = clip.state_dict()

            _c = CLIPFeatureExtractor.from_pretrained(
                clip_name, cache_dir=config["cache_dir"])
            self.clip_image_proc = lambda x: ttf.normalize(F.interpolate(
                x, (224, 224), mode="bicubic"), _c.image_mean, _c.image_std)

            self.clip_image_enc = CLIPVisionModel(
                clip.config.vision_config)
            self.clip_image_enc.load_state_dict(sd, strict=False)
            self.clip_image_enc.to(self.device).eval()
            self.clip_image_proj = torch.nn.Linear(
                clip.vision_embed_dim, clip.projection_dim, bias=False)
            self.clip_image_proj.load_state_dict(
                clip.visual_projection.state_dict())
            self.clip_image_proj.to(self.device).eval()
            del clip, sd
        else:
            self.clip_image_enc = None
            self.clip_image_proj = None

    def _resize_text_embedding(self, config):
        """Resize the text embedding size of CLIP text encoder."""
        mt_cfg = config["modifier_token"]
        tokens = [mt_cfg["name"], mt_cfg["init_token"]]
        self.tokenizer.add_tokens(tokens[:1])
        self.modifier_token_id, self.init_token_id = \
            self.tokenizer.convert_tokens_to_ids(tokens)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.clip_text_enc.resize_token_embeddings(len(self.tokenizer))

        # initialize the new vector
        token_embeds = self.clip_text_enc.get_input_embeddings().weight.data
        #token_embeds[self.modifier_token_id].copy_(token_embeds[self.init_token_id])
        token_embeds[self.modifier_token_id].fill_(0.)
        self.tuned_embedding = TunedEmbedding(
            token_embeds, self.modifier_token_id,
            mode=self.cfg["modifier_token"]["mode"]).to(self.device)
        vec = token_embeds[self.init_token_id].clone().detach().to(self.device)
        #vec = torch.zeros_like(token_embeds[self.init_token_id]).to(self.device)
        self.tuned_embedding.modifier_vec = torch.nn.Parameter(vec)
        self.clip_text_enc.text_model.embeddings.token_embedding = \
            self.tuned_embedding

    def _post_visual_feature(self, x):
        """Append visual feature after CLIP text encoder."""
        N, N_REF = x.shape[:2]
        x = x.view(N * N_REF, *x.shape[2:])
        visual_feat = None
        if self.cfg["text_out"] == "CLIP-A":
            x = self.clip_image_proc(x)
            if self.tune_clip in ["Proj", "None"]:
                with torch.no_grad():
                    visual_feat = self.clip_image_enc(x)[0]
            else:
                visual_feat = self.clip_image_enc(x)[0]
            visual_feat = self.clip_image_proj(visual_feat)
            visual_feat = visual_feat.view(N, -1, visual_feat.shape[-1])

        if self.cfg["text_out"] == "ID":
            id_feat = self.face_net_train(x * 2 - 1) # (N * N_REF, 512)
            visual_feat = self.trans_net(id_feat.view(N, N_REF, -1))

        return visual_feat

    def _pre_visual_feature(self, x, text_id=None):
        """Insert visual feature before CLIP text encoder."""
        if self.cfg["text_in"] == "Token":
            visual_feat = None
            self.tuned_embedding.set_extra_feature(self.tuned_embedding.modifier_vec)
        else:
            N, N_REF = x.shape[:2]
            x = x.view(N * N_REF, *x.shape[2:])
            if self.cfg["text_in"] == "ID":
                id_feat = self.face_net_train(x * 2 - 1) # (N * N_REF, 512)
                visual_feat = self.trans_net(id_feat.view(N, N_REF, -1))
            elif self.cfg["text_in"] == "CLIP-S":
                x = self.clip_image_proc(x)
                if self.tune_clip in ["Proj", "None"]:
                    with torch.no_grad():
                        visual_feat = self.clip_image_enc(x)[1]
                else:
                    visual_feat = self.clip_image_enc(x)[1]
                visual_feat = self.clip_image_proj(visual_feat)

        if text_id is not None:
            return self.clip_text_enc(text_id)[0], visual_feat

        return None, visual_feat

    def calc_visual_feature(self, x,
            text_id=None, text_feat=None):
        """Calculate the visual feature."""

        pre_visual_feat = None
        if self.cfg["text_in"] != "None":
            text_feat, pre_visual_feat = self._pre_visual_feature(x, text_id)

        if self.cfg["text_out"] != "None":
            post_visual_feat = self._post_visual_feature(x)
            if text_feat is not None:
                context_feat = torch.cat([text_feat, post_visual_feat], 1)
            else:
                context_feat = None
           
            return context_feat, post_visual_feat
        
        return text_feat, pre_visual_feat

    def setup_parameters(self):
        """Return the trainable parameters given the configuration and set requires grad accordingly."""
        self.requires_grad_(False)
        params = []

        # Enable gradient checkpointing to save memory
        if self.cfg["memory_saving"]:
            self.unet.enable_gradient_checkpointing()
            if hasattr(self, "clip_image_enc"):
                self.clip_image_enc.gradient_checkpointing_enable()

        # Setup the transformer if it is not fixed
        trainable = "fix" not in self.cfg["id_trans_net"]
        if "ID" in [self.cfg["text_in"], self.cfg["text_out"]] and trainable:
            self.trans_net.requires_grad_(True).train()
            params.append({
                "name": "trans_net",
                "params": self.trans_net.parameters(),
                "lr": self.cfg["id_trans_net"]["training"]["lr"]
                })
        
        # Add the modifier token to training if it is not fixed
        trainable = "fix" not in self.cfg["modifier_token"]
        if self.cfg["text_in"] == "Token" and trainable:
            self.tuned_embedding.modifier_vec.requires_grad = True
            params.append({
                "name": "modifier_token",
                "params": self.tuned_embedding.modifier_vec,
                "lr": self.cfg["modifier_token"]["lr"]})

        # Tune all parameters of the pretrained FaceNet
        if self.tune_facenet == "All":
            # set training in face_net cause large performance drop
            self.face_net_train.requires_grad_(True).eval()
            params.append({
                "name": "face_net",
                "params": self.face_net_train.parameters(),
            })

        # Tune the CLIP image encoder projection layer
        if self.tune_clip != "None":
            self.clip_image_proj.requires_grad_(True).train()
            params.append({
                "name": "clip_image_proj",
                "params": self.clip_image_proj.parameters(),
                })
        
        # Tune the CLIP image encoder
        if self.tune_clip not in ["None", "Proj"]:
            self.clip_image_enc.train()
            # Tune the last few layers
            if "Last" in self.tune_clip:
                num = int(self.tune_clip.split("-")[1])
                for i in range(1, num + 1):
                    m = self.clip_image_enc.vision_model.encoder.layers[-i]
                    m.requires_grad_(True).train()
                    params.append({
                        "name": f"clip_image_enc_last{i}",
                        "params": m.parameters()
                        })
            # Tune the attention layers
            for m in self.clip_image_enc.modules():
                if isinstance(m, CLIPAttention):
                    if self.tune_clip == "QKV":
                        m.requires_grad_(True)
                        params.append({
                            "name": "clip_image_enc_QKV",
                            "params": m.parameters()
                            })

        if self.tune_unet != "None":
            self.unet.train()

        def select_unet_params():
            # select some parameters of UNet
            for name, params in self.unet.named_parameters():
                # Custom Diffusion trains the Cross Attention
                if self.tune_unet == "CrossQKV" and 'attn2' in name:
                    params.requires_grad = True
                    yield params
                # Train Cross Attention and Self Attention
                elif self.tune_unet == "QKV" and 'attn' in name:
                    params.requires_grad = True
                    yield params
                # Train the new parameters of PVA modules
                elif self.tune_unet == "PVA" and 'to_visual' in name:
                    params.requires_grad = True
                    yield params

        if self.tune_unet != "None":
            params.append({
                "name": f"unet_{self.tune_unet}",
                "params": select_unet_params()
            })

        return params

    def personalized_prompt(self, prompt_template):
        """Return the prompt for personalized input."""
        if self.cfg["text_in"] != "None":
            return prompt_template.format(self.cfg["personalized_words"])
        return prompt_template.format(self.cfg["normal_words"])

    def normal_prompt(self, prompt_template):
        """Return the prompt for normal input."""
        return prompt_template.format(self.cfg["normal_words"])

    def restore(self, data):
        """Load components from saved weights.
        """
        if self.trans_net is not None and "transform_network" in data:
            self.trans_net.load_state_dict(data["transform_network"])
            print("=> Transformer loaded")
        
        # custom diffusion uses a different naming and format
        if "modifier_token" in data:
            data["modifier_vec"] = list(data["modifier_token"].values())[0]

        if "modifier_vec" in data:
            if isinstance(data["modifier_vec"], torch.Tensor):
                v = data["modifier_vec"]
            elif data["modifier_vec"] is dict:
                v = list(data["modifier_vec"].values())[0]
            self.tuned_embedding.modifier_vec = torch.nn.Parameter(v.to(self.device))
            print("=> Modifier loaded")

        if "face_net" in data:
            self.face_net_train.load_state_dict(data["face_net"])
            print("=> FaceNet loaded")

        if "clip_vision_projection" in data:
            self.clip_image_proj.load_state_dict(
                data["clip_vision_projection"])

        if "clip_vision_model" in data:
            self.clip_image_enc.load_state_dict(
                data["clip_vision_model"], strict=False)
            print("=> CLIP vision encoder loaded")

        if "unet" in data:
            self.unet.load_state_dict(data["unet"], strict=False)
            print("=> UNet loaded")

    def simple_state_dict(self, accelerator):
        """Saving training parameters. Accelerator is required during training. args is also required."""
        dic = {}
        if hasattr(self.tuned_embedding, "modifier_vec"):
            dic["modifier_vec"] = self.tuned_embedding.modifier_vec.clone().detach()

        if self.trans_net is not None and "fix" not in self.cfg["id_trans_net"]:
            dic["transform_network"] = accelerator.unwrap_model(
                self.trans_net).state_dict()

        if self.tune_clip != "None":
            vision_model = accelerator.unwrap_model(self.clip_image_enc)
            proj_model = accelerator.unwrap_model(self.clip_image_proj)
            dic["clip_vision_model"] = vision_model.state_dict()
            dic["clip_vision_projection"] = proj_model.state_dict()

        if self.tune_unet != "None":
            model = accelerator.unwrap_model(self.unet)
            sd = model.state_dict()
            if self.tune_unet == "CrossQKV":
                sd = {k:v for k, v in sd.items() if "attn2" in k}
            if self.tune_unet == "PVA":
                sd = {k:v.clone().detach().cpu()
                      for k, v in sd.items() if "attn2" in k and "visual" in k}
            dic["unet"] = sd

        if self.tune_facenet != "None":
            model = accelerator.unwrap_model(self.face_net_train)
            dic["face_net"] = model.state_dict()

        return dic

    def prepare_accelerator(self, accelerator):
        #self.tuned_embedding = accelerator.prepare(self.tuned_embedding)
        self.unet, self.face_net_train, self.clip_text_enc = accelerator.prepare(
            self.unet, self.face_net_train, self.clip_text_enc)

        if self.clip_image_enc is not None:
            self.clip_image_enc, self.clip_image_proj = \
                accelerator.prepare(self.clip_image_enc, self.clip_image_proj)

        if self.trans_net is not None:
            self.trans_net = accelerator.prepare(self.trans_net)

    def unwarp_model(self, accelerator):
        self.unet = accelerator.unwrap_model(self.unet)
        self.face_net_train = accelerator.unwrap_model(self.face_net_train)

        if self.clip_image_enc is not None:
            self.clip_image_enc = accelerator.unwrap_model(self.clip_image_enc)
            self.clip_image_proj = accelerator.unwrap_model(self.clip_image_proj)

        if self.trans_net is not None:
            self.trans_net = accelerator.unwrap_model(self.trans_net)

    def inpaint(self, prompt, infer_image, infer_mask, ref_image,
                extra_feat_neg=False, random_mask=None,
                num_infer_steps=50, guidance_scale=1.0, eta=1.0):
        """Inpaint image according the reference images and prompt.
        Args:
            prompt: str. A natrual language description.
            infer_image: torch.Tensor in [0, 1]. (N, 3, H, W) or (1, N, 3, H, W).
            infer_mask: torch.Tensor, 0 is the region to be inpainted. (N, M, 1, H, W) or (1, N, M, 1, H, W).
            ref_image: torch.Tensor. (N_REF, 3, H, W). Assume the batch is the same identity.
            extra_feat_neg: Whether to use the visual feature in negative prompt.
            random_mask: torch.Tensor or None. (N, 1, 1, H, W) or (1, N, 1, 1, H, W).
        Returns:
            A list of Tensors: [inputs, outputs, cosims]
            mask_ids: torch.Tensor, (NM,).
            inputs: torch.Tensor, (NM, 3, H, W) in [-1, 1].
            outputs: torch.Tensor, (NM, 3, H, W) in [-1, 1].
            cosims: torch.Tensor, (NM,)
        """
        # data preprocessing
        if len(infer_image.shape) == 5:
            infer_image = infer_image[0]
            infer_mask = infer_mask[0]
        infer_mask = infer_mask[:, :, :1]
        if random_mask is not None:
            random_masks = random_mask[0, :, None, :1]
            infer_mask = torch.cat([infer_mask, random_masks], 1)
        infer_image = infer_image.to(self.device) * 2 - 1
        infer_mask = infer_mask.to(self.device)
        ref_image = ref_image.to(self.device) # (1, N_REF, 3, H, W)
        # groundtruth image
        infer_id_feats = self.face_net_eval(infer_image)
        infer_id_feats /= infer_id_feats.norm(p=2, dim=1, keepdim=True)

        inputs, outputs, cosims, mask_ids = [], [], [], []

        for i in range(infer_mask.shape[0]):
            for j in range(infer_mask.shape[1]):
                mask = infer_mask[i:i+1, j]
                neg_prompt = list(PROMPT_TEMPLATES_SMALL.values())[0]
                neg_prompt = self.normal_prompt(neg_prompt)
                visual_feat = self.calc_visual_feature(x=ref_image)[1]
                x_in = infer_image[i:i+1] * (1 - mask)
                x_out_pil = self.inpaint_pipe(prompt, x_in, mask,
                    extra_feats=visual_feat, num_inference_steps=num_infer_steps,
                    guidance_scale=guidance_scale, negative_prompt=neg_prompt,
                    extra_feat_neg=extra_feat_neg, eta=eta)
                x_out = pil2torch(x_out_pil.images[0]).to(x_in.device) * 2 - 1
                inpaint_id_feat = self.face_net_eval(x_out)[0]
                inpaint_id_feat /= inpaint_id_feat.norm()
                inputs.append(x_in[0])
                outputs.append(x_out[0])
                cosims.append(infer_id_feats[i].dot(inpaint_id_feat))
                mask_ids.append(torch.Tensor([j]).to(self.device))
        return [torch.stack(l) for l in [mask_ids, inputs, outputs, cosims]]

    def inpaint_dl(self, dl,
                   html_dir=None, html_num=20, single_file=False,
                   image_dir=None,
                   prompt_temps=PROMPT_TEMPLATES_SMALL,
                   extra_feat_neg=False,
                   use_random_mask=False,
                   accelerator=None,
                   ft_dir=None,
                   max_batches=-1, max_infer_samples=-1, ref_num=-1,
                   num_infer_steps=50, guidance_scale=1):
        """Inpaint a dataloader and store the results.
        Args:
            dl: DataLoader. The dataloader.
            html_path: str. The path to the HTML directory. Set to None to disable storing.
            image_dir: str. The path to the image directory. Set to None to disable storing.
            prompt_temps: dict. The prompt templates.
            accelerator: huggingface accelerator. Set to None indicates single GPU.
            max_batches: int. The maximum number of batches used for visualization.
                         Set to -1 for using the original size of the dataloader.
            max_infer_samples: int. The maximum number of inference images per identity.
                               Set to -1 for using the default.
            ref_num: int. The number of reference images used in inference.
                     Set to -1 for using the default.
        Returns:
            The mean and std of the cosine similarity.
            The HTML visualization file is directly stored at html_path.
        """
        # save the grad_enabled status and disable gradient for now
        orig = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        main_res = None # storing the result for a single identity

        if image_dir is not None:
            files = glob.glob(f"{image_dir}/*")
            if len(files) > 0:
                id_num = [int(f[f.rfind("/")+1:].split("_")[0]) for f in files]
                exist_id = max(id_num)
                dl.dataset.set_start_id(f"{exist_id:04d}")
                print(f"=> Starting from {exist_id}")

        for idx, batch in enumerate(dl):
            id_name = str(batch["id"][0])

            if image_dir is not None:
                if len(glob.glob(f"{image_dir}/{int(id_name)}*")) > 0:
                    print(f"!> {int(id_name)} has existing files, skip.")
                    continue

            if ft_dir is not None:
                # load a finetuned model instead of directly training here
                ft_model_path = glob.glob(os.path.join(ft_dir, id_name, "*.bin"))[0]
                print(f"=> Load finetuned model from {ft_model_path}")
                self.restore(torch.load(ft_model_path, map_location="cpu"))

            B = batch["infer_image"].shape[1]
            if max_infer_samples > 0:
                B = min(B, max_infer_samples)
            ref_image = batch["ref_image"]
            ref_image = ref_image[:, :ref_num] if ref_num > 0 else ref_image
            id_names = torch.Tensor([int(batch["id"][0])])
            image_names = torch.Tensor([int(f[0][:-4]) for f in batch["all_file"][:B]])
            
            for p_name, prompt_temp in prompt_temps.items():
                prompt = self.personalized_prompt(prompt_temp)
                res = self.inpaint(
                        prompt=prompt,
                        infer_image=batch["infer_image"][:, :B],
                        infer_mask=batch["infer_mask"][:, :B],
                        ref_image=ref_image,
                        extra_feat_neg=extra_feat_neg,
                        random_mask=batch["random_mask"][:, :B] if use_random_mask else None,
                        num_infer_steps=num_infer_steps,
                        guidance_scale=guidance_scale)
                n = res[0].shape[0]
                updated_image_names = image_names[:, None].repeat(1, n // B).view(-1)
                res = [id_names.repeat(n).to(self.device), updated_image_names.to(self.device)] + res
                if accelerator is not None:
                    res = accelerator.gather(res)
                
                if accelerator is None or accelerator.is_main_process:
                    if main_res is None or (not single_file and main_res[0][-1] != id_names[-1]):
                        html_num -= 1
                        main_res = [r.clone().detach().cpu() for r in res]
                    else:
                        main_res = [torch.cat([x, r.cpu()]) for x, r in zip(main_res, res)]
                    # we have a new identity now, save as an HTML
                    if not single_file and html_dir and html_num > 0:
                        html_path = f"{html_dir}{int(id_name)}_{p_name}.html"
                        ref_inpaint_visualize_html(main_res, ref_image[0], html_path)
                    if image_dir:
                        res[-2] = res[-2].clamp(-1, 1) / 2 + 0.5
                        for id_name, image_name, mask_idx, _, x_out, _ in zip(*res):
                            image_path = os.path.join(image_dir,
                                f"{int(id_name)}_{int(image_name)}_{int(mask_idx)}_{p_name}.png")
                            vutils.save_image(x_out[None, ...], image_path)

            if max_batches > 0 and idx >= max_batches - 1:
                break
        if accelerator.is_main_process and single_file:
            ref_inpaint_visualize_html(main_res, None, f"{html_dir}.html")
        torch.set_grad_enabled(orig)
        if main_res is not None:
            return res[-1]

    def finetune(self, ref_image, accelerator):
        """Finetune the model using the reference images.
        If already finetuned for one identity,
        please reset before finetuning for another identity.
        Args:
            accelerator: huggingface accelerator.
            ref_image: (N_REF, 3, H, W). The reference images for a single identity.
        """
        orig = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        self.eval() # disable dropout and BN moving averaging

        train_ds = RotatingTensorDataset(ref_image,
                        inpaint_region=["wholeface"])
        train_dl = torch.utils.data.DataLoader(train_ds,
            batch_size=self.cfg["training"]["batch_size"], shuffle=True)
        optimizer = torch.optim.AdamW(self.setup_parameters(), **self.cfg["optimizer"])
        optimizer, train_dl = accelerator.prepare(optimizer, train_dl)

        # pre-compute language features, since they are the same in finetuning
        with torch.no_grad():
            prompts = [self.personalized_prompt("photo of {}")]
            input_ids = self.tokenizer(prompts,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt").input_ids.to(accelerator.device)
            text_feat_orig = self._pre_visual_feature(None, input_ids)[0]
        torch.cuda.empty_cache()

        global_step = 0
        train_steps = self.cfg["training"]["max_train_steps"]
        progress_bar = tqdm(range(train_steps),
            disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        while global_step < self.cfg["training"]["max_train_steps"]:
            for step, batch in enumerate(train_dl):
                mask = batch["infer_mask"][:, 0, 0, :1]
                smask = F.interpolate(mask, scale_factor=1/8)
                iv_img = batch["infer_image"][:, 0] * 2 - 1 # (N, C, H, W)
                # one of the reference is used as inference, so pad one.
                if "ref_image" in batch:
                    rv_imgs = batch["ref_image"]
                    rv_imgs = torch.cat([rv_imgs, rv_imgs[:, -1:].flip([4])], 1)
                # no other reference images in N_REF=1, use the flipped GT as reference.
                else:
                    rv_imgs = batch["infer_image"][:, 0].flip(4)
                N, N_REF = rv_imgs.shape[:2]
                masked_image = iv_img * (1 - mask)
                text_feat = text_feat_orig.repeat(N, 1, 1).requires_grad_(True)
                
                disp = torch.cat([(masked_image + 1) / 2, iv_img])
                vutils.save_image(disp / 2 + 0.5, f"expr/celebahq/test/{step}.png")

                self._training_step(accelerator, optimizer, iv_img,
                        masked_image, input_ids, text_feat, rv_imgs, smask)

                accelerator.wait_for_everyone()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
        torch.set_grad_enabled(orig)

    def _training_step(self, accelerator, optimizer, iv_img, masked_image, text_id, text_feat, rv_imgs, smask):
        N, N_REF = rv_imgs.shape[:2]
        bin_size = self.scheduler.config.num_train_timesteps // self.cfg["training"]["num_time_resample"]

        with torch.no_grad():
            vae_input = torch.cat([iv_img, masked_image])
            vae_output = 0.18215 * self.vae.encode(vae_input).latent_dist.sample()
            gt_latents, mask_latents = vae_output[:N], vae_output[N:]

        for repeat in range(self.cfg["training"]["num_time_resample"]):
            noise = torch.randn_like(mask_latents)
            bin_st, bin_ed = bin_size * repeat, bin_size * (repeat + 1)
            t = torch.randint(bin_st, bin_ed, (N,), device=noise.device)
            noisy_latents = self.scheduler.add_noise(gt_latents, noise, t)
            z_t = torch.cat([noisy_latents, smask, mask_latents], 1)
            z_t = z_t.requires_grad_(True)
            with accelerator.accumulate(self):
                context_feat, _ = self.calc_visual_feature(
                    rv_imgs.requires_grad_(True), text_id, text_feat)
                pred_eps = self.unet(z_t, t, context_feat).sample
                dsm_loss = torch.square(pred_eps - noise).mean()
                accelerator.backward(dsm_loss)
                optimizer.step()
                optimizer.zero_grad()


class TransformNetwork(torch.nn.Sequential):
    """The transformer that aligns face features to generative features.
    Args:
        in_dim: The input feature vector dimension (face features)
        out_dim: The output feature vector dimension (the same as text features)
        num_q_token: The number of query tokens.
        num_layer: The number of transformer blocks.
    """
    def __init__(self, in_dim, out_dim, num_q_token=10, num_layer=4, dropout_p=0.1):
        enc_layer = torch.nn.TransformerEncoderLayer(out_dim, 8,
            dropout=dropout_p, batch_first=True)
        layer_norm = torch.nn.LayerNorm(out_dim)
        trans_enc = torch.nn.TransformerEncoder(enc_layer, num_layer, layer_norm)
        proj_layer = torch.nn.Linear(in_dim, out_dim)
        self.num_q_token = num_q_token
        super().__init__(proj_layer, trans_enc)
        self.num_layer = num_layer
        pos_enc = torch.zeros(num_q_token, out_dim)
        self.pos_enc = torch.nn.Parameter(pos_enc)
        self.q_tokens = torch.nn.Parameter(torch.randn(num_q_token, in_dim))
    
    def forward(self, x):
        q_embed = self.q_tokens.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([x, q_embed], 1)
        x = super().forward(x)[:, -self.num_q_token:]
        return x + self.pos_enc.unsqueeze(0)


class TunedEmbedding(torch.nn.Embedding):
    """Handles the training and inference of special token.
    Args:
        weight: The embedding weight of the original embedding layer.
        stoken_id: The special token id.
        mode: single_token, append, multi_insert. Default: single_token.
    """
    def __init__(self, weight, stoken_id=-1, mode="single_token"):
        super().__init__(weight.shape[0], weight.shape[1])
        self.weight.data.copy_(weight)
        self.mode = mode
        self.stoken_id = stoken_id

    def set_extra_feature(self, feat):
        self.feat = feat
        self.n_dim = len(feat.shape)

    def forward(self, x):
        embed = super().forward(x)
        
        if self.mode == "append":
            pad_token_id = x[0, -1]
            row_idx = (x == pad_token_id).long().argmax(1)
        else:
            stoken_mask = x == self.stoken_id
            # prompt does not contain special token (e.g. in classifier-free)
            if stoken_mask.sum() < 1:
                return embed
            row_idx = [torch.where(m)[0][0].item() for m in stoken_mask]

        if self.mode == "single_token": # add with last token
            for i in range(embed.shape[0]):
                v = self.feat[i] if self.n_dim == 2 else self.feat
                embed[i, row_idx[i]] = embed[i, row_idx[i]] + v

        elif self.mode in ["multi_insert", "append"]:
            N = self.feat.shape[1]
            assert self.n_dim == 3
            embed_orig = embed.clone()
            for i in range(embed.shape[0]):
                st = row_idx[i]
                embed[i, st + 1 + N:] = embed_orig[i, st + 1 : -N]
                embed[i, st + 1: st + 1 + N] = self.feat[i]
        return embed

