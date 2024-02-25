# This code is built from the Custom Diffuions repository.
# Copyright 2022- The Hugging Face team. All rights reserved.
#                               Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2022 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================
#                               Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS

#    APPENDIX: How to apply the Apache License to your work.

#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.

#    Copyright [yyyy] [name of copyright owner]

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse, PIL, glob, json, hashlib, itertools, math, os, sys, random
import torch
import xformers
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
sys.path.insert(0, ".")
from lib.diffusers import diffusers
from diffusers import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.models.cross_attention import CrossAttention


logger = get_logger(__name__)


def create_custom_diffusion(unet, freeze_model):
    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                params.requires_grad = True
                print(name)
            else:
                params.requires_grad = False
        else:
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                params.requires_grad = True
                print(name)
            else:
                params.requires_grad = False

    def new_forward(self, hidden_states, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        crossattn = False
        context = None
        if 'context' in kwargs:
            context = kwargs['context']
        elif 'encoder_hidden_states' in kwargs:
            context = kwargs['encoder_hidden_states']
        if context is not None:
            crossattn = True

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        if crossattn:
            modifier = torch.ones_like(key)
            # print(key.shape)
            modifier[:, :1, :] = modifier[:, :1, :]*0.
            key = modifier*key + (1-modifier)*key.detach()
            value = modifier*value + (1-modifier)*value.detach()

        dim = query.shape[-1]

        #query = self.reshape_heads_to_batch_dim(query)
        #key = self.reshape_heads_to_batch_dim(key)
        #value = self.reshape_heads_to_batch_dim(value)

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = self.get_attention_scores(query, key)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def change_forward(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer)

    change_forward(unet)
    return unet


def save_progress(text_encoder, unet, modifier_token_id, accelerator, args, save_path):
    logger.info("Saving embeddings")
    delta_dict = {'unet': {}, 'modifier_vec': {}}
    if args.modifier_token is not None:
        for i in range(len(modifier_token_id)):
            learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[modifier_token_id[i]]
            delta_dict['modifier_vec'][args.modifier_token[i]] = learned_embeds.detach().cpu()
    elif args.train_text_encoder:
        delta_dict['text_encoder'] = accelerator.unwrap_model(text_encoder).state_dict()
    for name, params in accelerator.unwrap_model(unet).named_parameters():
        if args.freeze_model == 'crossattn':
            if 'attn2' in name:
                delta_dict['unet'][name] = params.cpu().clone()
        else:
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                delta_dict['unet'][name] = params.cpu().clone()

    torch.save(delta_dict, save_path)


def load_model(text_encoder, tokenizer, unet, save_path, compress=False, freeze_model='crossattn_kv'):
    st = torch.load(save_path)
    if 'text_encoder' in st:
        text_encoder.load_state_dict(st['text_encoder'])
    if 'modifier_token' in st:
        modifier_tokens = list(st['modifier_token'].keys())
        print(modifier_tokens)
        modifier_token_id = []
        for modifier_token in modifier_tokens:
            _ = tokenizer.add_tokens(modifier_token)
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for i, id_ in enumerate(modifier_token_id):
            token_embeds[id_] = st['modifier_token'][modifier_tokens[i]]

    print(st.keys())
    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                if compress and ('to_k' in name or 'to_v' in name):
                    params.data += st['unet'][name]['u']@st['unet'][name]['v']
                else:
                    params.data.copy_(st['unet'][f'{name}'])
        else:
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                if compress:
                    params.data += st['unet'][name]['u']@st['unet'][name]['v']
                else:
                    params.data.copy_(st['unet'][f'{name}'])


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="pretrained",
        help="Path to pretrained model directory.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default='crossattn_kv',
        help="crossattn to enable fine-tuning of all key, value, query matrices",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default='ktn+pll+ucd', help="A token to use as initializer word."
    )
    parser.add_argument("--flip_p", type=float, default=0, help="Apply horizontal flip data augmentation.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            logger.warning("You need not use --class_prompt without --with_prior_preservation.")

    return args


class RandomMaskDataset(torch.utils.data.Dataset):
    """Load a random mask from a folder."""

    def __init__(self, data_dir="data/celebahq/mask", size=(256, 256)):
        self.data_dir = data_dir
        self.size = size
        # mask should be stored in PNG format
        self.mask_paths = glob.glob(f"{data_dir}/*.png")
        self.mask_paths.sort()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mask_paths)
    
    def sample(self):
        return self[np.random.randint(0, len(self))]

    def sample_pil(self):
        i = np.random.randint(0, len(self))
        return Image.open(self.mask_paths[i])

    def __getitem__(self, i):
        img = Image.open(self.mask_paths[i])
        if img.size[0] != self.size[0]:
            img = img.resize(self.size, resample=Image.Resampling.NEAREST)
        return self.to_tensor(img)


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        flip_p=0,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = PIL.Image.BILINEAR

        self.ds = RandomMaskDataset(size=(size, size))
        # hardcoded temporarily
        self.inpaint_region = ["lowerface",  "eyebrow", "wholeface"]
        self.bboxes = torch.load(f"data/celebahq/annotation/region_bbox.pth")["bboxes"]

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()]
            inst_names = [str(s[0]) for s in inst_img_path]
            inst_names = [s[s.rfind("/") + 1 : -4] for s in inst_names]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x,y) in zip(class_images_path, class_prompt)]
                exclude_inst = lambda x: x[0][x[0].rfind("/") + 1 : -4] not in inst_names
                class_img_path = list(filter(exclude_inst, class_img_path))
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip_p = flip_p

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def get_mask(self, image_path, orig_size):
        random_mask = self.ds.sample()[None, ..., 0]
        rname = np.random.choice(self.inpaint_region, (1,))[0]
        image_path = str(image_path)
        gidx = int(image_path[image_path.rfind("/") + 1 : -4])
        iv_mask = torch.zeros(1, self.size, self.size)
        scale = float(self.size) / orig_size 
        x_min, y_min, x_max, y_max = (self.bboxes[rname][gidx] * scale).long()
        iv_mask[..., x_min:x_max, y_min:y_max].fill_(1)
        iv_mask = (iv_mask + random_mask).clamp(min=0, max=1)
        return iv_mask

    def __getitem__(self, index):
        example = {}

        image_path, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        orig_size = instance_image.size[0]
        iv_mask = self.get_mask(image_path, orig_size)

        #### apply augmentation and create a valid image regions mask ####
        if np.random.randint(0, 3) < 2:
            random_scale = np.random.randint(self.size // 3, self.size+1)
        else:
            random_scale = np.random.randint(int(1.2*self.size), int(1.4*self.size))

        if random_scale % 2 == 1:
            random_scale += 1

        random_scale = self.size # don't use augmentation
        
        if random_scale < 0.6*self.size:
            add_to_caption = np.random.choice(["A far away ", "A very small "])
            instance_prompt = add_to_caption + instance_prompt
            cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
            cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

            instance_image1 = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image1 = np.array(instance_image1).astype(np.uint8)
            instance_image1 = (instance_image1 / 127.5 - 1.0).astype(np.float32)

            instance_image =  np.zeros((self.size, self.size,3), dtype=np.float32)
            instance_image[cx - random_scale // 2: cx + random_scale // 2, cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1

            iv_mask1 = iv_mask.resize((random_scale, random_scale), resample=self.interpolation)
            iv_mask1 = np.array(iv_mask1).astype(np.uint8)
            iv_mask1 = (iv_mask1 > 127.5).astype(np.float32)

            iv_mask = np.zeros((self.size, self.size,3), dtype=np.float32)
            iv_mask[cx - random_scale // 2: cx + random_scale // 2, cy - random_scale // 2: cy + random_scale // 2, :] = iv_mask1

            mask = np.zeros((self.size // 8, self.size // 8))
            mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1, (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.
        elif random_scale > self.size:
            add_to_caption = np.random.choice(["A zoomed in ", "A close up "])
            instance_prompt = add_to_caption + instance_prompt
            cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
            cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

            instance_image = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            instance_image = instance_image[cx - self.size // 2: cx + self.size // 2, cy - self.size // 2: cy + self.size // 2, :]

            iv_mask = iv_mask.resize((random_scale, random_scale), resample=self.interpolation)
            iv_mask = np.array(iv_mask).astype(np.uint8)
            iv_mask = (iv_mask > 127.5).astype(np.float32)
            iv_mask = iv_mask[cx - self.size // 2: cx + self.size // 2, cy - self.size // 2: cy + self.size // 2, :]

            mask = np.ones((self.size // 8, self.size // 8))
        else:
            if self.size is not None:
                instance_image = instance_image.resize((self.size, self.size), resample=self.interpolation)
                iv_mask = F.interpolate(iv_mask.unsqueeze(0), (self.size, self.size), mode="nearest")[0]
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))
        ########################################################################

        instance_image = torch.from_numpy(instance_image).permute(2, 0, 1)
        if np.random.rand() < self.flip_p:
            instance_image = torch.flip(instance_image, (2,))
            iv_mask = torch.flip(iv_mask, (2,))
        example["instance_images"] = instance_image
        example["iv_mask"] = iv_mask
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids

        if self.with_prior_preservation:
            image_path, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(image_path).convert("RGB")
            class_image = self.image_transforms(class_image)
            class_mask = self.get_mask(image_path, orig_size)
            if np.random.rand() < self.flip_p:
                class_image = torch.flip(class_image, (2,))
                class_mask = torch.flip(class_mask, (2,))
            example["class_images"] = class_image
            example["class_mask"] = torch.ones_like(class_image)
            example["class_iv_mask"] = class_mask
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


#def main(args):
if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)


    if args.with_prior_preservation:
        for i, concept in enumerate(args.concepts_list):
            class_images_dir = Path(concept['class_data_dir'])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            if args.real_prior:
                """ # This downloads real images from Internet
                if accelerator.is_main_process:
                    name = '_'.join(concept['class_prompt'].split())
                    if not Path(os.path.join(class_images_dir, name)).exists() or len(list(Path(os.path.join(class_images_dir, name)).iterdir())) < args.num_class_images:
                        retrieve.retrieve(concept['class_prompt'], class_images_dir, args.num_class_images)
                accelerator.wait_for_everyone()
                """
                #images = glob.glob(f"{class_images_dir}/*")
                #with open(f"{args.output_dir}/caption.txt", "w") as f:
                #    f.writelines(["A face photo of a person.\n" for _ in images])
                #with open(f"{args.output_dir}/images.txt", "w") as f:
                #    f.writelines([f"{f}\n" for f in images])
                #concept['class_prompt'] = os.path.join(args.output_dir, f'caption.txt')
                #concept['class_data_dir'] = os.path.join(args.output_dir, f'images.txt')
                #args.concepts_list[i] = concept
            else:
                cur_class_images = len(list(class_images_dir.iterdir()))

                if cur_class_images < args.num_class_images:
                    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    pipeline.set_progress_bar_config(disable=True)

                    num_new_images = args.num_class_images - cur_class_images
                    logger.info(f"Number of class images to sample: {num_new_images}.")

                    sample_dataset = PromptDataset(args.class_prompt, num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                    sample_dataloader = accelerator.prepare(sample_dataloader)
                    pipeline.to(accelerator.device)

                    for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        images = pipeline(example["prompt"], num_inference_steps=50, guidance_scale=6., eta=1.).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)

    # Load models and create wrapper for stable diffusion
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate*2.

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    unet = create_custom_diffusion(unet, args.freeze_model)
    
    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split('+')
        args.initializer_token = args.initializer_token.split('+')
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(args.modifier_token, args.initializer_token[:len(args.modifier_token)]):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            print(token_ids)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for (x,y) in zip(modifier_token_id,initializer_token_id):
            token_embeds[x] = token_embeds[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)

        params_to_optimize = itertools.chain( text_encoder.get_input_embeddings().parameters() , [x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])] )

    ########################################################
    ########################################################
    else:
        params_to_optimize = (
            itertools.chain([x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])], text_encoder.parameters() if args.train_text_encoder else [] ) 
        )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        flip_p=args.flip_p
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        mask = [example["mask"] for example in examples]
        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            mask += [example["class_mask"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        mask = torch.stack(mask)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        mask = mask.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mask": mask.unsqueeze(1)
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1,
        #collate_fn=collate_fn,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder or args.modifier_token is not None:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder or args.modifier_token is None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    B = args.train_batch_size
    T = noise_scheduler.config.num_train_timesteps

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder or args.modifier_token is not None:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                # Convert images to latent space
                inst_image, prior_image = batch["instance_images"], batch["class_images"]
                inst_mask, prior_mask = batch["iv_mask"], batch["class_iv_mask"]
                masked_inst_image = inst_image * (1 - inst_mask)
                masked_class_mask = prior_image * (1 - prior_mask)
                vae_in = torch.cat([inst_image, prior_image,
                    masked_inst_image, masked_class_mask], 0).to(weight_dtype)
                vae_out = 0.18215 * vae.encode(vae_in).latent_dist.sample()
                gt_latent = vae_out[: 2 * B] # inst, prior
                masked_latent = vae_out[2 * B :] # inst, prior
                noise = torch.randn_like(gt_latent)
                timesteps = torch.randint(0, T, (2 * B,), device=vae_in.device).long()
                noisy_latent = noise_scheduler.add_noise(
                    gt_latent, noise, timesteps)
                mask = torch.cat([inst_mask, prior_mask], 0)
                smask = F.interpolate(mask, gt_latent.shape[2:])
                x_in = torch.cat([noisy_latent, smask, masked_latent], 1)
                comb_ids = torch.cat([batch["instance_prompt_ids"][0],
                                      batch["class_prompt_ids"][0]], 0)

            with accelerator.accumulate(unet):
                # Get the text embedding for conditioning
                text_feat = text_encoder(comb_ids)[0]

                # Predict the noise residual
                model_pred = unet(x_in, timesteps, text_feat).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        noisy_latent, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    model_pred_inst, model_pred_prior = model_pred[:B], model_pred[B:]
                    target_inst, target_prior = target[:B], target[B:]

                    # Compute instance loss
                    target_dsm_loss = torch.square(model_pred_inst.float()
                        - target_inst.float()).mean([1, 2, 3])

                    # Compute prior loss
                    prior_dsm_loss = torch.square(model_pred_prior.float()
                        - target_prior.float()).mean([1, 2, 3])

                    # Add the prior loss to the instance loss.
                    loss = target_dsm_loss.mean() + \
                        args.prior_loss_weight * prior_dsm_loss.mean()
                else:
                    mask = batch["mask"]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])).mean()

                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.modifier_token is not None:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                    for i in range(len(modifier_token_id[1:])):
                        index_grads_to_zero = index_grads_to_zero | torch.arange(len(tokenizer)) != modifier_token_id[i]
                    grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain([x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])], text_encoder.parameters())
                        if (args.train_text_encoder or args.modifier_token is not None)
                        else itertools.chain([x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])]) 
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        pipeline = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            revision=args.revision,
                        )
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        pipeline.save_pretrained(save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        #pipeline = DiffusionPipeline.from_pretrained(
        #    args.pretrained_model_name_or_path,
        #    unet=accelerator.unwrap_model(unet),
        #    text_encoder=accelerator.unwrap_model(text_encoder),
        #    tokenizer=tokenizer,
        #    revision=args.revision,
        #)
        #pipeline.save_pretrained(args.output_dir)
        save_path = os.path.join(args.output_dir, "delta.bin")
        save_progress(text_encoder, unet, modifier_token_id, accelerator, args, save_path)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()

#if __name__ == "__main__":
#    args = parse_args()
#    main(args)

