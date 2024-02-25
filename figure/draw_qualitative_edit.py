"""Collect the figures for drawing from all servers."""
import torch, os, cv2, glob, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from lib.misc import imread_tensor, imwrite
from lib.face_net import IDSimilarity
from figure.calc_clip_score import PROMPT_TEMPLATES_CONTROL, CLIPScoreMetric
from PIL import Image


# PPT color palette
fontcolor1 = (244, 127, 127)
fontcolor2 = (100, 244, 127)
bg_color = (43, 53, 66)


def preprocess(image):
    x = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    return x.cuda() / 127.5 - 1


def put_text(img, text, pos, fontscale=2.0, color=fontcolor1, thickness=1, center=False):
    if center:
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fontscale, thickness)[0]
        pos = (pos[0] - int(textsize[0] // 2),
                pos[1] + int(textsize[1] // 2))
    cv2.putText(img, text, pos,
        cv2.FONT_HERSHEY_DUPLEX, fontscale, color, thickness, cv2.LINE_AA)


def draw_canvas(args, imsize=512, v_padding=120, h_padding=20, text_height=60):
    N_col = len(args[0][0])
    N_row = len(args)
    canvas_width = imsize * N_col + h_padding * (N_col - 1)
    canvas_height = (v_padding + imsize) * N_row - v_padding
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
    canvas.fill(255)
    cell_width = imsize + h_padding
    cell_height = imsize + v_padding
    text_length = int(text_height * 1.8)
    count = 0

    #for i, title in enumerate(titles):
    #    sty = i * cell_width
    #    put_text(canvas, title, (sty + imsize // 2, 12),
    #             fontscale=0.8, color=bg_color, center=True)

    for row, (images, texts) in enumerate(args):
        for col, (image, text) in enumerate(zip(images, texts)):
            stx = cell_height * row
            edx = stx + imsize
            sty = cell_width * col
            edy = sty + imsize
            disp_image = cv2.resize(image, (imsize, imsize))
            canvas[stx:edx, sty:edy] = disp_image
            if len(text) > 0:
                alpha = 0.1
                d = h_padding // 2
                canvas[stx : stx + 2 * text_height + d, edy - text_length - d: edy] = (canvas[stx : stx + 2 * text_height + d, edy - text_length - d: edy] * alpha + (1 - alpha) * np.array([bg_color])).astype("uint8")
                put_text(canvas, text[0], (edy - text_length - d // 2, stx + text_height), thickness=2)
                put_text(canvas, text[1], (edy - text_length - d // 2, stx + 2 * text_height), color=fontcolor2, thickness=2)
            count += 1

    return canvas


def save_pdf(fp, canvas, scale=1/20):
    sizes = (canvas.shape[1] * scale, canvas.shape[0] * scale)
    plt.figure(figsize=sizes) # paper: 11, 7
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fp)
    plt.close()


target_dir = "data/celebahq/result/qualitative_editing"
data_dir = "data/celebahq"
selected_ids = [
    ("0057", "3562", "0", "smile"),
    ("0014", "24431", "0", "beard"),
    #("0018", "13013", "0", "sunglasses"), not good
    #("0038", "25720", "0", "sunglasses"), fair
    ("0065", "310", "0", "eyeglass"),
    ("0021", "26735", "0", "lipstick")
    #("0065", "2248",  "3", "angry"),
    #("0065", "3889",  "2", "angry")
    ]
methods_dics = {
    "Input": "data/celebahq/IDI/idi_5",
    "Textual Inversion": "SDI_TI_1_control",
    "Custom Diffusion": "SDI_CD_1_control",
    "PVA (Ours)": "SDI2_FP_FT20_control_15.0",
    "Groundtruth": "image"
}

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    os.makedirs(target_dir, exist_ok=True)

    id_crit = IDSimilarity(model_type="glint").cuda()
    #clip = get_clip()
    #short_lf_dic, long_lf_dic = get_text_feats(clip)
    #clip_image_func = get_vision_func(clip)
    clip_score_metric = CLIPScoreMetric()

    args = []
    for id_name, image_idx, mask_id, p_name in selected_ids:
        images = []
        for name, dir in methods_dics.items():
            if "Input" == name:
                fpath = glob.glob(f"{data_dir}/{dir}/*_{image_idx}_2*")[0]
            elif "Groundtruth" == name:
                fpath = f"{data_dir}/{dir}/{image_idx}.jpg"
                gt_image = imread_tensor(fpath)
                id_crit.set_reference(gt_image.unsqueeze(0).cuda() * 2 - 1)
            else:
                fpath = glob.glob(f"{data_dir}/{dir}/*_{image_idx}_{mask_id}_{p_name}*")[0]
            images.append(cv2.imread(fpath)[..., ::-1])
        # calculate ID similarity and CLIP score
        texts = [""]
        for image in images[1:-1]:
            x = preprocess(image)
            cosim = max(0, (1 - id_crit(x)).item())
            
            prompt = PROMPT_TEMPLATES_CONTROL[p_name].format("a person")
            clip_score = clip_score_metric(x / 2 + 0.5, [prompt]).cpu().item()
            texts.append([f"{cosim:.2f}"[1:], f"{clip_score:.2f}"[1:]])
        texts.append("")
        args.append((images, texts))
    canvas = draw_canvas(args)#, list(methods_dics.keys()))
    imwrite(f"{target_dir}/{id_name}.png", canvas)
    save_pdf(f"{target_dir}/{id_name}.pdf", canvas)