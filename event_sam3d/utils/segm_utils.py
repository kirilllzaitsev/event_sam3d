from pathlib import Path

import numpy as np
import torch
from PIL import Image

from event_sam3d.config import SAM3_DIR
from event_sam3d.utils.common_utils import adjust_img_for_plt
from event_sam3d.utils.vis_utils import plot_rgb


def get_sam3_model():
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # turn on tfloat32 for Ampere GPUs
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    torch.inference_mode().__enter__()

    bpe_path = f"{SAM3_DIR}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return {
        "model": model,
        "processor": processor,
    }


def get_sam3_preds(image, processor, prompt="person"):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(adjust_img_for_plt(image))
    # plot_rgb(np.asarray(image))
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    return inference_state


def save_sam3_pred(save_path, inference_state):
    if len(inference_state["scores"]) > 0:
        torch.save(
            {
                k: v
                for k, v in inference_state.items()
                if k
                in ["masks", "boxes", "scores", "original_height", "original_width"]
            },
            save_path,
        )
        return True
    return False
