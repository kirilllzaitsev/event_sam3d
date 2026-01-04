from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

try:
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded
    from sam3.train.data.sam3_image_dataset import Image as SAMImage
    from sam3.train.data.sam3_image_dataset import InferenceMetadata
except ImportError:
    print("sam3 package not available")

from event_sam3d.config import SAM3_DIR
from event_sam3d.utils.common_utils import adjust_img_for_plt
from event_sam3d.utils.vis_utils import plot_rgb

GLOBAL_COUNTER = 1


def create_empty_datapoint():
    """A datapoint is a single image on which we can apply several queries at once."""
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint, pil_image):
    """Add the image to be processed to the datapoint"""
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint, text_query):
    """Add a text query to the datapoint"""

    global GLOBAL_COUNTER
    # in this function, we require that the image is already set.
    # that's because we'll get its size to figure out what dimension to resize masks and boxes
    # In practice you're free to set any size you want, just edit the rest of the function
    assert len(datapoint.images) == 1, "please set the image first"

    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],  # unused for inference
            is_exhaustive=True,  # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


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


def get_sam3_model_batch():
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )

    postprocessor = PostProcessImage(
        max_dets_per_img=-1,  # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
        iou_type="segm",  # we want masks
        use_original_sizes_box=True,  # our boxes should be resized to the image size
        use_original_sizes_mask=True,  # our masks should be resized to the image size
        convert_mask_to_rle=False,  # the postprocessor supports efficient conversion to RLE format. In this demo we prefer the binary format for easy plotting
        detection_threshold=0.5,  # Only return confident detections
        to_cpu=False,
    )
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008, max_size=1008, square=True, consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    model = get_sam3_model()["model"]

    return {
        "model": model,
        "postprocessor": postprocessor,
        "transform": transform,
    }


def get_sam3_preds(image, processor, prompt="person"):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(adjust_img_for_plt(image))
    # plot_rgb(np.asarray(image))
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    return inference_state


def get_sam3_preds_batch(image, prompts, model, transform, postprocessor):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(adjust_img_for_plt(image))

    datapoint1 = create_empty_datapoint()
    set_image(datapoint1, image)
    pids = {}
    for p in prompts:
        id1 = add_text_prompt(datapoint1, p)
        pids[id1] = p
    datapoint1 = transform(datapoint1)
    batch = collate([datapoint1], dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
    with torch.no_grad():
        output = model(batch)
    processed_results = postprocessor.process_results(output, batch.find_metadatas)
    output_f = {pids[k]: v for k, v in processed_results.items()}
    return output_f


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
