import cv2
import numpy as np

from event_sam3d.nb_utils_static import get_crop_from_mask
from event_sam3d.utils.common_utils import adjust_img_for_plt, adjust_img_for_torch
from event_sam3d.utils.wavelet_utils import wavelet_decomposition


def random_crop(img: np.ndarray, n: int) -> np.ndarray:
    h, w, _ = img.shape
    if n > h or n > w:
        raise ValueError("crop size larger than image")

    y = np.random.randint(0, h - n + 1)
    x = np.random.randint(0, w - n + 1)

    return img[y : y + n, x : x + n, :]


def horizontal_flip(img):
    if img.ndim == 3:
        flipped = img[:, ::-1, :]
    else:
        flipped = img[:, ::-1]
    return flipped.copy()


def gauss_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def motion_blur(img, ksize=5):
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1) / 2), :] = (
        np.ones(
            ksize,
        )
        / ksize
    )
    return cv2.filter2D(img, -1, kernel)


class Transform:

    def __init__(
        self,
        names,
        probs=None,
        crop_size=224,
        blur_ksize_min=5,
        blur_ksize_max=15,
        wv_levels_min=0,
        wv_levels_max=3,
        resize_hw=None,
    ):
        self.names = names
        self.crop_size = crop_size
        self.blur_ksize_min = blur_ksize_min
        self.blur_ksize_max = blur_ksize_max
        self.wv_levels_min = wv_levels_min
        self.wv_levels_max = wv_levels_max
        self.resize_hw = resize_hw
        self.probs = {n: 0.5 for n in names} if probs is None else probs

        if "resize" in names:
            assert resize_hw is not None

    def __call__(self, sample):
        target_keys = ["rgb", "mask", "events", "rgb_clean"]
        if "hflip" in self.names and np.random.rand() < self.probs["hflip"]:
            for k in target_keys:
                if k in sample:
                    sample[k] = horizontal_flip(sample[k])
        if "obj_crop" in self.names:
            mask = sample["mask"]
            hw = (sample["rgb"].shape[0], sample["rgb"].shape[1])
            for k in target_keys:
                if k in sample and k != "mask":
                    mask2 = mask.copy()
                    if k == "events":
                        sample[k] = cv2.resize(sample[k], (hw[1], hw[0]))
                        # mask2 = cv2.resize(mask2, (self.resize_hw[1], self.resize_hw[0]))
                    mask2 = cv2.dilate(
                        mask2.astype(np.uint8),
                        np.ones((11, 11), np.uint8),
                        iterations=1,
                    ).astype(bool)
                    sample[k] = get_crop_from_mask(sample[k], mask2)
        if "random_crop" in self.names and np.random.rand() < self.probs["random_crop"]:
            h, w, _ = sample["rgb"].shape
            n = self.crop_size
            if n > h or n > w:
                raise ValueError("crop size larger than image")
            y = np.random.randint(0, h - n + 1)
            x = np.random.randint(0, w - n + 1)
            # ensure that sufficient number of obj pxs remains after crop
            use_crop = True
            if "mask" in sample:
                mask_crop = sample["mask"][y : y + n, x : x + n]
                if np.sum(mask_crop) / np.sum(sample["mask"]) < 0.4:
                    use_crop = False
            if use_crop:
                mask_crop = cv2.resize(
                    mask_crop.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                sample["mask"] = mask_crop
                for k in target_keys:
                    if k in sample and k != "mask":
                        sample[k] = sample[k][y : y + n, x : x + n, :]
                        sample[k] = cv2.resize(
                            sample[k],
                            (w, h),
                            interpolation=cv2.INTER_LINEAR,
                        )
        if "resize" in self.names:
            for k in target_keys:
                if k in sample:
                    sample[k] = cv2.resize(
                        sample[k],
                        (self.resize_hw[1], self.resize_hw[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
        if "rgb_clean" not in sample:
            sample["rgb_clean"] = sample["rgb"].copy()
        if "gblur" in self.names and np.random.rand() < self.probs["gblur"]:
            ksize = np.random.randint(self.blur_ksize_min, self.blur_ksize_max + 1)
            sample["rgb"] = gauss_blur(sample["rgb"], ksize=ksize)
        if "mblur" in self.names and np.random.rand() < self.probs["mblur"]:
            ksize = np.random.randint(self.blur_ksize_min, self.blur_ksize_max + 1)
            sample["rgb"] = motion_blur(sample["rgb"], ksize=ksize)
        if "wavelet" in self.names:
            rgb_feat = adjust_img_for_torch(sample["rgb"])
            wv_levels = np.random.randint(self.wv_levels_min, self.wv_levels_max + 1)
            if wv_levels > 0:
                rgb_high_freq, rgb_low_freq = wavelet_decomposition(
                    rgb_feat, levels=wv_levels
                )
                event_feat = adjust_img_for_torch(sample["events"])
                event_high_freq, event_low_freq = wavelet_decomposition(
                    event_feat, levels=wv_levels
                )
                sample["events"] = adjust_img_for_plt(event_high_freq)
                sample["rgb"] = adjust_img_for_plt(rgb_low_freq)
        return sample
