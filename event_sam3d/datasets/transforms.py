import cv2
import numpy as np

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
        self, names, probs=None, crop_size=224, blur_ksize_min=5, blur_ksize_max=15
    ):
        self.names = names
        self.crop_size = crop_size
        self.blur_ksize_min = blur_ksize_min
        self.blur_ksize_max = blur_ksize_max
        self.probs = {n: 0.5 for n in names} if probs is None else probs

    def __call__(self, sample):
        if "hflip" in self.names and np.random.rand() < self.probs["hflip"]:
            for k in ["rgb", "mask", "events"]:
                if k in sample:
                    sample[k] = horizontal_flip(sample[k])
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
                for k in ["rgb", "events"]:
                    if k in sample:
                        sample[k] = sample[k][y : y + n, x : x + n, :]
                        sample[k] = cv2.resize(
                            sample[k],
                            (w, h),
                            interpolation=cv2.INTER_LINEAR,
                        )
        sample["rgb_clean"] = sample["rgb"].copy()
        if "gblur" in self.names and np.random.rand() < self.probs["gblur"]:
            ksize = np.random.randint(self.blur_ksize_min, self.blur_ksize_max + 1)
            sample["rgb"] = gauss_blur(sample["rgb"], ksize=ksize)
        if "mblur" in self.names and np.random.rand() < self.probs["mblur"]:
            ksize = np.random.randint(self.blur_ksize_min, self.blur_ksize_max + 1)
            sample["rgb"] = motion_blur(sample["rgb"], ksize=ksize)
        if "wavelet" in self.names:
            rgb_feat = adjust_img_for_torch(sample["rgb"])
            rgb_high_freq, rgb_low_freq = wavelet_decomposition(rgb_feat)
            event_feat = adjust_img_for_torch(sample["events"])
            event_high_freq, event_low_freq = wavelet_decomposition(event_feat)
            sample["events"] = adjust_img_for_plt(event_high_freq)
            sample["rgb"] = adjust_img_for_plt(rgb_low_freq)
        return sample
