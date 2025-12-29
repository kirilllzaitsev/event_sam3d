import numpy as np


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
    return flipped


class Transform:

    def __init__(self, names, probs=None, crop_size=224):
        self.names = names
        self.crop_size = crop_size
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
            if "mask" in sample:
                mask_crop = sample["mask"][y : y + n, x : x + n]
                if np.sum(mask_crop) / np.sum(sample["mask"]) < 0.4:
                    return sample
                sample["mask"] = mask_crop
            for k in ["rgb", "events"]:
                if k in sample:
                    sample[k] = sample[k][y : y + n, x : x + n, :]
        return sample
