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
        sample["rgb_clean"] = sample["rgb"].copy()
        if "gblur" in self.names and np.random.rand() < self.probs["gblur"]:
            ksize = np.random.randint(self.blur_ksize_min, self.blur_ksize_max + 1)
            sample["rgb"] = gauss_blur(sample["rgb"], ksize=ksize)
        if "mblur" in self.names and np.random.rand() < self.probs["mblur"]:
            ksize = np.random.randint(self.blur_ksize_min, self.blur_ksize_max + 1)
            sample["rgb"] = motion_blur(sample["rgb"], ksize=ksize)
        return sample
