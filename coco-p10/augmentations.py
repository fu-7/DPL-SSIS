from cvpods.data.transforms import ComposeTransform, ResizeShortestEdge, RandomFlip, NoOpTransform
import torchvision.transforms as transforms
from PIL import ImageFilter
import numpy as np


class GaussianBlur:
    def __init__(self, rad_range=[0.1, 2.0]):
        self.rad_range = rad_range

    def __call__(self, x):
        rad = np.random.uniform(*self.rad_range)
        x = x.filter(ImageFilter.GaussianBlur(radius=rad))
        return x

class RandomApply:
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img, annotation):
        if self.p < np.random.random():
            return img, annotation
        for t in self.transforms:
            img = t(img)
        return img, annotation

class ToPILImage:
    def __init__(self):
        self.transform = transforms.ToPILImage()

    def __call__(self, img, annotation=None):
        return self.transform(img), annotation

class T
