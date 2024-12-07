## Additional transforms for PyTorch data augmentation
## It is very recommended to use Pillow-SIMD for speed gain in the 5x range.
## https://python-pillow.org/pillow-perf/
## OpenCV built with IPP and TBB is also fast but inaccurate

"""
Taken from https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/src/p_data_augmentation.py
"""

import torch
import random
import PIL.ImageEnhance as ie
import PIL.Image as im


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomFlip(object):
    """Randomly flips the given PIL.Image with a probability of 0.25 horizontal,
                                                                0.25 vertical,
                                                                0.5 as is
    """
    
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img.transpose(im.FLIP_LEFT_RIGHT),
            3: img.transpose(im.FLIP_TOP_BOTTOM)
        }
    
        return dispatcher[random.randint(0,3)] #randint is inclusive

class RandomRotate(object):
    """Randomly rotate the given PIL.Image with a probability of 1/6 90°,
                                                                 1/6 180°,
                                                                 1/6 270°,
                                                                 1/2 as is
    """
    
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img,            
            3: img.transpose(im.ROTATE_90),
            4: img.transpose(im.ROTATE_180),
            5: img.transpose(im.ROTATE_270)
        }
    
        return dispatcher[random.randint(0,5)] #randint is inclusive
    
class PILColorBalance(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Color(img).enhance(alpha)

class PILContrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Contrast(img).enhance(alpha)


class PILBrightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Brightness(img).enhance(alpha)

class PILSharpness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Sharpness(img).enhance(alpha)
    

# Check ImageEnhancer effect: https://www.youtube.com/watch?v=_7iDTpTop04
# Not documented but all enhancements can go beyond 1.0 to 2
# Image must be RGB
# Use Pillow-SIMD because Pillow is too slow
class PowerPIL(RandomOrder):
    def __init__(self, rotate=True,
                       flip=True,
                       colorbalance=0.4,
                       contrast=0.4,
                       brightness=0.4,
                       sharpness=0.4):
        self.transforms = []
        if rotate:
            self.transforms.append(RandomRotate())
        if flip:
            self.transforms.append(RandomFlip())
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if colorbalance != 0:
            self.transforms.append(PILColorBalance(colorbalance))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))

import random
from torchvision import transforms

class Enhanced(object):
    def __init__(self, rotate=True, flip=True, brightness=0.4, contrast=0.4, sharpness=0.4, saturation=0.4):
        self.rotate = rotate
        self.flip = flip
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
        self.saturation = saturation

    def __call__(self, img):
        functions = [self._rotate, self._flip, # self._vertical_flip, self._horizontal_flip,
                    self._sharpness, self._saturation, self._contrast, self._brightness]
        random.shuffle(functions)
        
        for func in functions:
            img = func(img)

        return img

    def _flip(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img.transpose(im.FLIP_LEFT_RIGHT),
            3: img.transpose(im.FLIP_TOP_BOTTOM)
        }
    
        return dispatcher[random.randint(0,3)] #randint is inclusive

    def _rotate(self, img):
        # return transforms.RandomRotation(270)(img)
        dispatcher = {
            0: img,
            1: img,
            2: img,            
            3: img.transpose(im.ROTATE_90),
            4: img.transpose(im.ROTATE_180),
            5: img.transpose(im.ROTATE_270)
        }
    
        return dispatcher[random.randint(0,5)] #randint is inclusive

    def _vertical_flip(self, img):
        return transforms.RandomVerticalFlip(p=0.5)(img)

    def _horizontal_flip(self, img):
        return transforms.RandomHorizontalFlip(p=0.5)(img)

    def _sharpness(self, img):
        value = random.uniform(1-self.sharpness, 1+self.sharpness)
        return transforms.functional.adjust_sharpness(img, value)

    def _saturation(self, img):
        value = random.uniform(1-self.saturation, 1+self.saturation)
        return transforms.functional.adjust_saturation(img, value)
    
    def _contrast(self, img):
        value = random.uniform(1-self.contrast, 1+self.contrast)
        return transforms.functional.adjust_contrast(img, value)

    def _brightness(self, img):
        value = random.uniform(1-self.brightness, 1+self.brightness)
        return transforms.functional.adjust_brightness(img, value)
        