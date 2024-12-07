import torch
import random
from torchvision import transforms

class Augmentation(object):
    def __init__(self, rotate=True, flip=True, brightness=0.4, contrast=0.4, sharpness=0.4, saturation=0.4):
        self.rotate = rotate
        self.flip = flip
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
        self.saturation = saturation

    def __call__(self, img):
        functions = [self._rotate, self._flip,
                    self._sharpness, self._saturation, self._contrast, self._brightness]
        random.shuffle(functions)
        
        for func in functions:
            img = func(img)

        return img

    def _flip(self, img):
        choice = random.randint(0,3)
        if choice < 2:
            return img
        elif choice == 2:
            return transforms.functional.vflip(img)
        else:
            return transforms.functional.hflip(img)

    def _rotate(self, img):
        choice = random.randint(0,5)
        if choice < 3:
            return img
        elif choice == 3:
            return transforms.functional.rotate(img, 90)
        elif choice == 4:
            return transforms.functional.rotate(img, 180)
        else:
            return transforms.functional.rotate(img, 270)

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
        