import numpy as np
from PIL import Image
from lib.path_config import ImgHeight, CharWidth


class RandomClip:
    def __init__(self, min_clip_width=ImgHeight * 2):
        self.min_clip_width = min_clip_width

    def _recalc_len(self, leng, scale=CharWidth):
        tmp = leng % scale
        return leng - tmp if tmp != 0 else leng

    def __call__(self, pic:Image):
        width, height = pic.size[0], pic.size[1]
        if width > self.min_clip_width:
            crop_width = np.random.randint(self.min_clip_width, width)
            # crop_width = self._recalc_len(crop_width, scale=CharWidth)
            rand_pos = np.random.randint(0, width - crop_width)
            pic = pic.crop((rand_pos, 0, rand_pos + crop_width, height))
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomScale:
    def __init__(self, var=0.4):
        self.var = var

    def _recalc_len(self, leng, scale=CharWidth):
        tmp = leng % scale
        return leng + scale - tmp if tmp != 0 else leng

    def __call__(self, pic:Image):
        width, height = pic.size[0], pic.size[1]
        ratio = (np.random.random() - 0.5) * 2 * self.var
        new_width = int(width * (1 + ratio))
        new_width = self._recalc_len(new_width, scale=CharWidth)
        if ratio > 0:
            pic = pic.resize((new_width, height), Image.BILINEAR)
        else:
            pic = pic.resize((new_width, height), Image.ANTIALIAS)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'
