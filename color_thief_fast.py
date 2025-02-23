# -*- coding: utf-8 -*-
# Copyright © 2025 2025 JiangLong Jia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
__version__ = '1.0.0'

import functools
from typing import Dict, Callable, List, Tuple, Optional, Union

import numpy as np
from PIL import Image


class ColorThiefFast:
    """Color thief main class."""

    def __init__(self, image: Union[str, Image.Image, np.ndarray], return_percent: bool = False):
        self.image: np.ndarray = self.process_image(image)
        self.return_percent = return_percent

    def process_image(self, image) -> np.ndarray:
        if isinstance(image, str):
            try:
                image = Image.open(image)
                image = image.convert('RGBA')
                image = np.array(image)
            except FileNotFoundError:
                raise ValueError(f"File {image} not found.")

        elif isinstance(image, Image.Image):
            image = image.convert('RGBA')
            image = np.array(image)

        elif not isinstance(image, np.ndarray):
            raise TypeError("Input must be a file path, a PIL Image, or a numpy array.")

        if image.ndim == 2:
            alpha = np.full_like(image, 255, dtype=np.uint8)
            image = np.stack((image, image, image, alpha), axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                alpha = np.full_like(image[:, :, 0], 255, dtype=np.uint8)
                image = np.repeat(image, 3, axis=2)
                image = np.dstack((image, alpha))
            elif image.shape[2] == 3:
                alpha = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
                image = np.dstack((image, alpha))
            elif image.shape[2] != 4:
                raise ValueError("Unsupported number of channels in the numpy array.")
        else:
            raise ValueError("The input image is incorrect")

        return image

    def get_color(self, quality: int = 10) -> Union[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """Get the dominant color.

        Args:
            quality:
                quality settings, 1 is the highest quality, the bigger
                the number, the faster a color will be returned but
                the greater the likelihood that it will not be the
                visually most dominant color

        Returns:
            tuple: (r, g, b) or (r, g, b, percent)
        """
        palette = self.get_palette(5, quality)
        return palette[0]

    def get_palette(self,
                    color_count: int = 10,
                    quality: int = 10,
                    custom_processing: Callable[[np.ndarray], np.ndarray] = None
                    ) -> Union[List[Tuple[int, int, int]], List[Tuple[int, int, int, int]]]:
        """Build a color palette.  We are using the median cut algorithm to
        cluster similar colors.

        Args:
            color_count:
                the size of the palette, max number of colors
            quality:
                quality settings, 1 is the highest quality, the bigger
                the number, the faster the palette generation, but the
                greater the likelihood that colors will be missed.
            custom_processing:
                Customize pixel processing functions.
        Returns:
            list: a list of tuple in the form (r, g, b) or (r, g, b, percent)
        """
        if custom_processing is None:
            custom_processing = PixelsProcessor.get_valid_pixels

        image_array = np.reshape(self.image, (-1, 4))
        image_array = image_array[::quality, :]

        valid_pixels = custom_processing(image_array)

        # Send array to quantize function which clusters values using median cut algorithm
        cmap = MMCQ.quantize(valid_pixels, color_count)
        cmap.return_percent = self.return_percent
        return cmap.palette


class PixelsProcessor:
    def get_valid_pixels(image_array: np.ndarray) -> np.ndarray:
        """Default method to obtain valid pixels"""
        condition = ((image_array[..., 3] >= 125) &
                     ~((image_array[..., 0] > 250) & (image_array[..., 1] > 250) & (image_array[..., 2] > 250)))

        valid_pixels = image_array[condition][:, :3]
        return valid_pixels


class MMCQ:
    """Basic Python port of the MMCQ (modified median cut quantization)
    algorithm from the Leptonica library (http://www.leptonica.com/).
    """
    SIGBITS = 5
    RSHIFT = 8 - SIGBITS
    MAX_ITERATION = 1000
    FRACT_BY_POPULATIONS = 0.75

    @staticmethod
    def get_color_index(r, g, b):
        """Each quantified (r,g,b) will calculate a unique index value"""
        return (r << (2 * MMCQ.SIGBITS)) + (g << MMCQ.SIGBITS) + b

    @staticmethod
    def get_histo(pixels: np.ndarray) -> Dict[int, int]:
        """Calculate the number of pixels in each quantized region of the color space,
        and save it in histo dict.
        """
        pixels = pixels.astype(np.uint32)

        r = np.right_shift(pixels[:, 0], MMCQ.RSHIFT)
        g = np.right_shift(pixels[:, 1], MMCQ.RSHIFT)
        b = np.right_shift(pixels[:, 2], MMCQ.RSHIFT)
        color_index_array = MMCQ.get_color_index(r, g, b)

        unique_indices, counts = np.unique(color_index_array, return_counts=True)
        histo = {i: int(count) for i, count in zip(unique_indices, counts) if count != 0}
        return histo

    @staticmethod
    def vbox_from_pixels(pixels: np.ndarray, histo: Dict[int, int]) -> "VBox":
        rval = np.right_shift(pixels[:, 0], MMCQ.RSHIFT)
        gval = np.right_shift(pixels[:, 1], MMCQ.RSHIFT)
        bval = np.right_shift(pixels[:, 2], MMCQ.RSHIFT)
        rmin = rval.min()
        rmax = rval.max()
        gmin = gval.min()
        gmax = gval.max()
        bmin = bval.min()
        bmax = bval.max()
        return VBox(rmin, rmax, gmin, gmax, bmin, bmax, histo, pixels.shape[0])

    @staticmethod
    def median_cut_apply(histo, vbox: "VBox"):
        if not vbox.count:
            return (None, None)

        rw = vbox.r2 - vbox.r1 + 1
        gw = vbox.g2 - vbox.g1 + 1
        bw = vbox.b2 - vbox.b1 + 1
        maxw = max([rw, gw, bw])

        # only one pixel, no split
        if vbox.count == 1:
            return (vbox.copy, None)

        # Find the partial sum arrays along the selected axis.
        partial_sum = {}
        lookahead_sum = {}

        if maxw == rw:
            do_cut_color = 'r'
            column_num = 0
        elif maxw == gw:
            do_cut_color = 'g'
            column_num = 1
        else:
            do_cut_color = 'b'
            column_num = 2

        rgb_array, color_index_array = vbox.get_rgb_array_and_color_index_array()
        color_num_list = [histo.get(index, 0) for index in color_index_array]
        unique_values, indices = np.unique(rgb_array[:, column_num], return_inverse=True)
        color_sum_array = np.bincount(indices, weights=color_num_list)

        total_array = np.cumsum(color_sum_array, dtype=np.int32)
        for i, t in zip(unique_values, total_array):
            partial_sum[i] = t

        total = total_array[-1]
        for i, d in partial_sum.items():
            lookahead_sum[i] = total - d

        # determine the cut planes
        dim1 = do_cut_color + '1'
        dim2 = do_cut_color + '2'
        dim1_val = getattr(vbox, dim1)
        dim2_val = getattr(vbox, dim2)
        for i in range(dim1_val, dim2_val + 1):
            if partial_sum[i] > (total / 2):
                vbox1 = vbox.copy
                vbox2 = vbox.copy
                left = i - dim1_val
                right = dim2_val - i
                if left <= right:
                    d2 = min([dim2_val - 1, int(i + right / 2)])
                else:
                    d2 = max([dim1_val, int(i - 1 - left / 2)])

                # avoid 0-count boxes
                while not partial_sum.get(d2, False):
                    d2 += 1
                count2 = lookahead_sum.get(d2)
                while not count2 and partial_sum.get(d2 - 1, False):
                    d2 -= 1
                    count2 = lookahead_sum.get(d2)

                # set dimensions
                setattr(vbox1, dim2, d2)
                setattr(vbox2, dim1, getattr(vbox1, dim2) + 1)
                return (vbox1, vbox2)
        return (None, None)

    @staticmethod
    def quantize(pixels: np.ndarray, max_color: int):
        """pixels quantize.

        Args:
            pixels: a array of pixel in the form (r, g, b)
            max_color: max number of colors

        Returns:

        """
        if pixels.shape[0] == 0:
            raise Exception('Empty pixels when quantize.')
        if max_color < 2 or max_color > 256:
            raise Exception('Wrong number of max colors when quantize.')

        histo = MMCQ.get_histo(pixels)

        # get the beginning vbox from the colors
        vbox = MMCQ.vbox_from_pixels(pixels, histo)
        pq = PQueue(lambda x: x.count)
        pq.push(vbox)

        # inner function to do the iteration
        def iter_(lh: "PQueue", target: int | float):
            n_color = 1
            n_iter = 0
            while n_iter < MMCQ.MAX_ITERATION:
                vbox = lh.pop()
                if not vbox.count:  # just put it back
                    lh.push(vbox)
                    n_iter += 1
                    continue

                # do the cut
                vbox1, vbox2 = MMCQ.median_cut_apply(histo, vbox)
                if not vbox1:
                    raise Exception("vbox1 not defined; shouldn't happen!")
                lh.push(vbox1)

                if vbox2:  # vbox2 can be null
                    lh.push(vbox2)
                    n_color += 1
                if n_color >= target:
                    return
                if n_iter > MMCQ.MAX_ITERATION:
                    return
                n_iter += 1

        # first set of colors, sorted by population
        iter_(pq, MMCQ.FRACT_BY_POPULATIONS * max_color)

        # Re-sort by the product of pixel occupancy times the size in color space.
        pq2 = PQueue(lambda x: x.count * x.volume)
        while pq.size():
            pq2.push(pq.pop())

        # next set - generate the median cuts using the (npix * vol) sorting.
        iter_(pq2, max_color - pq2.size())

        # calculate the actual colors
        cmap = CMap()
        while pq2.size():
            cmap.push(pq2.pop())
        return cmap


class VBox:
    """3d color space box"""

    def __init__(self, r1, r2, g1, g2, b1, b2, histo: Dict[int, int], total_pixel_count: int):
        self.r1 = r1
        self.r2 = r2
        self.g1 = g1
        self.g2 = g2
        self.b1 = b1
        self.b2 = b2
        self.histo = histo
        self.total_pixel_count = total_pixel_count

        self._rgb_array: Optional[np.ndarray] = None
        self._color_index: Optional[np.ndarray] = None

    @functools.cached_property
    def volume(self) -> int:
        """The volume of pixels in VBox"""
        sub_r = self.r2 - self.r1
        sub_g = self.g2 - self.g1
        sub_b = self.b2 - self.b1
        return (sub_r + 1) * (sub_g + 1) * (sub_b + 1)

    @property
    def copy(self) -> "VBox":
        return VBox(self.r1, self.r2, self.g1, self.g2, self.b1, self.b2, self.histo, self.total_pixel_count)

    @functools.cached_property
    def avg(self) -> Tuple[int, int, int]:
        """Calculate the color avg of VBox"""
        mult = 1 << (8 - MMCQ.SIGBITS)
        rgb_array, color_index_array = self.get_rgb_array_and_color_index_array()
        histo_num = np.array([self.histo.get(c, 0) for c in color_index_array], dtype=np.int32)
        r_sum = np.sum(histo_num * (rgb_array[:, 0] + 0.5) * mult)
        g_sum = np.sum(histo_num * (rgb_array[:, 1] + 0.5) * mult)
        b_sum = np.sum(histo_num * (rgb_array[:, 2] + 0.5) * mult)

        total = histo_num.sum()
        if total:
            r_avg = int(r_sum / total)
            g_avg = int(g_sum / total)
            b_avg = int(b_sum / total)
        else:
            r_avg = int(mult * (self.r1 + self.r2 + 1) / 2)
            g_avg = int(mult * (self.g1 + self.g2 + 1) / 2)
            b_avg = int(mult * (self.b1 + self.b2 + 1) / 2)

        r_avg = np.clip(r_avg, 0, 255)
        g_avg = np.clip(g_avg, 0, 255)
        g_avg = np.clip(g_avg, 0, 255)
        return r_avg, g_avg, b_avg

    @functools.cached_property
    def count(self) -> int:
        """The number of pixels in VBox"""
        rgb_array, color_index_array = self.get_rgb_array_and_color_index_array()
        npix_list = [self.histo.get(i, 0) for i in color_index_array]
        npix = sum(npix_list)
        return npix

    @functools.cached_property
    def percent(self) -> int:
        """The percentage of VBox pixels to all pixels"""
        return int(self.count / self.total_pixel_count * 100)

    def get_rgb_array(self):
        return np.mgrid[self.r1: self.r2 + 1, self.g1: self.g2 + 1, self.b1: self.b2 + 1].T.reshape(-1, 3)

    def get_rgb_array_and_color_index_array(self):
        if self._rgb_array is None:
            rgb_array = np.mgrid[self.r1: self.r2 + 1, self.g1: self.g2 + 1, self.b1: self.b2 + 1].T.reshape(-1, 3)
            self._rgb_array = rgb_array
            self._color_index_array = MMCQ.get_color_index(rgb_array[:, 0], rgb_array[:, 1], rgb_array[:, 2])
        return self._rgb_array, self._color_index_array


class CMap:
    """Color map"""

    def __init__(self, return_percent: bool = False):
        self.pqueue: "PQueue" = PQueue(lambda x: x['vbox'].count * x['vbox'].volume)
        self.return_percent = return_percent

    @property
    def palette(self) -> List:
        if self.return_percent:
            return self.pqueue.map(lambda x: x['color'] + (x['percent'],))
        else:
            return self.pqueue.map(lambda x: x['color'])

    def push(self, vbox: "VBox"):
        self.pqueue.push({
            'vbox': vbox,
            'color': vbox.avg,
            'percent': vbox.percent,
        })

    def size(self):
        return self.pqueue.size()


class PQueue:
    """Simple priority queue."""

    def __init__(self, sort_key: Callable):
        self.sort_key = sort_key
        self.contents: List = []
        self._sorted = False

    def sort(self):
        self.contents.sort(key=self.sort_key)
        self._sorted = True

    def push(self, o):
        self.contents.append(o)
        self._sorted = False

    def pop(self):
        if not self._sorted:
            self.sort()
        return self.contents.pop()

    def size(self) -> int:
        return len(self.contents)

    def map(self, f: Callable) -> List:
        return list(map(f, self.contents))
