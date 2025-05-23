import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from typing import List
from PIL import Image
import torch
import random
import numpy as np
import copy
import torchvision.transforms.functional as tvtf
from src.models.vae import uint82fp


def center_crop_arr(pil_image, width, height):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while pil_image.size[0] >= 2 * width and pil_image.size[1] >= 2 * height:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = max(width / pil_image.size[0], height / pil_image.size[1])
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = random.randint(0, (arr.shape[0] - height))
    crop_x = random.randint(0, (arr.shape[1] - width))
    return Image.fromarray(arr[crop_y: crop_y + height, crop_x: crop_x + width])

def process_fn(width, height, data, hflip=0.5):
    image, label = data
    if random.uniform(0, 1) > hflip:   # hflip
        image = tvtf.hflip(image)
    image = center_crop_arr(image, width, height) # crop
    image = np.array(image).transpose(2, 0, 1)
    return image, label

class VARCandidate:
    def __init__(self, aspect_ratio, width, height, buffer, max_buffer_size=1024):
        self.aspect_ratio = aspect_ratio
        self.width = int(width)
        self.height = int(height)
        self.buffer = buffer
        self.max_buffer_size = max_buffer_size

    def add_sample(self, data):
        self.buffer.append(data)
        self.buffer = self.buffer[-self.max_buffer_size:]

    def ready(self, batch_size):
        return len(self.buffer) >= batch_size

    def get_batch(self, batch_size):
        batch = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]
        batch = [copy.deepcopy(b.result()) for b in batch]
        x, y = zip(*batch)
        x = torch.stack([torch.from_numpy(im).cuda() for im in x], dim=0)
        x = list(map(uint82fp, x))
        return x, y

class VARTransformEngine:
    def __init__(self,
                 base_image_size,
                 num_aspect_ratios,
                 min_aspect_ratio,
                 max_aspect_ratio,
                 num_workers = 8,
                 ):
        self.base_image_size = base_image_size
        self.num_aspect_ratios = num_aspect_ratios
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.aspect_ratios = np.linspace(self.min_aspect_ratio, self.max_aspect_ratio, self.num_aspect_ratios)
        self.aspect_ratios = self.aspect_ratios.tolist()
        self.candidates_pool = []
        for i in range(self.num_aspect_ratios):
            candidate = VARCandidate(
                aspect_ratio=self.aspect_ratios[i],
                width=int(self.base_image_size * self.aspect_ratios[i] ** 0.5 // 16 * 16),
                height=int(self.base_image_size * self.aspect_ratios[i] ** -0.5 // 16 * 16),
                buffer=[],
                max_buffer_size=1024
            )
            self.candidates_pool.append(candidate)
        self.default_candidate = VARCandidate(
            aspect_ratio=1.0,
            width=self.base_image_size,
            height=self.base_image_size,
            buffer=[],
            max_buffer_size=1024,
        )
        self.executor_pool = ProcessPoolExecutor(max_workers=num_workers)
        self._prefill_count = 100

    def find_candidate(self, data):
        image = data[0]
        aspect_ratio = image.size[0] / image.size[1]
        min_distance = 1000000
        min_candidate = None
        for candidate in self.candidates_pool:
            dis = abs(aspect_ratio - candidate.aspect_ratio)
            if dis < min_distance:
                min_distance = dis
                min_candidate = candidate
        return min_candidate


    def __call__(self, batch_data):
        self._prefill_count -= 1
        if isinstance(batch_data[0], torch.Tensor):
            batch_data[0] = batch_data[0].unbind(0)

        batch_data = list(zip(*batch_data))
        for data in batch_data:
            candidate = self.find_candidate(data)
            future = self.executor_pool.submit(process_fn, candidate.width, candidate.height, data)
            candidate.add_sample(future)
            if self._prefill_count >= 0:
                future = self.executor_pool.submit(process_fn,
                                                   self.default_candidate.width,
                                                   self.default_candidate.height,
                                                   data)
                self.default_candidate.add_sample(future)

        batch_size = len(batch_data)
        random.shuffle(self.candidates_pool)
        for candidate in self.candidates_pool:
            if candidate.ready(batch_size=batch_size):
                return candidate.get_batch(batch_size=batch_size)

        # fallback to default 256
        for data in batch_data:
            future = self.executor_pool.submit(process_fn,
                             self.default_candidate.width,
                             self.default_candidate.height,
                             data)
            self.default_candidate.add_sample(future)
        return self.default_candidate.get_batch(batch_size=batch_size)