import glob
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, SequentialSampler, DataLoader
from torchvision import transforms


class DepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, limit_cameras=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform

        self.target_paths = []
        self.raw_paths = []

        path_pattern = str(self.root_dir / "*" / "*" / "target" / "*.png") if not limit_cameras else str(self.root_dir / "*" / "image_02" / "target" / "*.png")

        for target_path in glob.glob():
            target_path = Path(target_path)

            parts = target_path.parts
            target_idx = parts.index('target')
            drive_id = parts[target_idx - 2]
            camera_id = parts[target_idx - 1]
            img_name = target_path.name

            raw_path = self.root_dir / drive_id / camera_id / 'raw' / img_name

            if raw_path.exists():
                self.target_paths.append(str(target_path))
                self.raw_paths.append(str(raw_path))

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        target_path = self.target_paths[idx]

        raw_img = Image.open(raw_path).convert('RGB')
        target_img = np.array(Image.open(target_path), dtype=int)

        depth = target_img.astype(np.float32) / 256.
        depth[target_img == 0] = -1.

        if self.transform:
            raw_img = self.transform(raw_img)
        else:
            raw_img = transforms.ToTensor()(raw_img)

        if self.target_transform:
            depth = self.target_transform(depth)
        else:
            depth = transforms.ToTensor()(depth)

        return raw_img, depth


def create_data_loader(dataset, batch_size=8, num_workers=None, distributed=False, shuffle=True, pin_memory=True, prefetch_factor=2):
    if num_workers is None:
        num_workers = min(8, multiprocessing.cpu_count())

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
