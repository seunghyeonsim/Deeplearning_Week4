from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

base_dir = "/SSD/CIFAR"

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

CIFAR_MEAN = (0.4913999, 0.48215866, 0.44653133)
CIFAR_STD = (0.24703476, 0.24348757, 0.26159027)

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])


def get_DDP_loader(test_batch, train_batch, root=base_dir, valid_size=0, valid_batch=0,
                   cutout=16, num_workers=0, download=True, random_seed=12345, shuffle=True):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # (A) base datasets
    base_train = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform_train)
    base_valid = torchvision.datasets.CIFAR10(   # <-- 검증용은 transform_test
        root=root, train=True, download=False, transform=transform_test)

    if valid_size > 0:
        gen = torch.Generator().manual_seed(random_seed)
        train_indices, valid_indices = torch.utils.data.random_split(
            range(50000), [50000 - valid_size, valid_size], generator=gen)
        # 서로 다른 base + 같은 인덱스 집합을 사용
        train_dataset = torch.utils.data.Subset(base_train, train_indices.indices)
        valid_dataset = torch.utils.data.Subset(base_valid, valid_indices.indices)
    else:
        train_dataset = base_train
        valid_dataset = None

    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=download, transform=transform_test)

    # (B) samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = (DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                     if valid_dataset is not None else None)
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=rank, shuffle=False)

    # (C) loaders
    if train_batch > 0:
        if cutout > 0:
            transform_train.transforms.append(Cutout(cutout))  # 각 프로세스 1회 호출 → OK
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(num_workers > 0))
    else:
        train_loader = None

    if valid_size > 0:
        assert valid_batch > 0, "validation set follows test batch size"
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=valid_batch, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0))
    else:
        valid_loader = None

    if test_batch > 0:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch, sampler=test_sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0))
    else:
        test_loader = None

    return test_loader, train_loader, valid_loader

