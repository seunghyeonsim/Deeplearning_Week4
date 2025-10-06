from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from math import ceil

import torch
import torch.distributed as dist
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

try:
    import nvidia.dali as dali
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI")

base_dir = "/SSD/CIFAR"

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# DALI는 uint8 기준이므로 255를 곱해서 넣어준다.
CIFAR_MEAN = [0.4913999 * 255, 0.48215866 * 255, 0.44653133 * 255]
CIFAR_STD  = [0.24703476 * 255, 0.24348757 * 255, 0.26159027 * 255]


def fn_dali_cutout(images, cutout_length):
    side = float(cutout_length) / 32.0
    ax = fn.random.uniform(range=(0.0, 1.0 - side))
    ay = fn.random.uniform(range=(0.0, 1.0 - side))
    anchor = fn.stack(ay, ax)
    # CHW 텐서에서 H,W 축(1,2)을 기준으로 지운다.
    return fn.erase(
        images, anchor=anchor, shape=[side, side],
        normalized_anchor=True, normalized_shape=True,
        axes=(1, 2), fill_value=0.0
    )


class DALIWrapper:
    """DALIGenericIterator를 torch DataLoader처럼 쓸 수 있게 래핑"""
    def __init__(self, dali_iter):
        self.dali_iter = dali_iter

    def __iter__(self):
        return self

    def __next__(self):
        data = self.dali_iter.__next__()[0]
        return data['data'], data['label'].squeeze(-1).long()

    def __len__(self):
        return ceil(self.dali_iter.size / self.dali_iter.batch_size)

    def reset(self):
        self.dali_iter.reset()


class CifarPipeline(Pipeline):
    """
    CIFAR-10용 DALI 파이프라인
    - train: pad(4) → random crop(32) → random horizontal flip → normalize(CHW)
             + (옵션) cutout
    - test : center/identity crop(32) → normalize(CHW)
    """
    def __init__(self, data_dir, batch_size, is_train, cutout_length,
                 device_id, shard_id, num_shards, num_workers):
        super(CifarPipeline, self).__init__(batch_size, num_threads=num_workers,
                                            device_id=device_id, seed=12345)
        self.data_dir = data_dir
        self.is_train = is_train
        self.cutout_length = cutout_length
        self.shard_id = shard_id
        self.num_shards = num_shards

    def define_graph(self):
        # 파일 리더: class-subdir 구조 가정 (…/train/cls_x/*.png, …/test/cls_x/*.png)
        images, labels = fn.readers.file(
            file_root=self.data_dir,
            random_shuffle=self.is_train,
            name="Reader",
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            pad_last_batch=self.is_train,  # train에서 drop_last와 상응
            stick_to_shard=True,
        )

        # JPEG/PNG 디코더 (혼합 모드: CPU+GPU)
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

        if self.is_train:
            # 1) padding 4 → 40x40
            images = fn.pad(images, axes=(1, 2), fill_value=0, shape=[40, 40])

            # 2) random crop 32x32 (pad된 40x40 기준)
            span = 1.0 - (32.0 / 40.0)
            ax = fn.random.uniform(range=(0.0, span))
            ay = fn.random.uniform(range=(0.0, span))
            images = fn.crop(images, crop=[32, 32], normalized_anchor=True,
                             anchor=fn.stack(ay, ax))

            # 3) random horizontal flip + normalize + CHW
            mirror = fn.random.coin_flip(probability=0.5)
            images = fn.crop_mirror_normalize(
                images, dtype=types.FLOAT, output_layout="CHW",
                mean=CIFAR_MEAN, std=CIFAR_STD, mirror=mirror
            )

            # 4) cutout (옵션)
            if self.cutout_length and self.cutout_length > 0:
                images = fn_dali_cutout(images, self.cutout_length)
        else:
            # 평가: normalize + CHW
            images = fn.crop_mirror_normalize(
                images, dtype=types.FLOAT, output_layout="CHW",
                mean=CIFAR_MEAN, std=CIFAR_STD
            )

        return images, labels


def get_DALI_loader(test_batch, train_batch, root=base_dir, valid_size=0, valid_batch=0,
                    cutout=16, num_workers=4, download=True, random_seed=12345, shuffle=True):
    """
    DALI 기반 로더 반환
    반환 형식은 기존과 동일: (test_loader, train_loader, valid_loader)
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    train_dir = os.path.join(root, "train")
    valid_dir = os.path.join(root, "valid")
    test_dir  = os.path.join(root, "test")

    train_loader = valid_loader = test_loader = None

    # ---- Train ----
    if train_batch > 0:
        train_pipe = CifarPipeline(
            data_dir=train_dir, batch_size=train_batch, is_train=True,
            cutout_length=cutout, device_id=rank, shard_id=rank,
            num_shards=world_size, num_workers=num_workers
        )
        train_pipe.build()
        train_iter = DALIGenericIterator(
            pipelines=[train_pipe],
            output_map=["data", "label"],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP  # drop_last=True와 동일
        )
        train_loader = DALIWrapper(train_iter)

    # ---- Valid (선택: valid 디렉토리가 존재할 때만) ----
    if valid_size > 0 and valid_batch > 0 and os.path.isdir(valid_dir):
        valid_pipe = CifarPipeline(
            data_dir=valid_dir, batch_size=valid_batch, is_train=False,
            cutout_length=0, device_id=rank, shard_id=rank,
            num_shards=world_size, num_workers=num_workers
        )
        valid_pipe.build()
        valid_iter = DALIGenericIterator(
            pipelines=[valid_pipe],
            output_map=["data", "label"],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL
        )
        valid_loader = DALIWrapper(valid_iter)

    # ---- Test ----
    if test_batch > 0:
        test_pipe = CifarPipeline(
            data_dir=test_dir, batch_size=test_batch, is_train=False,
            cutout_length=0, device_id=rank, shard_id=rank,
            num_shards=world_size, num_workers=num_workers
        )
        test_pipe.build()
        test_iter = DALIGenericIterator(
            pipelines=[test_pipe],
            output_map=["data", "label"],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL
        )
        test_loader = DALIWrapper(test_iter)

    return test_loader, train_loader, valid_loader
