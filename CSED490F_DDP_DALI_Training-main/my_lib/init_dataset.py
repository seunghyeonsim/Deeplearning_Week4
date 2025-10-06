import os
import shutil
import argparse
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
import numpy as np

def save_cifar10(origin_dataset_dir, seed = 42):
    transform = transforms.ToTensor()
    print(f"Downloading CIFAR10 dataset to {origin_dataset_dir} ... ")
    train_dataset = datasets.CIFAR10(root=origin_dataset_dir,train=True,download=True,transform=transform)
    test_dataset = datasets.CIFAR10(root=origin_dataset_dir,train=False,download=True,transform=transform)
    print(f"Downloaded CIFAR10 dataset to {origin_dataset_dir} Done")
    return train_dataset, test_dataset

def save_cifar10_with_DALI_format(origin_dataset_dir, dali_dir, train_ratio = 1,
                                  seed = 42, train_dataset = None, test_dataset = None):
    print(f"Saving CIFAR10 dataset to {dali_dir} ... ")
    os.makedirs(os.path.join(dali_dir,'train'),exist_ok = True)
    os.makedirs(os.path.join(dali_dir,'valid'),exist_ok = True)
    os.makedirs(os.path.join(dali_dir,'test'),exist_ok = True)

    transform = transforms.ToTensor()
    if train_dataset is None:
        train_dataset = datasets.CIFAR10(root=origin_dataset_dir,train=True,download=True,transform=transform)
    if test_dataset is None:
        test_dataset = datasets.CIFAR10(root=origin_dataset_dir,train=False,download=True,transform=transform)
    class_names = train_dataset.classes
    ## First save the test dataset
    for idx, (img,label) in enumerate(test_dataset):
        class_dir = os.path.join(dali_dir,'test',class_names[label])
        os.makedirs(class_dir,exist_ok=True)
        save_image(img,os.path.join(class_dir, f'{idx}.png'))
    print(f"Saved {len(test_dataset)} images to 'test/'")
    print(f"Saving CIFAR10 dataset to {dali_dir} Done")
    # Now for splitting train dataset into train and valid and saving them to disk
    np.random.seed(seed)
    indices = np.random.permutation(len(train_dataset))
    split_idx = int(train_ratio * len(train_dataset))
    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]

    #Save train images
    for idx in train_indices:
        img,label = train_dataset[idx]
        class_dir = os.path.join(dali_dir,'train',class_names[label])
        os.makedirs(class_dir,exist_ok=True)
        save_image(img,os.path.join(class_dir,f'{idx}.png'))
    print(f"Saved {len(train_indices)} images to 'train/'")
    if train_ratio < 1.0:
        # Save val images
        for idx in valid_indices:
            img,label = train_dataset[idx]
            class_dir = os.path.join(dali_dir,'valid',class_names[label])
            os.makedirs(class_dir,exist_ok = True)
            save_image(img,os.path.join(class_dir,f'{idx}.png'))
    print(f"Saved {len(train_indices)} images to 'train/' and {len(valid_indices)} to 'val/'.")
    print(f"Saving CIFAR10 dataset to {dali_dir} Done")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    origin_dataset_dir = os.path.join(args.dataset_dir, "cifar10")
    dali_dir = os.path.join(args.dataset_dir, "cifar10_images")
    if args.force:
        if os.path.exists(origin_dataset_dir) and os.path.exists(dali_dir):
            shutil.rmtree(origin_dataset_dir)
            shutil.rmtree(dali_dir)

    if not args.force and (os.path.exists(origin_dataset_dir) and os.path.exists(dali_dir)):
        print("Dataset already exists")
        exit()

    if (os.path.exists(origin_dataset_dir) and not os.path.exists(dali_dir)) or (not os.path.exists(origin_dataset_dir) and os.path.exists(dali_dir)):
        print("Dataset is not consistent")
        print("Removing dataset and re-downloading")
        shutil.rmtree(origin_dataset_dir)
        shutil.rmtree(dali_dir)

    train_dataset, test_dataset = save_cifar10(origin_dataset_dir, args.seed)
    save_cifar10_with_DALI_format(origin_dataset_dir, dali_dir, train_dataset=train_dataset, test_dataset=test_dataset, seed=args.seed)
    print("Prepare dataset Done")