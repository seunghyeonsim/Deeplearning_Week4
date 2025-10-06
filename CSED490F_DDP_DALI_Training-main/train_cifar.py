import argparse
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel as DP
import nvtx

from my_lib.train_test import (
    CosineWithWarmup
)
from my_lib.utils import (
    find_free_port,
)

parser = argparse.ArgumentParser()
parser.add_argument("--num_gpu", type=int, required=True)
parser.add_argument("--ip", default="localhost")
parser.add_argument("--port", default=find_free_port())

parser.add_argument("--data", default="/SSD/CIFAR")
parser.add_argument("--ckpt", required=True, help="checkpoint directory")
parser.add_argument("--save_ckpt", action="store_true", help="save checkpoint")

parser.add_argument("--lr", default=0.4, type=float)
parser.add_argument("--decay", default=4e-5, type=float)
parser.add_argument("--seed", default=42, type=int)

parser.add_argument("--mode", choices=["dp", "ddp", "ddp_dali"], default="dp")

args = parser.parse_args()


    
def main_func(proc_id, args):
    try:
        # Setup GPU group for DDP multiprocess.
        if args.mode == "dp":
            pass
        elif args.mode == "ddp" or args.mode == "ddp_dali":
            ''' Problem 2: Setup GPU group
            (./handler/DDP/utils.py)
            DDP requires to setup GPU group, which can broadcast weights to all GPUs.
            This function set tcp connection between processes.
            Implement initialize_group function.
            '''
            from handler.DDP.utils import initialize_group
            initialize_group(proc_id, args.ip, args.port, args.num_gpu)
            
        ckpt_root = args.ckpt
        data_root = args.data
        use_cuda = torch.cuda.is_available()
        
        if use_cuda is False:
            print("Current DP implementation doesn't consider non-cuda execution")
            raise NotImplementedError()
        
        # Prepare data loader.
        print("==> Prepare data..")
        get_loader = None
        if args.mode == "dp":
            from handler.DP.cifar10_loader import get_DP_loader
            get_loader = get_DP_loader
        elif args.mode == "ddp":
            ''' Problem 5: Get DDP loader
            (./handler/DDP/cifar10_loader.py)
            Implement get_DDP_loader function.
            get_DDP_loader function is used to get DDP loader.
            You can use get_DP_loader function as a reference.
            '''
            from handler.DDP.cifar10_loader import get_DDP_loader
            get_loader = get_DDP_loader
        elif args.mode == "ddp_dali":
            ''' 
            Problem 6: make DALI Pipeline
            (./handler/DALI/cifar10_loader.py)
            To implement the get_DALI_loader function, you need to implement CifarPipeline class.
            CifarPipeline class is a DALI pipeline for CIFAR-10 dataset.
            Because DALI data process pipeline is differenct from general data loader, you should implement DALI pipeline in your own.

            +++

            Problem 7: Get DALI loader
            (./handler/DALI/cifar10_loader.py)
            Implement get_DALI_loader function.
            get_DALI_loader function is used to get DALI loader.
            Because DALI loader is slightly different with DP/DDP loader, you may change few parts of get_DP_loader.
            '''
            from handler.DALI.cifar10_loader import get_DALI_loader
            get_loader = get_DALI_loader

        if get_loader is None:
            raise NotImplementedError()

        with nvtx.annotate("Set data loader"):
            test_loader, train_loader, _ = get_loader(root=data_root, test_batch=512, train_batch=512, download=False)
                
        # Prepare ResNet18 model.
        with nvtx.annotate("Set Data Parallelism"):
            print("==> Prepare model..")
            class_num = 10
            from models.cifar.resnet import ResNet18
            model = ResNet18(class_num)
            if args.mode == "dp": # General DP only requires to transfer model to GPU, simply.
                device_ids = list(range(torch.cuda.device_count()))
                model = model.to(f'cuda:{device_ids[0]}')
                model = DP(model, device_ids=device_ids)
            elif args.mode == "ddp" or args.mode == "ddp_dali": # Handle DDP multiprocess model load.
                ''' Problem 4: model to DDP
                (./handler/DDP/model.py)
                Implement model_to_DDP function.
                model_to_DDP function is used to transfer model to DDP, SIMILAR with DP.
                Be careful for set devices. Set profer device id is important part in DDP.
                '''
                from handler.DDP.model import model_to_DDP
                model = model_to_DDP(model)

        
        best_acc = 0  # best test accuracy
        last_epoch = -1  # start from epoch 0 or last checkpoint epoch

        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(add_weight_decay(model), lr=0.4, momentum=0.9, nesterov=True)
        
        # Prepare train functions
        if args.mode == "dp":
            from handler.DP.train import train, test, create_checkpoint
        elif args.mode == "ddp" or args.mode == "ddp_dali":
            from handler.DDP.train import sync_checkpoint, train, test, create_checkpoint

        if args.mode == "ddp" or args.mode == "ddp_dali":
            with nvtx.annotate("Sync model parameters"):
                print("==> Sync model parameters")
                sync_checkpoint(ckpt_root, model)
            
        print("==> Start training..")
        target_epoch = 1
        warmup_len = 1
        scheduler = CosineWithWarmup(optimizer, 
                        warmup_len=warmup_len, warmup_start_multiplier=0.1,
                        max_epochs=target_epoch, eta_min=1e-2, last_epoch=last_epoch)
        
        for epoch in range(last_epoch+1, target_epoch):
            train(train_loader, model, criterion, optimizer, epoch)
            acc = test(test_loader, model, criterion, epoch)

            if acc > best_acc:
                best_acc = acc
                is_best = True
            else:
                is_best = False
            
            if args.save_ckpt:
                create_checkpoint(ckpt_root, 
                                model, optimizer,
                                is_best, 
                                best_acc,                        
                                epoch)
            scheduler.step()

    finally:
        if args.mode == "ddp" or args.mode == "ddp_dali":
            ''' Problem 3: Destroy GPU group
            (./handler/DDP/utils.py)
            Implement destroy_process function.
            Just call the torch.distributed's destroy function.
            '''
            from handler.DDP.utils import destroy_process
            destroy_process()

if __name__ == "__main__":
    if args.mode == "dp":
        main_func(0, args)
    elif args.mode == "ddp" or args.mode == "ddp_dali":
        ''' Problem 1: Run process
        (./handler/DDP/utils.py)
        Implement run_process function.
        run_process function is used to run main_func in multiple processes, for DDP GPU group.
        It is a wrapper of mp.spawn function.
        You can use mp.spawn function as a reference.
        '''
        from handler.DDP.utils import run_process
        run_process(main_func, args)

    