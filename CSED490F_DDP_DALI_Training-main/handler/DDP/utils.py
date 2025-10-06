import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# basic rule of func:
#   1st argument: process index
#   2nd argument: collection of args, including addr, port and num_gpu
def run_process(func, args):
    ''' Problem 1: Run process
    Run main_func in multiple processes for DDP GPU group.
    Wrapper of mp.spawn function.
    '''
    nprocs = args.num_gpu
    mp.spawn(func, nprocs=nprocs, args=(args,), join=True)


def initialize_group(proc_id, host, port, num_gpu):
    ''' Problem 2: Setup GPU group
    Setup TCP connection and initialize distributed process group.
    '''
    torch.cuda.set_device(proc_id)
    dist_url = f"tcp://{host}:{port}"
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=num_gpu,
        rank=proc_id
    )
    dist.barrier()  # optional sync


def destroy_process():
    ''' Problem 6: Destroy GPU group
    Destroy distributed process group.
    '''
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
