import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def model_to_DDP(model):
    ''' Problem 4: model to DDP
    Transfer model to DDP. Similar to DP, but each process must use its own GPU.
    '''
    local_rank = torch.cuda.current_device()
    torch.cuda.set_device(local_rank)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model
