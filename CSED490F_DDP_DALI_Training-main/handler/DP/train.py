from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import glob
import time
import torch
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import nvtx

_print_freq = 50

def set_print_freq(freq):
    global _print_freq
    _print_freq = freq

from my_lib.train_test import (
    AverageMeter,
    accuracy,
)

def is_main_process():
    return True

def create_checkpoint(save_dir, 
                      model, 
                      optimizer, 
                      is_best, 
                      best_acc, 
                      epoch, 
                      save_freq=10, 
                      prefix='train'):
    if not is_main_process(): 
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(save_dir, '{}_{}.ckpt'.format(prefix, epoch))
    bestname = os.path.join(save_dir, '{}_best.pth'.format(prefix))
    
    if is_best or (epoch+1) % save_freq == 0: # save model state dict
        if isinstance(model, (DP, DDP)): # DP / DDP wrap the model, so we need to get module for extract state_dict
            model_state = model.module.state_dict()        
        else:
            model_state = model.state_dict()
        
    if is_best:
        torch.save(model_state, bestname)

    if (epoch+1) % save_freq == 0:
        state = {
            'model': model_state,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(state, filename)

        
def resume_checkpoint(model, optimizer, root, prefix='train'):  
    files = glob.glob(os.path.join(root, "{}_*.ckpt".format(prefix)))

    max_idx = -1
    for file in files:
        num = re.search("{}_(\d+).ckpt".format(prefix), file)
        if num is not None:
            num = num.group(1)
            max_idx = max(max_idx, int(num))

    if max_idx != -1:
        checkpoint = torch.load(
            os.path.join(root, "{}_{}.ckpt".format(prefix, max_idx)), 
            map_location={'cuda:%d' % 0: 'cuda:%d' % torch.cuda.current_device()})
        
        if isinstance(model, (DP, DDP)):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
            
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        
        return (epoch, best_acc)
    else:
        print("==> Can't find checkpoint...training from initial stage")
        return (-1, 0)

def train(train_loader, model, criterion, optimizer, epoch):    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()
    end = time.time()

    with nvtx.annotate(f"Epoch {epoch}"):
        for i, (input, target) in enumerate(train_loader):  
            with nvtx.annotate(f"Batch {i}"):
                batch_size = torch.IntTensor([input.size(0)]).cuda(non_blocking=True)
                
                # measure data loading time
                data_time.update(time.time() - end)

                with nvtx.annotate("upload data to GPU"):
                    with torch.no_grad():
                        target = target.cuda(non_blocking=True)
                        input = input.cuda(non_blocking=True)
                        input_var = torch.autograd.Variable(input)
                        target_var = torch.autograd.Variable(target)

                # compute output
                with nvtx.annotate("forward"):
                    output = model(input_var)

                with nvtx.annotate("loss"):
                    loss = criterion(output, target_var)
                
                # compute gradient and do SGD step
                with nvtx.annotate("backward"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # measure accuracy and record loss                       
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))        
            batch_size = batch_size.item()
            losses.update(loss.item() / batch_size, batch_size)
            top1.update(prec1.item() / batch_size, batch_size)
            top5.update(prec5.item() / batch_size, batch_size)
            # batch_size = reduce_tensor(batch_size).item() 
            # losses.update(reduce_tensor(loss * input.size(0)).item() / batch_size, batch_size)               
            # top1.update(reduce_tensor(prec1).item() / batch_size, batch_size)
            # top5.update(reduce_tensor(prec5).item() / batch_size, batch_size)
                        
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if is_main_process() and ((i+1) % _print_freq) == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i+1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
            
    if is_main_process():
        print('Epoch {0} : Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(epoch, loss=losses, top1=top1, top5=top5))
        

def test(val_loader, model, criterion, epoch, train=False):   
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.train(train)

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        batch_size = torch.IntTensor([input.size(0)]).cuda(non_blocking=True)
        
        with torch.no_grad():                   
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)              
            loss = criterion(output, target_var)
            
        # record loss and accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        batch_size = batch_size.item()
        losses.update(loss.item() / batch_size, batch_size)
        top1.update(prec1.item() / batch_size, batch_size)
        top5.update(prec5.item() / batch_size, batch_size)
        # batch_size = reduce_tensor(batch_size).item() 
        # losses.update(reduce_tensor(loss * input.size(0)).item() / batch_size, batch_size)                
        # top1.update(reduce_tensor(prec1).item() / batch_size, batch_size)
        # top5.update(reduce_tensor(prec5).item() / batch_size, batch_size)
                        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and ((i+1) % _print_freq) == 0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i+1, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if is_main_process():
        print('Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(loss=losses, top1=top1, top5=top5))

    return top1.avg