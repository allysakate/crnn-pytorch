#python train.py --test_init True --test_epoch 10 --output_dir crnn
import os
import argparse
import string
import random
import numpy as np
from tqdm import tqdm #progress bar
from models.model_loader import load_model
from torchvision.transforms import Compose
from dataset.data_transform import Resize, Rotation, Translation, Scale
from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from lr_policy import StepLR

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from torch.nn import CTCLoss
from test import test

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/media/allysakatebrillantes/MyPassport/DATASET/catchall-dataset/cvat', help='Path to dataset')
parser.add_argument('--abc', type=str, default='1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ#', help='Alphabet')
parser.add_argument('--seq_proj', type=str, default="10x20", help='Projection of sequence')
parser.add_argument('--backend', type=str, default="resnet18", help='Backend network')
parser.add_argument('--snapshot', type=str, default=None, help='Pre-trained weights')
parser.add_argument('--input_size', type=str, default="320x32", help='Input size')
parser.add_argument('--base_lr', type=float, default=1e-3, help='Base learning rate')
parser.add_argument('--step_size', type=int, default=500, help='Step size')
parser.add_argument('--max_iter', type=int, default=600, help='Max iterations')
parser.add_argument('--max_epoch', type=int, default=250, help='Max iterations')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--output_dir', type=str, default='results', help='Path for snapshot')
parser.add_argument('--test_epoch', type=int, default=None, help='Test epoch')
parser.add_argument('--test_init', type=bool, default=False, help='Test initialization')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
opt = parser.parse_args()
print(opt)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

use_cuda = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()
print('Using cuda: ', use_cuda, cuda_count)

#dataset/data_transform.py
input_size = [int(x) for x in opt.input_size.split('x')]
transform = Compose([
    Rotation(),
    Translation(),
    Scale(),
    Resize(size=(input_size[0], input_size[1]))
])

#dataset/text_data.py
if opt.data_path is not None:
    data = TextDataset(data_path=opt.data_path, mode="train", transform=transform)
else:
    data = TestDataset(transform=transform, abc=opt.abc)
seq_proj = [int(x) for x in opt.seq_proj.split('x')]
net = load_model(data.get_abc(), seq_proj, opt.backend, opt.snapshot, use_cuda)
optimizer = optim.Adam(net.parameters(), lr = opt.base_lr, weight_decay=0.0001)
lr_scheduler = StepLR(optimizer, step_size=opt.step_size, max_iter=opt.max_iter)
loss_function =  CTCLoss(zero_infinity=True)


if use_cuda:
    net = torch.nn.DataParallel(net, device_ids=range(opt.ngpu))
    net.cuda()
    loss_function.cuda()

acc_best = 0
epoch_count = 0
while True:
    if (opt.test_epoch is not None and epoch_count != 0 and epoch_count % opt.test_epoch == 0) or (opt.test_init and epoch_count == 0):
        print("Test phase")
        data.set_mode("test")
        net = net.eval()
        acc, avg_ed = test(net, data, data.get_abc(), use_cuda, visualize=False, batch_size=opt.batch_size)
        net = net.train()
        data.set_mode("train")
        if acc > acc_best:
            if opt.output_dir is not None:
                torch.save(net.state_dict(), os.path.join(opt.output_dir, "crnn_" + opt.backend + "_" + str(data.get_abc()) + "_best"))
            acc_best = acc
        print("acc: {}\tacc_best: {}; avg_ed: {}".format(acc, acc_best, avg_ed))

    data_loader = DataLoader(data, batch_size=opt.batch_size, num_workers=1, shuffle=True, collate_fn=text_collate)
    loss_mean = []
    iterator = tqdm(data_loader)
    iter_count = 0
    for sample in iterator:
        # for multi-gpu support
        #if sample["img"].size(0) % len(opt.gpu.split(',')) != 0:
        #    continue
        optimizer.zero_grad()
        imgs = Variable(sample["img"])
        labels = Variable(sample["seq"]).view(-1)
        label_lens = Variable(sample["seq_len"].int())
        if use_cuda:
            imgs = imgs.cuda()
        preds = net(imgs).cpu()
        pred_lens = Variable(Tensor([preds.size(0)] * len(label_lens)).int())
        #torch.Size([10, 4, 38]) torch.Size([4] 10*4) torch.Size([27]) torch.Size([4] 7*4)
        #print(preds.size(), labels.size(), pred_lens, label_lens)
        loss = loss_function(preds, labels, pred_lens, label_lens) / opt.batch_size
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), 10.0)
        loss_mean.append(loss.item())
        status = "epoch: {}; iter: {}; lr: {}; loss_mean: {}; loss: {}".format(epoch_count, lr_scheduler.last_iter, lr_scheduler.get_lr(), np.mean(loss_mean), loss.item())
        iterator.set_description(status)
        optimizer.step()
        lr_scheduler.step()
        iter_count += 1
    if opt.output_dir is not None and epoch_count % 50 == 0:
        output_name = "crnn_{}_ep{}_iter{}_last.pth".format(opt.backend,epoch_count, iter_count)
        torch.save(net.state_dict(), os.path.join(opt.output_dir,output_name))
    if epoch_count == opt.max_epoch:
        break
    epoch_count += 1
