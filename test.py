import os
import cv2
import string
from tqdm import tqdm
import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from dataset.data_transform import Resize, Rotation, Translation, Scale
from models.model_loader import load_model
from torchvision.transforms import Compose

import editdistance

def test(net, data, abc, cuda, visualize, batch_size=4):
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)
    count = 0
    tp = 0
    avg_ed = 0
    iterator = tqdm(data_loader)
    for sample in iterator:
       #print(f'sample: {sample}')
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        out = net(imgs, decode=True)
        gt = (sample["seq"].numpy() - 1).tolist()
        lens = sample["seq_len"].numpy().tolist()
        pos = 0
        key = ''
        for i in range(len(out)):
            gts = ''.join(abc[c] for c in gt[pos:pos+lens[i]])
            pos += lens[i]
            if gts == out[i]:
                tp += 1
            else:
                avg_ed += editdistance.eval(out[i], gts)
            count += 1
            if visualize:
                status = "pred: {}; gt: {}".format(out[i], gts)
                iterator.set_description(status)
                img = imgs[i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
                cv2.imshow("img", img)
                key = chr(cv2.waitKey() & 255)
                if key == 'q':
                    break
        if key == 'q':
            break
        if not visualize:
            iterator.set_description("acc: {0:.4f}; avg_ed: {0:.4f}".format(tp / count, avg_ed / count))

    acc = tp / count
    avg_ed = avg_ed / count
    return acc, avg_ed

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/media/allysakatebrillantes/MyPassport/DATASET/catchall-dataset/cvat', help='Path to dataset')
parser.add_argument('--abc', type=str, default='1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ#', help='Alphabet')
parser.add_argument('--seq_proj', type=str, default="10x20", help='Projection of sequence')
parser.add_argument('--backend', type=str, default="resnet18", help='Backend network')
parser.add_argument('--snapshot', type=str, default=None, help='Pre-trained weights')
parser.add_argument('--input_size', type=str, default="320x32", help='Input size')
parser.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
parser.add_argument('--visualize', type=bool, default=False, help='Visualize output')
opt = parser.parse_args()
print(opt)


def main(data_path, abc, seq_proj, backend, snapshot, input_size, gpu, visualize):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cuda = True if gpu is not '' else False

    input_size = [int(x) for x in input_size.split('x')]
    transform = Compose([
        Rotation(),
        Resize(size=(input_size[0], input_size[1]))
    ])
    if data_path is not None:
        data = TextDataset(data_path=data_path, mode="test", transform=transform)
       #print(f'TextDataset {data[0]}')
    else:
        data = TestDataset(transform=transform, abc=abc)
    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(data.get_abc(), seq_proj, backend, snapshot, cuda).eval()
    acc, avg_ed = test(net, data, data.get_abc(), cuda, visualize)
    print("Accuracy: {}".format(acc))
    print("Edit distance: {}".format(avg_ed))

if __name__ == '__main__':
    main(opt.data_path, opt.abc, opt.seq_proj, opt.backend, opt.snapshot, opt.input_size, opt.gpu, opt.visualize)
