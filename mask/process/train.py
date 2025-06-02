import os
import sys
import time
import yaml
import cv2
import pprint
import traceback
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import CustomDatasetDataLoader, sample_data
from utils.parser import parser
from utils.checkpoints import save_checkpoints
from utils.checkpoints import load_checkpoint
from network.u2net import U2NET



def options_printing_saving(opt):
    os.makedirs(opt.logs_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "checkpoints"), exist_ok=True)

    option_dict = vars(opt)
    with open(os.path.join(opt.save_dir, "training_options.yml"), "w") as outfile:
        yaml.dump(option_dict, outfile)

    for key, value in option_dict.items():
        print(key, value)


def training_loop(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u_net = U2NET(in_ch=3, out_ch=4)
    if opt.continue_train:
        u_net = load_checkpoint(u_net, opt.unet_checkpoint)
    u_net = u_net.to(device)
    u_net.train()

    with open('output/logs', "w") as outfile:
        print("<----U-2-Net---->", file=outfile)
        print(u_net, file=outfile)

    optimizer = optim.Adam(
        u_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    custom_dataloader = CustomDatasetDataLoader()
    custom_dataloader.initialize(opt)
    loader = custom_dataloader.get_loader()

    dataset_size = len(custom_dataloader)
    print("Total number of images available for training: %d" % dataset_size)
    writer = SummaryWriter(opt.logs_dir)
    print("Entering training loop!")

    weights = np.array([1, 1.5, 1.5, 1.5], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)

    pbar = range(opt.iter)
    get_data = sample_data(loader)

    start_time = time.time()
    for itr in pbar:
        data_batch = next(get_data)
        image, label = data_batch
        image = Variable(image.to(device))
        label = label.type(torch.long)
        label = Variable(label.to(device))

        d0, d1, d2, d3, d4, d5, d6 = u_net(image)

        loss0 = loss_CE(d0, label)
        loss1 = loss_CE(d1, label)
        loss2 = loss_CE(d2, label)
        loss3 = loss_CE(d3, label)
        loss4 = loss_CE(d4, label)
        loss5 = loss_CE(d5, label)
        loss6 = loss_CE(d6, label)
        del d1, d2, d3, d4, d5, d6

        total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        optimizer.zero_grad()
        total_loss.backward()
        if opt.clip_grad != 0:
            nn.utils.clip_grad_norm_(u_net.parameters(), opt.clip_grad)
        optimizer.step()

        if itr % opt.print_freq == 0:
            pprint.pprint(
                "[step-{:08d}] [time-{:.3f}] [total_loss-{:.6f}]  [loss0-{:.6f}]".format(
                    itr, time.time() - start_time, total_loss, loss0
                )
            )

            writer.add_scalar("total_loss", total_loss, itr)
            writer.add_scalar("loss0", loss0, itr)

            if itr % opt.save_freq == 0:
                save_checkpoints(opt, itr, u_net)

    print("Training done!")
    # if local_rank == 0:
    itr += 1
    save_checkpoints(opt, itr, u_net)


if __name__ == "__main__":

    opt = parser()
    options_printing_saving(opt)
    try:
        training_loop(opt)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
    print("Exiting..............")