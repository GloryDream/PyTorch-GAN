import argparse
import os
import numpy as np
import math
import datetime

import torchvision.transforms as transforms
from torchvision.utils import make_grid
from dataset import CelebaDataset40

from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tqdm import tqdm

import dateutil.tz

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=24, help='dimensionality of the latent space')
# parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--n_id', type=int, default=40, help='number of classes for dataset')
parser.add_argument('--id_embed_dim', type=int, default=24, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
# parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--log_interval', type=int, default=20, help='interval between image sampling')
parser.add_argument('--save_interval', type=int, default=20, help='interval between model saving')
parser.add_argument('--reload', type=bool, default=False, help='reload or not')
parser.add_argument('--epoch', type=int, default=-1, help='reload epoch')
parser.add_argument('--name', type=str, help='the name of the model')
parser.add_argument('--data_dir', type=str, default='CelebA', help='the path of training data')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# prepare dir
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
model_name = opt.name
output_dir = os.path.join('output', model_name)
logdir_path = os.path.join(output_dir, 'log', timestamp)
model_path = os.path.join(output_dir, 'model')

# mkdir
if not opt.reload:
    os.makedirs(output_dir)
    os.makedirs(logdir_path)
    os.makedirs(model_path)
TB = SummaryWriter(log_dir=logdir_path)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_id, opt.id_embed_dim)

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim+opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


def save_checkpoint(state, filename):
    """
    from pytorch/examples
    """
    basename = os.path.dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)


def save(save_path, net, optimizer, epoch, ckpt_name):
    resume_file = os.path.join(save_path, ckpt_name)
    print('==>save', resume_file)
    save_checkpoint({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, filename=resume_file)


def save_model(netG, netD, optimizerG, optimizerD, epoch_flag, current_epoch):
    if epoch_flag == -1:
        g_ckpt_name = 'G_checkpoint.pth.tar'
        d_ckpt_name = 'D_checkpoint.pth.tar'
    else:
        g_ckpt_name = 'G_checkpoint_' + str(current_epoch) + '.pth.tar'
        d_ckpt_name = 'D_checkpoint_' + str(current_epoch) + '.pth.tar'
    save_path = model_path
    save(save_path, netG, optimizerG, current_epoch, g_ckpt_name)
    save(save_path, netD, optimizerD, current_epoch, d_ckpt_name)


def reload(netG, netD, optimizerG, optimizerD, epoch):
    restore_path = model_path

    if epoch == -1:
        g_ckpt_name = 'G_checkpoint.pth.tar'
        d_ckpt_name = 'D_checkpoint.pth.tar'
    else:
        g_ckpt_name = 'G_checkpoint_' + str(epoch) + '.pth.tar'
        d_ckpt_name = 'D_checkpoint_' + str(epoch) + '.pth.tar'

    g_resume_file = os.path.join(restore_path, g_ckpt_name)
    d_resume_file = os.path.join(restore_path, d_ckpt_name)

    if os.path.isfile(g_resume_file) and os.path.isfile(d_resume_file):
        print("=> loading checkpoint '{}'".format(g_ckpt_name))
        g_checkpoint = torch.load(g_resume_file)
        netG.load_state_dict(g_checkpoint['state_dict'])
        optimizerG.load_state_dict(g_checkpoint['optimizer'])

        print("=> loading checkpoint '{}'".format(d_ckpt_name))
        d_checkpoint = torch.load(d_resume_file)
        netD.load_state_dict(d_checkpoint['state_dict'])
        optimizerD.load_state_dict(d_checkpoint['optimizer'])

        epoch = g_checkpoint['epoch']
        print("=> loaded checkpoint (epoch {})".format(epoch))
    else:
        print("=> no checkpoint found at '{}'".format(g_ckpt_name))
        exit(0)

    return epoch

# Loss functions
adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Configure data loader
transform = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(64),
    transforms.ToTensor()
    ])
dataloader = torch.utils.data.DataLoader(CelebaDataset40(opt.data_dir, transform=transform),
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=4,
                                         drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)

    def log_image(tensor, filename, nrow=8, padding=2,
                   normalize=False, range=None, scale_each=False, pad_value=0):
        """Log a given Tensor into an image file.

        Args:
            tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
                saves the tensor as a grid of images by calling ``make_grid``.
            **kwargs: Other arguments are documented in ``make_grid``.
        """
        from PIL import Image
        grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        TB.add_image(filename + '/epoch{}'.format(epoch), grid, epoch)

    log_image(gen_imgs.data, 'imgs', nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(tqdm(dataloader)):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
        #                                                     d_loss.item(), 100 * d_acc,
        #                                                     g_loss.item()))
        # batches_done = epoch * len(dataloader) + i
    if epoch % opt.log_interval == 0:
        sample_image(n_row=10, epoch=epoch)

    # save anyway each epoch
    save_model(netG=generator, netD=discriminator, optimizerG=optimizer_G, optimizerD=optimizer_D, epoch_flag=-1,
               current_epoch=epoch)

    # save at milestone
    if epoch % opt.save_interval == 0:
        save_model(netG=generator, netD=discriminator, optimizerG=optimizer_G, optimizerD=optimizer_D, epoch_flag=1,
                   current_epoch=epoch)

# save the final results
sample_image(n_row=10, epoch=epoch)
