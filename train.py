from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torchvision
import numpy as np
import cv2 as cv

import dataset
from TrackNet import TrackNet

from sys import modules
if "ipykernel" in modules:  # executed in a jupyter notebook
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def wbce_loss(output, target):
    return -(
        ((1-output)**2) * target *
        torch.log(torch.clamp(output, min=1e-15, max=1)) +
        output**2 * (1-target) *
        torch.log(torch.clamp(1-output, min=1e-15, max=1))
    ).sum()
    

def euclidean_loss(output, target):
    return ((output-target)**2).sum()


def my_loss(output, target):
    diff = output - target
    diff[diff < 0] *= 5
    return (diff ** 2).mean()


class AdaptiveWingLoss(torch.nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

    
#TODO: save_path for images
def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='Path to initial weights the model should be loaded with. If not specified, the model will be initialized with random weights.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint, chekpoint differs from weights due to including information about current loss, epoch and optimizer state.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size of the training dataset.')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size of the validation dataset.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--train_size', type=float, default=0.8, help='Training dataset size.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate. If equals to 0.0, no dropout is used.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--sequence_length', type=int, default=3, help='Length of the images sequence used as X.')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024], help='Size of the images used for training (y, x).')
    parser.add_argument('--dataset', type=str, default='dataset/', help='Path to dataset.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, mps).')
    parser.add_argument('--type', type=str, default='auto', help='Type of dataset to create (auto, image, video). If auto, the dataset type will be inferred from the dataset directory, defaulting to image.')
    parser.add_argument('--save_period', type=int, default=1, help='Save checkpoint every x epochs (disabled if <1).')
    parser.add_argument('--save_path', type=str, default='weights/', help='Path to save checkpoints at.')
    parser.add_argument('--images_dir', type=str, default='images/', help="Path to dataset's images.")
    parser.add_argument('--videos_dir', type=str, default='videos/', help="Path to dataset's videos.")
    parser.add_argument('--csvs_dir', type=str, default='csvs/', help="Path to dataset's csv files.")
    parser.add_argument('--save_weights_only', action='store_true', help='Save only weights, not the whole checkpoint')
    parser.add_argument('--no_shuffle', action='store_true', help="Don't shuffle the training dataset.")
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard to log training progress.')
    parser.add_argument('--one_output_frame', action='store_true', help='Demand only one output frame instead of three.')
    parser.add_argument('--no_save_output_examples', action='store_true', help="Don't save output examples to results folder.")
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale images instead of RGB.')
    parser.add_argument('--single_batch_overfit', action='store_true', help='Overfit the model on a single batch.')
    opt = parser.parse_args()
    return opt


#TODO: add ball detection
#TODO: add accuracy
#TODO: add inference
def training_loop(opt, device, model, writer, loss_function, optimizer, train_loader, val_loader):
    for epoch in range(opt.epochs):
        tqdm.write("Epoch: " + str(epoch))
        running_loss = 0.0
        
        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            # running loss calculation
            running_loss += loss.item()
            pbar.set_description(f'Loss: {running_loss / (batch_idx+1):.6f}')
            
            if batch_idx % opt.save_period == 0:
                with torch.inference_mode():
                    writer.add_scalar('Loss/train/batch', running_loss / (batch_idx+1), epoch * len(train_loader) + batch_idx)
                    if not opt.no_save_output_examples:
                        save_image(y[0,0,:,:], f'results/epoch_{epoch}_batch{batch_idx}_y.png')
                        save_image(y_pred[0,0,:,:], f'results/epoch_{epoch}_batch{batch_idx}_y_pred.png')
                        save_image(torch.abs((y[0,0,:,:] - y_pred[0,0,:,:])), f'results/epoch_{epoch}_batch{batch_idx}_y_diff.png')
                        # writer.add_image('Train/y', torch.unsqueeze(y[0,0,:,:], 0), epoch * len(train_loader) + batch_idx)
                        # writer.add_image('Train/y_pred', torch.unsqueeze(y_pred[0,0,:,:], 0), epoch * len(train_loader) + batch_idx)
                        images = [
                            torch.unsqueeze(y[0,0,:,:], 0).repeat(3,1,1).cpu(),
                            torch.unsqueeze(y_pred[0,0,:,:], 0).repeat(3,1,1).cpu(),
                        ]
                        if opt.grayscale:
                            # writer.add_image('Train/X', X[0,:,:,:], epoch * len(train_loader) + batch_idx)
                            save_image(X[0,:,:,:], f'results/epoch_{epoch}_batch{batch_idx}_X.png')
                            images.append(X[0,:,:,:].cpu())
                            res = X[0,:,:,:] * y[0,0,:,:]

                        else:
                            # writer.add_image('Train/X', X[0,(2,1,0),:,:], epoch * len(train_loader) + batch_idx)
                            save_image(X[0,(2,1,0),:,:], f'results/epoch_{epoch}_batch{batch_idx}_X.png')
                            images.append(X[0,(2,1,0),:,:].cpu())
                            res = X[0, (2,1,0),:,:] * y[0,0,:,:]
                        images.append(res.cpu())
                        save_image(res, f'results/epoch_{epoch}_batch{batch_idx}_mask.png' )
                        grid = torchvision.utils.make_grid(images, nrow=1)#, padding=2)
                        writer.add_image('Train', grid, epoch*len(train_loader) + batch_idx)

        if val_loader is not None:
            val_loss = validation_loop(device, model, loss_function, val_loader)
            writer.add_scalars('Loss', {'train': running_loss / len(train_loader), 'val': val_loss}, epoch)

            # save the model
            if epoch % opt.save_period == opt.save_period - 1:
                save_path = ((Path(opt.save_path) / str(datetime.now())) / f"_epoch:{epoch}").with_suffix('.pth')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if opt.save_weights_only:
                    tqdm.write('\n--- Saving weights to: ' + str(save_path))
                    save_path = save_path.name+"_weights_only".with_suffix('.pth')
                    torch.save(model.state_dict(), save_path)   
                else:
                    tqdm.write('\n--- Saving checkpoint to: ' + str(save_path))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        }, save_path)


def validation_loop(device, model, loss_function, val_loader):
    model.eval()
    loss_sum = 0
    with torch.inference_mode():
        for X, y in tqdm(val_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss_sum += loss_function(y_pred, y)
        tqdm.write('Validation loss: ' + str(loss_sum/len(val_loader)))
    
    return loss_sum/len(val_loader)


if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device(opt.device)
    model = TrackNet(opt).to(device)
    writer = SummaryWriter()
    # loss_function = torch.nn.MSELoss()
    # loss_function = euclidean_loss
    # loss_function = torch.nn.HuberLoss()
    # loss_function = wbce_loss # doesn't work for some reason
    # loss_function = torch.nn.L1Loss()
    # loss_function = AdaptiveWingLoss()
    loss_function = my_loss

    if opt.weights:
        model.load_state_dict(torch.load(opt.weights))

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    #TODO: smaller ball in ground truth?


    if opt.type == 'auto':
        full_dataset = dataset.GenericDataset.from_dir(opt)
    elif opt.type == 'image':
        full_dataset = dataset.ImagesDataset(opt)
    elif opt.type == 'video':
        full_dataset = dataset.VideosDataset(opt)
    
    full_dataset.video_tags = full_dataset.video_tags[full_dataset.video_tags['ball_type'] == 'MikasaNewIndoor']

    train_size = int(opt.train_size * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(not opt.no_shuffle))
    val_loader = DataLoader(test_dataset, batch_size=opt.val_batch_size)

    
    images, heatmaps = next(iter(train_loader))
    print('Loss using zeros: ', loss_function(torch.zeros_like(heatmaps), heatmaps), '\n')
    writer.add_graph(model, images.to(device))

    if opt.single_batch_overfit:
        print('Overfitting on a single batch.')
        training_loop(opt, device, model, writer, loss_function, optimizer, [(images, heatmaps)], None)

    else:
        print("Starting training")
        training_loop(opt, device, model, writer, loss_function, optimizer, train_loader, val_loader)