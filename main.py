import argparse
import os
import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from model import *
from sprites_env.envs import sprites
from torchvision.utils import save_image
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
from dataset import *
import copy
import rl

def train_encode(model, batch, optimizer):
    avg_loss = 0.0

    for obs, agent_x, agent_y, target_x, target_y in zip(batch['obs'], batch['agent_x'], batch['agent_y'], batch['target_x'], batch['target_y']):
        optimizer.zero_grad()
        reward_targets = torch.stack((agent_x, agent_y, target_x, target_y))
        reward_predicted = model(obs)
        loss = model.criterion(reward_predicted, reward_targets)
        avg_loss += loss
        loss.backward(retain_graph=True)
        optimizer.step()

    # avg_loss.backward(retain_graph=True)
    # optimizer.step()

    l = len(batch['obs'])
    avg_loss = avg_loss / l

    return avg_loss.item()

def train_decode(model, batch, decoder_optimizer):
    avg_decoded_loss = 0.0

    for obs in batch['obs']:
        decoder_optimizer.zero_grad()
        encoded_img = model.encoder(obs[-1][None, None, :].detach().clone())
        decoded_img = model.decoder(encoded_img).squeeze()
        decoded_loss = model.criterion(decoded_img, obs[-1])
        avg_decoded_loss += decoded_loss
        # decoded_loss.backward()
        # decoder_optimizer.step()

    avg_decoded_loss.backward()
    decoder_optimizer.step()

    l = len(batch['obs'])
    avg_decoded_loss = avg_decoded_loss / l

    return decoded_img[None, :], avg_decoded_loss.item()

# argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--image_resolution', type=int, default=64)
    parser.add_argument('--time_steps', type=int, default=5)
    parser.add_argument('--tasks', type=int, default=4)
    parser.add_argument('--conditioning_frames', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--env', type=str, default='Sprites-v0')
    parser.add_argument('--reward', type=str, default='follow')
    parser.add_argument('--dataset_length', type=int, default=200)   
    parser.add_argument('--total_timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--rl_lr', type=float, default=1e-3)
    args = parser.parse_args()
    return args

def main():
    
    # parse arguments
    args = parse_args()
    

    f = args.conditioning_frames
    t = args.time_steps
    assert t > f
    

    log_dir = 'rl_runs/num_epochs=' + str(args.num_epochs) + 'env=' + args.env + '_lr=' + str(args.rl_lr) + ' ||' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if not(os.path.exists(log_dir)):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make(args.env)
    '''
    dl, traj_images, ground_truth = dataloader(args.image_resolution, t, args.batch_size, f, args.reward, args.dataset_length)
    traj_images = traj_images.to(device)
    model = Model(t, f+1, args.tasks, args.image_resolution, device).to(device)
    #print(model.parameters)
    #print(model.decoder.tconvs)
    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    decoder_optimizer = torch.optim.Adam(list(model.decoder.parameters()), lr=args.learning_rate)
    train_loss = []
    train_decoded_loss = []

    # train the encoder and decoder seperately
    for epoch in range(args.num_epochs-1):
        running_loss = 0.0
        num_batch = 0
        for batch in dl:
            loss = train_encode(model, batch, optimizer)
            running_loss += loss
            num_batch += 1

        # print or store data
        running_loss = running_loss / num_batch
        print('Epoch: {} \tLoss: {:.6f}'.format(epoch, running_loss))
        train_loss.append(running_loss)

        writer.add_scalar('Loss/train', running_loss, epoch)

    for epoch in range(args.num_epochs-1):
        running_decoded_loss = 0.0
        num_batch = 0
        for batch in dl:
            decoded_img, decoded_loss = train_decode(model, batch, decoder_optimizer)
            running_decoded_loss += decoded_loss
            num_batch += 1

        # print or store data
        running_decoded_loss = running_decoded_loss / num_batch
        # print('Epoch: {} \tLoss: {:.6f}'.format(epoch, running_decoded_loss)) # added
        train_decoded_loss.append(running_decoded_loss)

        writer.add_scalar('Loss/decoded', running_decoded_loss, epoch)
        if epoch % 5 == 0:
            decoded_img = (decoded_img + 1.0) * (255./2)
            writer.add_image('decoded_epoch{}'.format(
                epoch), decoded_img.to(torch.uint8))

    # save model
    save_dir = './trained_models/'
    save_path = os.path.join(save_dir, args.env)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    torch.save(model.encoder.state_dict(), os.path.join(
        save_path, 'seed=' + str(args.seed) + ".pt"))

    # decode and generate images with respect to reward functions
    output = model.test_decode(traj_images)
    output = (output + 1.0) * (255./2)

    img = make_image_seq_strip([output[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)   
    writer.add_image('ground_truth', ground_truth)
    writer.add_image('test_decoded', img[0])

    print("---------Done--------")
    '''
    # set hyperparameters for PPO
    hyperparameters = {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 200,
        'gamma': 0.99,
        'gae_lamda': 0.95,
        'n_updates_per_iteration': 10,
        'lr': args.rl_lr,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    # Trains the RL model
    #image_res = 64
    #frames = f+1
    #mlp2 = MLP_2(input_size=image_res*frames, output_size=image_res, device=device).to(device)
    ppo = rl.PPO(MLP_2, env, writer, device, **hyperparameters) # oracle
    #trained_encoder = model.encoder    #num_epochs=70env=Sprites-v0_lr=0.001 ||20-04-2024_22-31-31
    #ppo = rl.PPO(MLP_2, env, writer, device, trained_encoder, **hyperparameters)
    #cnn = CNN().to(device)
    #pp o = PPO(MLP_2, env, writer, device, cnn, **hyperparameters)
    #ppo = PPO(CNN_MLP, env, writer, device, **hyperparameters)

    # Train the PPO model with a specified total timesteps
    ppo.learn(total_timesteps=args.total_timesteps)
    writer.flush()


# create a directory to save the results
def make_dir():
    image_dir = 'Decoded_images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

# save the reconstructed images as generated by the model
def save_decod_img(img, epoch):
    img = img.view(-1, 64, 64)
    save_image(img, './Decoded_images/epoch{}.png'.format(epoch))

if __name__ == '__main__':
    main()