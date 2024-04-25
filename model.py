import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
reduce by a factor of 2 in every layer, until the spatial resolution is 1x1 
channels gets doubled in every layer starting with 4 channels in the first layer
"""
class Encoder(nn.Module):
    def __init__(self, image_res=64):
        super(Encoder, self).__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)] # input: 64x64x1, output: 32x32x4
        )
        self.convs.append(nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)) # input: 32x32x4, output: 16x16x8
        self.convs.append(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1))  # input: 16x16x8, output: 8x8x16
        self.convs.append(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))  # input: 8x8x16, output: 4x4x32
        self.convs.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)) # input: 4x4x32, output: 2x2x62
        self.convs.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)) # input: 2x2x64, output: 1x1x128

        self.fc = nn.Linear(in_features=128, out_features=64) #input: 1x1x128, output:64

    # x is the observation at one time step
    def forward(self, x, detach=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        # if 3 dim reshaped to add a dimension for the batch
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # if 2 dim reshaped to add a dimension for the batch and channel at the beginning of the 
        if len(x.shape) == 2:
            x = x[None, None, :]
        for i in range(len(self.convs)):
            x = torch.relu(self.convs[i](x))
        out = self.fc(x.squeeze())
        # freeze the encoder
        #if detach:
        #    out = out.detach()
        return out


# add a detached decoder network and train the model on a single reward
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tfc = nn.Linear(64, 128)

        # self.tconvs = nn.ModuleList(
        #     [nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2, output_padding=1)]
        # )

        # self.tconvs.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.tconvs.append(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.tconvs.append(nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.tconvs.append(nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.tconvs.append(nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1))


        # transpose fully connected layer
        self.tconvs = nn.ModuleList(
            [nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)]
            )
        self.tconvs.append(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)) # input: 2x2x62, output: 4x4x32
        self.tconvs.append(nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2))  # input: 4x4x32, output: 8x8x16
        self.tconvs.append(nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2))  # input: 8x8x16, output: 16x16x8
        self.tconvs.append(nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)) # input: 16x16x8, output: 32x32x4
        self.tconvs.append(nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2))

        
    def forward(self, x):  
        x = self.tfc(x).view(-1, 128, 1, 1)
        for i in range(len(self.tconvs)):
            x = torch.relu(self.tconvs[i](x))
        # output: (1, 1, 64, 64)
        return x

# 3 layers with 32 hidden units
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=32):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_size)
        self.relu = nn.ReLU
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = self.fc1(x)
        hidden_layer = torch.relu(x)
        output_layer = self.fc3(hidden_layer)
        return output_layer

# Build a 2-layer feedforward neural network
class MLP_2(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=32, is_actor=True):
        super(MLP_2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)
        self.is_actor = is_actor
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = 'cpu'

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        x = x.to(self.device)
        hidden_layer = self.relu(self.fc1(x))
        if self.is_actor:
            # [-1, 1]
            output_layer = self.tanh(self.fc2(hidden_layer))
        else:
            # is_critic: [0, inf)
            output_layer = self.relu(self.fc2(hidden_layer))
        return output_layer

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Architecture 1
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)])
        # self.convs.append(nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1))
        # self.convs.append(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1))

        # Architecture 2
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)])
        # self.convs.append(nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1))
        # self.convs.append(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1))

        # self.fc = nn.Linear(in_features=16*8*8, out_features=64)

        # Architecture 3
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 32, kernel_size=3, stride=2)])  # 31
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))  # 29
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))  # 27

        # self.fc = nn.Linear(in_features=32*27*27, out_features=64)

    # x is the observation at one time step
    def forward(self, x, detach=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if len(x.shape) == 2:
            x = x[None, None, :]
        for i in range(3):
            x = torch.relu(self.convs[i](x))

        # freeze
        if detach:
            x.detach()
        return x.view(-1, 32*27*27)
        # return x.flatten()


class CNN_MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=64):
        super(CNN_MLP, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 32, kernel_size=3, stride=2)])  # 31
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))  # 29
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))  # 27

        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if len(x.shape) == 2:
            x = x[None, None, :]
        for i in range(3):
            x = torch.relu(self.convs[i](x))

        hidden_layer = self.relu(self.fc1(x.view(-1, 32*27*27)))
        output_layer = self.fc2(hidden_layer)
        # output_layer = self.fc3(hidden_layer)
        return output_layer

# single-layer LSTM
class LSTM(nn.Module):
    def __init__(self, sequence_length, input_size=64, hidden_size=64, num_layers=1, output_size=64):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x): # (64,)
        # hidden states
        h0 = x[None, None, :]
        input = torch.zeros(self.num_layers, self.sequence_length, self.hidden_size) # (1, sequence_length, 64)
        # initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        # forward propagate LSTM
        out, _ = self.lstm(input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out.squeeze(0))
        return out    

# reward-induced representation learning 
class Model(nn.Module):
    def __init__(self, time_steps, frames, tasks, image_resolution, device):
        super(Model, self).__init__()
        self.encoder = Encoder().to(device)
        self.mlp = MLP(input_size=image_resolution*frames, output_size=image_resolution).to(device)
        self.lstm = LSTM(sequence_length=time_steps)
        self.time_steps = time_steps
        self.tasks = tasks
        self.frames = frames
        self.image_resolution = image_resolution
        self.device = device
        self.loss = nn.MSELoss()
        self.decoder = Decoder().to(device)
        
        reward_head_ax = MLP(input_size=image_resolution, output_size=1).to(device)
        reward_head_ay = MLP(input_size=image_resolution, output_size=1).to(device)
        reward_head_tx = MLP(input_size=image_resolution, output_size=1).to(device)
        reward_head_ty = MLP(input_size=image_resolution, output_size=1).to(device)
        self.reward_heads = list((
            reward_head_ax, reward_head_ay, reward_head_tx, reward_head_ty))
        
    def forward(self, obs):
        z = []
        # get the representation, z_t, at each time step
        for frame in range(self.frames):
            z_frame = self.encoder(obs[frame][None, None, :]) #tensor(64,)
            z.append(z_frame)
        z = torch.stack(z, dim=0) # tensor(len(self.frames),64)
        z_mlp = self.mlp(torch.flatten(z)) # (64,)
        h = self.lstm(z_mlp) # LSTM output (h_hat), (time_steps, 64)

        # then feed h to each reward head to predict the reward of all time_steps for every task
        reward_predicted_tasks = []
        for task in range(self.tasks):
            # reward_heads is a list of self.tasks MLP's
            reward_head = self.reward_heads[task] #LSTM outputs are mapped to each reward head  
            reward_predicted = []
            for t in range(self.time_steps):
                r_t = reward_head(h[t])   
                reward_predicted.append(r_t) 
            reward_predicted = torch.stack(reward_predicted, dim=0).squeeze() # (time_steps,) 
            reward_predicted_tasks.append(reward_predicted)
        reward_predicted_tasks = torch.stack(reward_predicted_tasks, dim=0) # should be (self.tasks, self.time_steps)
        return reward_predicted_tasks
    
    def criterion(self, reward_predicted, reward_targets):
        reward_predicted = reward_predicted.squeeze()
        assert reward_predicted.shape == reward_targets.shape
        return self.loss(reward_predicted, reward_targets)
    
    def test_decode(self, traj_images):
        output = []
        for t in range(self.time_steps):
            # the input to encoder should be in (1, 1, 64, 64)
            z_t = self.encoder(traj_images[t][None, None, :], detach=True) #tensor(64,)
            decoded_img = self.decoder(z_t).squeeze()
            output.append(decoded_img.detach().numpy())

        return np.array(output)
    