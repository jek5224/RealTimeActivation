import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class SpdNN(nn.Module):
    def __init__(
        self,
        num_states = 56, # Current character's state dimension
        num_actions = 50, # Current character's action dimension (target pose)
        num_desired_torques = 50, # Current character's desired torque dimension
        device = "cuda",
        config={"sizes" : [256,256,256]}
    ):
        super(SpdNN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_desired_torques = num_desired_torques
        self.config = config
        self.std = torch.ones(num_desired_torques) * 100

        layers = []
        prev_size = num_states + num_actions
        for size in self.config["sizes"]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_size = size
        layers.append(nn.Linear(prev_size, num_desired_torques))
        self.fc = nn.Sequential(*layers)

        if torch.cuda.is_available() and device=="cuda":
            self.cuda()
            self.std = self.std.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.fc.apply(weights_init)
    
    def forward(self, state, action) -> torch.Tensor:
        return self.fc(torch.cat([state, action], dim=-1)) * self.std

    def get_torque(self, state, action) -> np.array:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.fc[0].weight.device, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.fc[0].weight.device, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(state, action).cpu().detach().numpy()


class SpdLearner:
    def __init__(
        self, 
        num_states,
        num_actions,
        num_desired_torques,
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=128,
        model_weight=None,
        device="cuda",
        sizes = [256,256,256],
        learningStd=True,
    ):
        self.device = device
        if not torch.cuda.is_available():
            self.device = "cpu"
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_desired_torques = num_desired_torques
        self.num_epochs_spd = num_epochs
        self.spd_batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = SpdNN(self.num_states, self.num_actions, self.num_desired_torques, device = self.device, config={"sizes":sizes, "learningStd":learningStd})
        
        if model_weight:
            self.model.load_state_dict(model_weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for param in self.model.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}
        self.model.train()

    def train(self, states, actions, desired_torques):
        idx_all = np.asarray(range(len(states)))
        stats = {}
        stats["loss"] = {}

        for iter in range(self.num_epochs_spd):
            ## shuffle
            np.random.shuffle(idx_all)

            ## Efficient shuffle 
            for i in range(len(states) // self.spd_batch_size):
                mini_batch_idx = torch.from_numpy(idx_all[i * self.spd_batch_size : (i + 1) * self.spd_batch_size]).cuda()
                batch_states = torch.index_select(states, 0, mini_batch_idx)
                batch_actions = torch.index_select(actions, 0, mini_batch_idx)
                batch_desired_torques = torch.index_select(desired_torques, 0, mini_batch_idx)

                self.optimizer.zero_grad()
                a_pred = self.model.forward(batch_states, batch_actions)
                
                # mse loss
                mse_loss = ((batch_desired_torques - a_pred)/100.0).pow(2).mean() ## torch.nn.functional.mse_loss(batch_desired_torques, a_pred)
                
                loss = mse_loss
                ## put loss to stats
                if iter == self.num_epochs_spd - 1 and i == (len(states) // self.spd_batch_size - 1):
                    stats["loss"]["mse_loss"] = mse_loss.item()
                    stats["loss"]["total_loss"] = loss.item()
                loss.backward()
                self.optimizer.step()

        return stats
    
    def get_model_weights(self, device=None):
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()
    
    def set_model_weights(self, weights):
        self.model.load_state_dict(weights)