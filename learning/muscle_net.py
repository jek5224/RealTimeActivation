import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class MuscleNN(nn.Module):
    def __init__(
        self, 
        num_total_muscle_related_dofs, 
        num_dofs, 
        num_muscles, 
        device = "cuda", 
        config={"sizes" : [256,256,256], "learningStd" : False}# [256,256,256]
    ):
        super(MuscleNN, self).__init__()

        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs  
        self.num_muscles = num_muscles
        self.config = config

        layers = []
        prev_size = num_total_muscle_related_dofs + num_dofs
        for size in self.config["sizes"]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_size = size
        layers.append(nn.Linear(prev_size, num_muscles))
        
        self.fc = nn.Sequential(*layers)

        # Normalization
        self.std_muscle_tau = torch.ones(num_total_muscle_related_dofs) * 200
        self.std_tau = torch.ones(num_dofs) * 200
        
        if self.config["learningStd"]:
            self.std_muscle_tau = nn.Parameter(self.std_muscle_tau)
            self.std_tau = nn.Parameter(self.std_tau)

        if torch.cuda.is_available() and device=="cuda":
            self.cuda()
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.fc.apply(weights_init) ## initialize 

    def forward(self, reduced_JtA, tau) -> torch.Tensor:
        reduced_JtA = reduced_JtA / self.std_muscle_tau
        tau = tau / self.std_tau


        return torch.relu(torch.tanh(self.fc(torch.cat([reduced_JtA, tau], dim=-1))))

    def get_activation(self, reduced_JtA, tau_des) -> np.array:   
        if not isinstance(reduced_JtA, torch.Tensor):
            reduced_JtA = torch.tensor(reduced_JtA, device=self.fc[0].weight.device, dtype=torch.float32)
        if not isinstance(tau_des, torch.Tensor):
            tau_des = torch.tensor(tau_des, device=self.fc[0].weight.device, dtype=torch.float32) 
        with torch.no_grad():
            return self.forward(reduced_JtA, tau_des).cpu().detach().numpy()


## Supervised learner for muscle model
class MuscleLearner:
    def __init__(
        self,
        num_tau_des,
        num_muscles,
        num_reduced_JtA,
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
        
        self.num_tau_des = num_tau_des
        self.num_muscles = num_muscles
        self.num_reduced_JtA = num_reduced_JtA
        self.num_epochs_muscle = num_epochs
        self.muscle_batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = MuscleNN(self.num_reduced_JtA, self.num_tau_des, self.num_muscles, device = self.device, config={"sizes":sizes, "learningStd":learningStd})
        
        if model_weight:
            self.model.load_state_dict(model_weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for param in self.model.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}
        self.model.train()
    
    def train(self, reduced_JtA, net_tau_des, full_JtA):
        idx_all = np.asarray(range(len(reduced_JtA)))
        stats = {}
        stats["loss"] = {}
        
        for iter in range(self.num_epochs_muscle):
            ## shuffle
            np.random.shuffle(idx_all)

            ## Efficient shuffle 
            for i in range(len(reduced_JtA) // self.muscle_batch_size):
                mini_batch_idx = torch.from_numpy(idx_all[i * self.muscle_batch_size : (i + 1) * self.muscle_batch_size]).cuda()
                batch_reduced_JtA = torch.index_select(reduced_JtA, 0, mini_batch_idx)
                batch_net_tau_des = torch.index_select(net_tau_des, 0, mini_batch_idx)
                batch_full_JtA = torch.index_select(full_JtA, 0, mini_batch_idx)

                self.optimizer.zero_grad()
                a_pred = self.model.forward(batch_reduced_JtA, batch_net_tau_des)
                
                # MSE loss
                mse_loss = ((batch_net_tau_des - (a_pred.unsqueeze(1)@batch_full_JtA.transpose(1,2)).squeeze(1))/100.0).pow(2).mean()
                reg_loss = a_pred.pow(2).mean()
                
                loss = mse_loss + 0.01 * reg_loss

                if iter == self.num_epochs_muscle - 1 and i == (len(reduced_JtA) // self.muscle_batch_size - 1):
                    stats["loss"]["mse_loss"] = mse_loss.item()
                    stats["loss"]["reg_loss"] = reg_loss.item()
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

