import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

## Marginal Network which input is (motion idx and phase) and output is value of value function 
## Tuple Collection 방법 
## 1. Env 에서 모든 state 모음
## 2. Trainer 에서 Marginalization 진행
class MarginalNN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim ,
                 device, 
                 config
        ):
        super(MarginalNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        layers = []
        prev_size = input_dim
        for size in self.config["sizes"]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU(inplace=True))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_dim))
        self.fc = nn.Sequential(*layers)

        if torch.cuda.is_available() and device=="cuda":
            self.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.fc.apply(weights_init)
        
    def forward(self, x) -> torch.Tensor:
        return self.fc(x)
    
    ## For Rendering
    def get_value(self, x) -> np.ndarray:
        # x = torch.tensor(x, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            return self.fc(x).cpu().detach().numpy()

class MarginalLearner:
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            learning_rate,
            num_epochs,
            batch_size,
            model_weight,
            sizes,
            device = "cuda"
    ):
        self.device = device
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate        
        self.model = MarginalNN(input_dim, output_dim, self.device, {"sizes":sizes}).to(self.device)

        if model_weight:
            self.model.load_state_dict(model_weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for param in self.model.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}
        self.model.train()
        
    def train(self, input, output):
        idx_all = np.asarray(range(len(input)))
        stats = {}
        stats["loss"] = {}

        for iter in range(self.num_epochs):
            np.random.shuffle(idx_all)
            
            for i in range(len(input) // self.batch_size):
                mini_batch_idx = torch.from_numpy(idx_all[i * self.batch_size : (i + 1) * self.batch_size]).cuda()
                batch_input = torch.index_select(input, 0, mini_batch_idx)
                batch_output = torch.index_select(output, 0, mini_batch_idx)

                self.optimizer.zero_grad()
                pred_output = self.model.forward(batch_input)

                loss = ((pred_output - batch_output)/100.0).pow(2).mean()
                
                if iter == self.num_epochs - 1 and i  == (len(input) // self.batch_size - 1):
                    stats["loss"] = loss.item()
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