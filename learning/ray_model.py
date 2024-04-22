import torch
import torch.nn as nn
import numpy as np

from learning.vqvae import VAE
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

from learning.muscle_net import MuscleNN
from learning.spd_net import SpdNN
from learning.marginal_net import MarginalNN

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class SimulationNN(nn.Module):
    def __init__(self, 
                 num_states, 
                 num_actions, 
                 config, # ={"sizes" : {"policy" : [512,512,512], "value" : [512,512,512]}, "learningStd" : False, "VAE" : None, "VAE_condition_dim" : 345} ,
                 device="cuda",
        ):
        super(SimulationNN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config
        
        ## For Old Version 
        
        if "VQVAE" in self.config.keys() and self.config["VQVAE"]:
            self.config["VAE"] = "vq"
        if "VQVAE_condition_dim" in self.config.keys():
            self.config["VAE_condition_dim"] = self.config["VQVAE_condition_dim"]

        if "VAE" not in self.config.keys():
            self.config["VAE"] = None
        self.isVAE = self.config["VAE"]
        if "VAE_condition_dim" not in self.config.keys() and self.config["VAE"]:
            self.config["VAE_condition_dim"] = 345
        if "num_embeddings" not in self.config.keys():
            self.config["num_embeddings"] = 256
        if "embedding_dim" not in self.config.keys():
            self.config["embedding_dim"] = 64
        # from IPython import embed; embed()
        if not self.config["VAE"]:
            p_layers = []
            prev_size = num_states
            for size in self.config["sizes"]["policy"]:
                p_layers.append(nn.Linear(prev_size, size))
                p_layers.append(nn.ReLU(inplace=True))
                prev_size = size
            p_layers.append(nn.Linear(prev_size, num_actions))
            self.p_fc = nn.Sequential(*p_layers)
        else:
            self.p_fc = VAE(self.config["VAE"],
                            num_states, 
                            num_actions, 
                            0.25, 
                            self.config["VAE_condition_dim"],
                            self.config["sizes"]["policy"], ## Encoder Size
                            self.config["sizes"]["policy"][::-1],  ## Decoder Size
                            num_embeddings=self.config["num_embeddings"],
                            embedding_dim=self.config["embedding_dim"]
                            )

        v_layers = []
        prev_size = num_states
        for size in self.config["sizes"]["value"]:
            v_layers.append(nn.Linear(prev_size, size))
            v_layers.append(nn.ReLU(inplace=True))
            prev_size = size

        v_layers.append(nn.Linear(prev_size, 1))
        self.v_fc = nn.Sequential(*v_layers)

        self.log_std = None
        if self.config["learningStd"]:
            self.log_std = nn.Parameter(torch.ones(num_actions))
        else:
            self.log_std = torch.ones(num_actions)

        if torch.cuda.is_available() and device == "cuda":
            if not self.config["learningStd"]:
                self.log_std = self.log_std.cuda()
            self.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # initialize
        self.p_fc.apply(weights_init)
        self.v_fc.apply(weights_init)

    def forward(self, x):
        if self.config["VAE"]:
            p_out, vae_loss = self.p_fc.forward(x)
            p_out = MultiVariateNormal(p_out, self.log_std.exp())
            v_out = self.v_fc.forward(x)
            return p_out, v_out, vae_loss
        else:
            p_out = MultiVariateNormal(self.p_fc.forward(x), self.log_std.exp())
            v_out = self.v_fc.forward(x)
            return p_out, v_out, None
        

class SimulationNN_Ray(TorchModelV2, SimulationNN):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        num_states = np.prod(obs_space.shape)
        num_actions = np.prod(action_space.shape)
        SimulationNN.__init__(self, 
                              num_states, 
                              num_actions, 
                            #   {"sizes" : model_config["custom_model_config"]["sizes"], 
                            #    "learningStd" : model_config["custom_model_config"]["learningStd"],
                            #    "VAE" : model_config["custom_model_config"]["VAE"],
                            #    "VAE_condition_dim" : model_config["custom_model_config"]["VAE_condition_dim"]}
                               model_config["custom_model_config"], 
                              "cuda" if torch.cuda.is_available() else "cpu")
        
        if "initialNN" in model_config["custom_model_config"] and model_config["custom_model_config"]["initialNN"]:
            self.load_state_dict(convert_to_torch_tensor(model_config["custom_model_config"]["initialNN"]))
            self.v_fc.apply(weights_init)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, {}, "SimulationNN_Ray"
        )
        num_outputs = 2 * np.prod(action_space.shape)
        self._value = None
        self.vae_loss = None

    ## For Marginalization 
    def custom_get_value(self, obs) -> np.ndarray: # 1-d array
        with torch.no_grad():
            # if obs is not tensor
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            return self.v_fc.forward(obs)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)
        action_dist, self._value, self.vae_loss = SimulationNN.forward(self, x)
        action_tensor = torch.cat([action_dist.loc, action_dist.scale.log()], dim=1)
        return action_tensor, state

    def get_vae_loss(self):
        return self.vae_loss
    
    def value_function(self):
        return self._value.squeeze(1)


## This class is for integration of simulationNN and ray filter
class PolicyNN:
    def __init__(
        self,
        num_states,
        num_actions,
        policy_state,
        filter_state,
        device,
        config={"sizes" : {"policy" : [512,512,512], "value" : [512,512,512]}, "learningStd" : False} 
    ):
        self.device = device
        self.policy = SimulationNN(num_states=num_states, num_actions=num_actions, config=config, device=device)
        self.policy.load_state_dict(convert_to_torch_tensor(policy_state))
        self.policy.eval()
        self.filter = filter_state

    def get_action(self, obs, is_random=False) -> np.ndarray:
        with torch.no_grad():
            obs = self.filter(obs, update=False)
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)

            if self.policy.isVAE:
                p_out, _ = self.policy.p_fc.forward(obs)
                p_out = MultiVariateNormal(p_out, self.policy.log_std.exp())
                return (
                    p_out.mean.cpu().detach().numpy()
                    if not is_random
                    else p_out.sample().cpu().detach().numpy()
                )
            else:
                return (
                    self.policy.p_fc.forward(obs).cpu().detach().numpy()
                    if not is_random
                    else self.policy.forward(obs)[0].sample().cpu().detach().numpy()
                )
    def get_value(self, obs) -> np.ndarray: # 1-d array
        with torch.no_grad():
            obs = self.filter(obs, update=False)
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            return self.policy.v_fc.forward(obs).cpu().detach().numpy()
    # Deprecated
    # def vae_sampling_action(self, obs) -> np.ndarray:
    #     with torch.no_grad():
    #         obs = self.filter(obs, update=False)
    #         if not isinstance(obs, torch.Tensor):
    #             obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            
    #         condition_dim = self.policy.config["VAE_condition_dim"]
    #         if len(obs.shape) == 2:
    #             condition = obs[:, :condition_dim]
    #             quantized_idx = torch.randint(0, self.policy.p_fc.num_embeddings, (obs.shape[0], self.policy.p_fc.embedding_dim), device=self.device)
    #         else:
    #             condition = obs[:condition_dim]
    #             quantized_idx = np.random.randint(0,self.policy.p_fc.num_embeddings)  # torch.randint(0, self.policy.p_fc.num_embeddings, (1,), device=self.device)
            
    #         self.policy.p_fc.quantizer.idx = quantized_idx
    #         quantized_code = self.policy.p_fc.quantizer.embeddings.weight[quantized_idx]
            
    #         return self.policy.p_fc.decoder(torch.concat([condition, quantized_code], dim=-1)).cpu().detach().numpy()

import pickle
def loading_network(path, device="cuda") -> (SimulationNN, MuscleNN, SpdNN, MarginalNN, str):
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    mus_nn = None 
    spd_nn = None
    marginal_nn = None
    env_str = None

    state = pickle.load(open(path, "rb"))
    if 'env_str' in state.keys():
        env_str = state['env_str']
        
    worker_state = pickle.loads(state["worker"])
    policy_state = worker_state["state"]["default_policy"]["weights"]
    filter_state = worker_state["filters"]["default_policy"]
    
    policy_config = state["policyNN"]
    
    ## For previous setting
    if "VQVAE" in policy_config.keys() and policy_config["VQVAE"]:
        policy_config["VAE"] = "vq"
    if "VQVAE_condition_dim" in policy_config.keys():
        policy_config["VAE_condition_dim"] = policy_config["VQVAE_condition_dim"]

    if "sizes" not in policy_config.keys():
        policy_config["sizes"] = {"policy" : [512,512,512], "value" : [512,512,512]}
    if "learningStd" not in policy_config.keys():
        policy_config["learningStd"] = False
    if "VAE" not in policy_config.keys():
        policy_config["VAE"] = False
    if "VAE_condition_dim" not in policy_config.keys():
        policy_config["VAE_condition_dim"] = 345

    policy = PolicyNN(
        policy_config["num_obs"], 
        policy_config["num_actions"], 
        policy_state, 
        filter_state, 
        device,
        policy_config,    
    )

    if 'marginal' in state.keys():
        marginal_nn = MarginalNN(
            state["marginal"]["input_dim"],
            state["marginal"]["output_dim"],
            device,
            config={"sizes" : state["marginal"]["sizes"]}
        )
        marginal_nn.load_state_dict(state["marginal"]["weights"])

    if 'muscle' in state.keys():
        mus_nn = MuscleNN(
            state["muscle"]["num_reduced_JtA"],
            state["muscle"]["num_tau_des"],
            state["muscle"]["num_muscles"],
            config={"sizes": state["muscle"]["sizes"], "learningStd": state["muscle"]["learningStd"]}
        )
        mus_nn.load_state_dict(state["muscle"]["weights"])

    if 'spd' in state.keys():
        spd_nn = SpdNN(
            state["spd"]["num_states"],
            policy_config["num_actions"],
            policy_config["num_actions"],
            config={"sizes" : state["spd"]["sizes"]}
        )
        spd_nn.load_state_dict(state["spd"]["weights"])
    return policy, mus_nn, spd_nn, marginal_nn, env_str
