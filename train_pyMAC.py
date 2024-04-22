import argparse
import os
from pathlib import Path
from learning.ray_model import SimulationNN_Ray
from core.env import Env as MyEnv

import ray
from ray import tune
from learning.ray_ppo import CustomPPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
import pickle
import torch
import time
torch, nn = try_import_torch()
import numpy as np

# from learning.ray_model import MuscleNN
# from learning.ray_model import SpdNN
from learning.ray_model import PolicyNN
from learning.ray_model import loading_network

from learning.muscle_net import MuscleLearner
from learning.marginal_net import MarginalLearner
from learning.spd_net import SpdLearner

def create_my_trainer(rl_algorithm: str):
    if rl_algorithm == "PPO":
        RLTrainer = CustomPPOTrainer
    else:
        raise RuntimeError(f"Invalid algorithm {rl_algorithm}!")

    class MyTrainer(RLTrainer):
        def setup(self, config):
            self.env_str = config.pop("env_str")
            self.num_bvhs = config.pop("num_bvhs")

            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )      
                
            self.policy_nn_config = config.pop("policyNN") 
            self.policy_nn_config["sizes"] = config.pop("sizes")
            self.policy_nn_config["learningStd"] = config.pop("learningStd")
            self.policy_nn_config.update(config["model"]["custom_model_config"])
            
            self.actuator_type = config.pop("actuator_type")
            self.trainer_config = config.pop("trainer_config")

            if self.actuator_type.find("mass") != -1:
                self.muscle_nn_config = config.pop("muscleNN") 
                self.indices = torch.tensor(self.muscle_nn_config["muscle_idxs"], device=self.device)
            RLTrainer.setup(self, config=config)       
            self.remote_workers = self.workers.remote_workers()
            
            ## Spd Learning Setting 
            self.spd_learner = None
            if self.trainer_config["learningSpd"]:
                self.spd_learner = SpdLearner(
                    self.trainer_config["spd"]["num_states"],
                    self.policy_nn_config["num_actions"],
                    self.policy_nn_config["num_actions"],
                    learning_rate=self.trainer_config["spd"]["lr"],
                    num_epochs=self.trainer_config["spd"]["num_epochs"],
                    batch_size=self.trainer_config["spd"]["sgd_minibatch_size"],
                    sizes=self.trainer_config["spd"]["sizes"],
                )
                for worker in self.remote_workers:
                    worker.foreach_env.remote(lambda env: env.set_spd_learning(True))
            
            ## MarinalNN Learning Setting
            self.marginal_learner = None
            if self.trainer_config["learningMarginal"]:
                self.marginal_learner = MarginalLearner(
                    self.trainer_config["marginal"]["input_dim"],
                    self.trainer_config["marginal"]["output_dim"],
                    learning_rate=self.trainer_config["marginal"]["lr"],
                    num_epochs=self.trainer_config["marginal"]["num_epochs"],
                    batch_size=self.trainer_config["marginal"]["sgd_minibatch_size"],
                    model_weight = None,
                    sizes=self.trainer_config["marginal"]["sizes"],
                )
                for worker in self.remote_workers:
                    worker.foreach_env.remote(lambda env: env.set_marginal_learning(True))

            ## Initialize 2-Level Muscle Learner
            if self.actuator_type.find("mass") != -1:
                self.muscle_learner = MuscleLearner(
                    self.muscle_nn_config["num_tau_des"],
                    self.muscle_nn_config["num_muscles"],
                    self.muscle_nn_config["num_reduced_JtA"],
                    learning_rate=self.trainer_config["muscle"]["lr"],
                    num_epochs=self.trainer_config["muscle"]["num_epochs"],
                    batch_size=self.trainer_config["muscle"]["sgd_minibatch_size"],
                    sizes=self.trainer_config["muscle"]["sizes"],
                    learningStd=self.trainer_config["muscle"]["learningStd"],
                    )
            
                model_weights = ray.put(self.muscle_learner.get_model_weights(device=torch.device("cpu")))
                muscle_nn_config = {"sizes":self.trainer_config["muscle"]["sizes"], "learningStd":self.trainer_config["muscle"]["learningStd"]}
                for worker in self.remote_workers:
                    worker.foreach_env.remote(lambda env: env.set_muscle_network(muscle_nn_config))
                    worker.foreach_env.remote(lambda env: env.load_muscle_model_weight(model_weights))
            self.max_reward = -float("inf")
            self.idx = 0

        def step(self):
            result = RLTrainer.step(self)
            current_reward = result["episode_reward_mean"]
            result["sampler_results"].pop("hist_stats")
            result["loss"] = {}

            ## Update Policy Again
            ## Function set_weights/get_weights 
            if self.spd_learner:
                start = time.perf_counter()
                spd_transitions = [None, None, None] 
                for idx in range(len(spd_transitions)):
                    spdts = np.array(ray.get([worker.foreach_env.remote(lambda env: env.get_spd_tuples(idx)) for worker in self.remote_workers]), dtype=np.float32)
                    spd_transitions[idx] = torch.tensor(spdts.reshape(-1, spdts.shape[-1]), device="cuda")
                loading_time = (time.perf_counter() - start) * 1000

                learning_time = time.perf_counter()
                stats = self.spd_learner.train(spd_transitions[0], spd_transitions[1], spd_transitions[2])
                learning_time = (time.perf_counter() - learning_time) * 1000
                total_time = (time.perf_counter() - start) * 1000

                result["timers"]["spd_learning"] = {"learning_time_ms" : learning_time, "loading_time_ms" : loading_time, "total_ms" : total_time}
                result["loss"]["spd"] = stats["loss"]

            if self.marginal_learner:
                start = time.perf_counter()
                marginal_transitions = [None, None]
                for idx in range(len(marginal_transitions)):
                    mts = np.array(ray.get([worker.foreach_env.remote(lambda env: env.get_marginal_tuples(idx)) for worker in self.remote_workers]), dtype=np.float32)
                    marginal_transitions[idx] = torch.tensor(mts.reshape(-1, mts.shape[-1]), device="cuda")
                loading_time = (time.perf_counter() - start) * 1000
                
                converting_time = time.perf_counter()
                value_transitions = self.get_policy().model.custom_get_value(marginal_transitions[0])
                converting_time = (time.perf_counter() - converting_time) * 1000

                learning_time = time.perf_counter()
                stats = self.marginal_learner.train(marginal_transitions[1], value_transitions)
                learning_time = (time.perf_counter() - learning_time) * 1000
                total_time = (time.perf_counter() - start) * 1000

                result["timers"]["marginal_learning"] = {"learning_time_ms" : learning_time, "loading_time_ms" : loading_time, "converting_time_ms" : converting_time, "total_ms" : total_time}
                result["loss"]["marginal"] = stats["loss"]

                ## Update Env's sampled prob distribution
                dist_resolution = 100
                scaling_weight = 4
                input = np.ones((self.num_bvhs, dist_resolution, 2), dtype=np.float32)   
                input[:, :, 1] = np.array(range(dist_resolution)) * 1.0 / dist_resolution
                for i in range(self.num_bvhs):
                    input[i, :, 0] = i
                phases = torch.tensor(input.reshape(-1, input.shape[-1]), device="cuda")
                marginalized_values = self.marginal_learner.model.get_value(phases)[:].transpose()
                sampling_prob_values = marginalized_values - np.min(marginalized_values)
                sampling_prob_values = np.max(sampling_prob_values) - sampling_prob_values
                sampling_prob_values = sampling_prob_values ** scaling_weight
                sampling_prob_values = ray.put((sampling_prob_values / np.sum(sampling_prob_values))[0])
                for worker in self.remote_workers:
                    worker.foreach_env.remote(lambda env: env.set_sampling_prob_values(sampling_prob_values))            

            if self.actuator_type.find("mass") != -1:
                start = time.perf_counter()
                # mts = []
                # muscle_transitions = []
                muscle_transitions = [None, None]
                ## Collect Muscle Tuples 
                for idx in range(len(muscle_transitions)):
                    mts = np.array(ray.get([worker.foreach_env.remote(lambda env: env.get_muscle_tuples(idx)) for worker in self.remote_workers]), dtype=np.float32)
                    muscle_transitions[idx] = torch.tensor(mts.reshape(-1, mts.shape[-1]), device="cuda")                                  
                loading_time = (time.perf_counter() - start) * 1000
  
                ## Making L . shape : (batch, num_tau_dest, num_muscles)
                converting_time = time.perf_counter()
                muscle_full_JtA = torch.zeros(muscle_transitions[1].shape[0], self.muscle_nn_config["num_tau_des"], self.muscle_nn_config["num_muscles"], device=self.device)
                muscle_full_JtA[:, self.indices[:,1], self.indices[:,0]] = muscle_transitions[0]
                converting_time = (time.perf_counter() - converting_time) * 1000

                learning_time = time.perf_counter()
                stats = self.muscle_learner.train(muscle_transitions[0], muscle_transitions[1], muscle_full_JtA)
                learning_time = (time.perf_counter() - learning_time) * 1000
                distribute_time = time.perf_counter()
                model_weights = ray.put(
                    self.muscle_learner.get_model_weights(device=torch.device("cpu"))
                )
                for worker in self.remote_workers:
                    worker.foreach_env.remote(
                        lambda env: env.load_muscle_model_weight(model_weights)
                    )

                distribute_time = (time.perf_counter() - distribute_time) * 1000
                total_time = (time.perf_counter() - start) * 1000

                result["timers"]["muscle_learning"] = {"distribution_time_ms" : distribute_time, "learning_time_ms" : learning_time, "loading_time_ms" : loading_time, "converting_time_ms" : converting_time, "total_ms" : total_time}
                result["loss"]["muscle"] = stats["loss"]

            if self.max_reward < current_reward:
                self.max_reward = current_reward
                self.save_checkpoint(self._logdir, "max")
            self.save_checkpoint(self._logdir, "last")
            self.idx += 1

            return result

        def __getstate__(self):
            state = RLTrainer.__getstate__(self)
            state['env_str'] = self.env_str
            state['policyNN'] = self.policy_nn_config
    
            if self.actuator_type.find("mass") != -1:
                state["muscle"] = self.muscle_nn_config
                state["muscle"]["weights"] = self.muscle_learner.get_model_weights(torch.device("cpu"))
                state["muscle"]["sizes"] = self.muscle_learner.model.config["sizes"]
                state["muscle"]["learningStd"] = self.muscle_learner.model.config["learningStd"]
            
            if self.spd_learner:
                state["spd"] = self.trainer_config["spd"]
                state["spd"]["weights"] = self.spd_learner.get_model_weights(torch.device("cpu"))
                state["spd"]["sizes"] = self.spd_learner.model.config["sizes"]
            
            if self.marginal_learner:
                state["marginal"] = self.trainer_config["marginal"]
                state["marginal"]["weights"] = self.marginal_learner.get_model_weights(torch.device("cpu"))
                state["marginal"]["sizes"] = self.marginal_learner.model.config["sizes"]
    
            return state

        def __setstate__(self, state):
            RLTrainer.__setstate__(self, state)
            if self.actuator_type.find("mass") != -1:
                self.muscle_nn_config = state["muscle"]
                self.muscle_learner = MuscleLearner(
                    self.muscle_nn_config["num_tau_des"],
                    self.muscle_nn_config["num_muscles"],
                    self.muscle_nn_config["num_reduced_JtA"],
                    learning_rate=self.trainer_config["muscle"]["lr"],
                    num_epochs=self.trainer_config["muscle"]["num_epochs"],
                    batch_size=self.trainer_config["muscle"]["sgd_minibatch_size"],
                    model_weight=state["muscle"]["weights"],
                    sizes=state["muscle"]["sizes"],
                    learningStd=state["muscle"]["learningStd"],
                )

        def save_checkpoint(self, checkpoint_path, str=None):
            if str == None:
                print(f"Saving checkpoint at path {checkpoint_path}")
                RLTrainer.save_checkpoint(self, checkpoint_path)
            else:
                with open(Path(checkpoint_path) / f"{str}_checkpoint", "wb") as f:
                    pickle.dump(self.__getstate__(), f)
            return checkpoint_path

        def load_checkpoint(self, checkpoint_path):
            print(f"Loading checkpoint at path {checkpoint_path}")
            checkpoint_file = list(Path(checkpoint_path).glob("checkpoint-*"))
            if len(checkpoint_file) == 0:
                raise RuntimeError("Missing checkpoint file!")
            RLTrainer.load_checkpoint(self, checkpoint_file[0])

    return MyTrainer


def get_config_from_file(filename: str, config: str):
    exec(open(filename).read(), globals())
    config = CONFIG[config]
    return config


parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true")
parser.add_argument("--config", type=str, default="ppo_mini")
parser.add_argument("--config-file", type=str, default="learning/ray_config.py")
parser.add_argument("-n", "--name", type=str)
parser.add_argument("--env", type=str, default="data/env.xml")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--initialNN", type=str, default=None)

if __name__ == "__main__":
    env_path = None
    args = parser.parse_args()
    print("Argument : ", args)

    env_xml = Path(args.env).resolve()
    checkpoint_path = args.checkpoint

    # read all text from the file
    env_str = None
    with open(env_xml, "r") as file:
        env_str = file.read()

    if args.cluster:
        ray.init(address=os.environ["ip_head"])
    else:
        if "node" in args.config:
            ray.init(num_cpus=128)
        else:
            ray.init()

    print("Nodes in the Ray cluster:")
    print(ray.nodes())

    config = get_config_from_file(args.config_file, args.config)
    ModelCatalog.register_custom_model("MyModel", SimulationNN_Ray)

    register_env("MyEnv", lambda config: MyEnv(env_str))
    print(f"Loading config {args.config} from config file {args.config_file}.")

    config["rollout_fragment_length"] = config["train_batch_size"] / (config["num_workers"] * config["num_envs_per_worker"])
    config["env_str"] = env_str

    if args.initialNN:
        initialNN, _, _, _, _ = loading_network(args.initialNN)
        config['model']['custom_model_config']['initialNN'] = initialNN.policy.cpu().state_dict()
        # config["initialNN"] = initialNN.policy.cpu().state_dict()

    with MyEnv(env_str) as env:
        config["actuator_type"] = env.actuator_type
        config["policyNN"] = {"num_obs" : env.num_obs, "num_actions" : env.num_action}
        config["num_bvhs"] = len(env.bvhs)
        if config["actuator_type"].find("mass") != -1:
            config["muscleNN"] = {}
            config["muscleNN"]["num_tau_des"] = len(env.get_zero_action())
            config["muscleNN"]["num_muscles"] = env.muscles.getNumMuscles()
            config["muscleNN"]["num_reduced_JtA"] = env.muscles.getNumMuscleRelatedDofs()
            config["muscleNN"]["muscle_idxs"] = env.muscle_idxs

    local_dir = "./ray_results"
    algorithm = config["trainer_config"]["algorithm"]
    MyTrainer = create_my_trainer(algorithm)

    from ray.tune import CLIReporter

    tune.run(
        MyTrainer,
        name=args.name,
        config=config,
        local_dir=local_dir,
        restore=checkpoint_path,
        progress_reporter=CLIReporter(max_report_frequency=25),
        checkpoint_freq=200,
    )

    ray.shutdown()
