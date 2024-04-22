import numpy as np
import gym
import dartpy as dart
import xml.etree.ElementTree as ET

from core.dartHelper import buildFromFile 
from core.bvhparser import MyBVH
from numpy.linalg import inv
from numba import njit

from learning.ray_model import MuscleNN
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

import ray
import os
from learning.ray_model import loading_network


## Muscle Tuple : reduced_JtA, JtP, JtA

@njit
def mat_inv(mat):
    return inv(mat)

class Env(gym.Env):
    def __init__(self, metadata):
        
        self.world = dart.simulation.World()
        self.skel = None
        self.target_skel = None
        self.ees_name = ["Head", "HandL", "HandR", "TalusL", "TalusR"]
        self.ground = None 
        self.step_counter = 0

        # Simulation Configuration
        self.simulationHz = 480
        self.controlHz = 30

        # Motion Configuration
        self.bvhs = None
        self.bvh_idx = 0
        self.bvh_phase = 0.0
        self.horizon = 300
        self.bvh_info = None
        # Sampling Configuration
        self.sampling_strategy = "uniform"
        self.minimum_sampling = 0 ## int
        self.current_sampling = -1
        self.current_initial_time = 0.0 ## initial time
        ## Marginal Configuration Related to Sampling
        self.marginal_learning = False
        self.marginal_buffer = [[],[]]
        self.marginal_nn = None
        self.sampling_prob_values = None    
        # actuator Type 
        self.actuator_type = "pd_ref_residual"

        # Muscle Configuration
        self.muscles = None
        self.muscle_pos = []
        self.muscle_control_type = "activation" # mass or activation or musclepd
        self.muscle_activation_levels = None
        self.muscle_buffer = [[],[]]
        self.muscle_nn = None
        self.muscle_vbo_idx = None
        
        # SPD Configuration 
        self.kp = 0.0
        self.kv = 0.0
        self.learning_gain = False
        self.target_frames = 1 ## Include Current
        self.spd_learning = False
        self.spd_buffer = [[], [], []]
        self.spd_nn = None
        self.use_spd_nn = False

        # Load Environment Configuration
        self.prevNN = None
        self.onlyLocal = False
        self.loading_xml(metadata) 
        
        self.world.setTimeStep(1.0 / self.simulationHz)
        self.world.setGravity([0, -9.81, 0])
        
        # Related to Target
        self.target_pos = None
        self.target_vel = None
        self.target_positions = []
        self.target_displacement = np.zeros(self.skel.getNumDofs() - self.skel.getJoint(0).getNumDofs())
        
        self.cur_obs = None
        self.cur_reward = 0.0
        
        self.cur_root_T = None
        self.cur_root_T_inv = None
        self.pd_target = None

        self.reset()
        
        self.num_obs = len(self.get_obs())
        self.num_action = len(self.get_zero_action()) * (3 if self.learning_gain else 1)
        self.observation_space = gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_obs,))
        self.action_space = gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_action,))
        
        self.metadata = metadata

        ## For Muslce Accuracy Reward
        # self.muscle_accuracy = []

    def add_bvh(self, path, firstT = None, rootScale=0.01):
        self.bvhs.append(MyBVH(path, self.bvh_info, self.skel, firstT, rootScale))
        self.bvh_names.append(path)
        self.bvh_total_num_frames += self.bvhs[-1].num_frames
        if self.sampling_strategy == "adaptive" and type(self.sampling_prob_values) == np.ndarray:
            dist_resolution = 100
            self.sampling_prob_values = 1.0 / (len(self.bvhs) * dist_resolution) * np.ones(len(self.bvhs) * dist_resolution)
            # self.sampling_prob_values = np.concatenate([self.sampling_prob_values, sampling_dist])
            # self.sampling_prob_values.append(sampling_dist)

    def loading_xml(self, metadata):
        ## XML loading
        doc = ET.ElementTree(ET.fromstring(metadata))  # ET.parse(metadata)
        root = doc.getroot()
        # bvh_info = None
        joints_pd_gain = None
        for child in root:
            if child.tag == "skeleton":
                self.skel, self.bvh_info, joints_pd_gain = buildFromFile(child.text)
                self.target_skel = self.skel.clone()
                self.world.addSkeleton(self.skel)
            elif child.tag == "ground":
                self.ground, _ , _ = buildFromFile(child.text)
                self.ground.setMobile(False)
                self.world.addSkeleton(self.ground)
            elif child.tag == "simHz":
                self.simulationHz = int(child.text)
            elif child.tag == "controlHz":
                self.controlHz = int(child.text)
            elif child.tag == "bvh":
                Tframe = None 
                rootScale = 0.01
                if "rootScale" in child.attrib.keys():
                    rootScale = float(child.attrib["rootScale"])
                if "firstT" in child.attrib.keys():
                    Tframe = 1 if(child.attrib["firstT"].upper() == "TRUE") else None   
                if child.text[-3:] == "bvh":
                    self.bvhs = [MyBVH(child.text, self.bvh_info, self.skel, Tframe, rootScale)]
                else:   
                    files = os.listdir(child.text)
                    self.bvhs = [MyBVH(child.text + "/" + f, self.bvh_info, self.skel, Tframe, rootScale) for f in files if f[-3:] == "bvh"]
                self.bvh_names = [bvh.bvh_file for bvh in self.bvhs]
                self.bvh_total_num_frames = sum([bvh.num_frames for bvh in self.bvhs])
            elif child.tag == "actionScale":
                self.action_scale = float(child.text)
            elif child.tag == "muscle":
                self.loading_muscle(child.text)
            elif child.tag == "actuator" or child.tag == "actuactor":
                self.actuator_type = child.text
            elif child.tag == "kp":
                self.kp = float(child.text) * np.ones(self.skel.getNumDofs())
            elif child.tag == "kv":
                self.kv = float(child.text) * np.ones(self.skel.getNumDofs())
            elif child.tag == "learningGain":
                self.learning_gain = True if child.text.upper() == "TRUE" else False
            elif child.tag == "frames":
                self.target_frames = int(child.text)
            elif child.tag == "horizon":
                self.horizon = int(child.text)
            elif child.tag == "minimumSampling":
                self.minimum_sampling = int(child.text)
            elif child.tag == "samplingStrategy":
                self.sampling_strategy = child.text.lower()
            elif child.tag == "prevNN":
                self.prevNN, _, _, _ , _ = loading_network(child.text, "cpu") 
            elif child.tag == "onlyLocal":
                self.onlyLocal = True if child.text.upper() == "TRUE" else False

        if type(self.kp) != np.ndarray:
            self.kp = 300.0 * np.ones(self.skel.getNumDofs()) 
        if type(self.kv) != np.ndarray:
            self.kv = 20.0 * np.ones(self.skel.getNumDofs())

        for skel_jn_i in range(self.skel.getNumJoints()):
            jn = self.skel.getJoint(skel_jn_i)
            if joints_pd_gain[skel_jn_i] != None:
                self.kp[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0) + len(joints_pd_gain[skel_jn_i][0])] = joints_pd_gain[skel_jn_i][0]
                self.kv[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0) + len(joints_pd_gain[skel_jn_i][1])] = joints_pd_gain[skel_jn_i][1]       
        self.kp[:6] = 0.0
        self.kv[:6] = 0.0
        if self.actuator_type.find("mass") != -1:
            self.muscle_nn = MuscleNN(self.muscles.getNumMuscleRelatedDofs(), len(self.get_zero_action()), self.muscles.getNumMuscles()) ## Create Dummy Default Network    
        self.root_jn = self.skel.getRootJoint()
    
        if self.muscles:
            self.muscle_idxs = np.array(self.muscles.getMuscleIdxs())
            self.muscle_idxs[:,1] -= 6

    def set_muscle_network(self, nn_config = {"sizes" : [256, 256, 256], "learningStd" : False}):
        self.muscle_nn = MuscleNN(self.muscles.getNumMuscleRelatedDofs(), len(self.get_zero_action()), self.muscles.getNumMuscles(), config = {"sizes" : nn_config["sizes"], "learningStd" : nn_config["learningStd"]})

    def loading_muscle(self, path):
        ## Open XML
        doc = ET.parse(path)
        if doc is None:
            return
        self.muscles = dart.dynamics.Muscles(self.skel)
        root = doc.getroot()
        for child in root:
            if child.tag == "Unit":
                new_waypoints = []
                for waypoint in child:
                    if waypoint.tag == "Waypoint":
                        new_waypoints.append((waypoint.attrib["body"], np.array([float(p) for p in waypoint.attrib["p"].strip().split(" ")])))
                self.muscles.addMuscle(child.attrib["name"],[float(child.attrib["f0"]), float(child.attrib["lm"]), float(child.attrib["lt"]), float(child.attrib["pen_angle"]), float(child.attrib["lmax"]), 0.0], False, new_waypoints)
        self.muscle_activation_levels = np.zeros(self.muscles.getNumMuscles())

        ## For Fast Rendering 
        self.muscle_vbo_idx = []
        mus_wps = self.muscles.getMusclePositions()
        base_idx = 0
        for mus_wp in mus_wps:
            self.muscle_vbo_idx.append(np.arange(len(mus_wp)) + base_idx)
            base_idx += len(mus_wp)

    def get_zero_action(self):
        if True:  # if using PD servo # self.actuator_type == "pd_ref_residual" or self.actuator_type == "mass":
            return np.zeros(self.skel.getNumDofs() - self.skel.getJoint(0).getNumDofs())

    def get_root_T(self, skel):
        root_y = np.array([0, 1, 0])
        root_z = skel.getRootBodyNode().getWorldTransform().rotation() @ np.array([0, 0, 1]); root_z[1] = 0.0; root_z = root_z / np.linalg.norm(root_z)
        root_x = np.cross(root_y, root_z)

        root_rot = np.array([root_x, root_y, root_z]).transpose()
        root_T = np.identity(4); root_T[:3, :3] = root_rot; root_T[:3,  3] = skel.getRootBodyNode().getWorldTransform().translation(); root_T[1,   3] = 0.0
        return root_T

    def get_obs(self):
        return self.cur_obs.copy()
        
    def update_target(self, time):
        self.target_pos, self.bvh_phase = self.bvhs[self.bvh_idx].getPose(time)
        pos_next, _ = self.bvhs[self.bvh_idx].getPose(time + 1.0 / self.controlHz)
        self.target_vel = self.skel.getPositionDifferences(pos_next, self.target_pos) * self.controlHz
        self.target_skel.setPositions(self.target_pos)
        self.target_skel.setVelocities(self.target_vel)
        self.target_positions = [self.bvhs[self.bvh_idx].getPose(time + i * 0.2)[0] for i in range(self.target_frames)]
        

    def update_obs(self):
        w_bn_ang_vel = 0.1

        ## Skeleton Information 
        self.cur_root_T = self.get_root_T(self.skel)
        self.cur_root_T_inv = mat_inv(self.cur_root_T)

        bn_lin_pos = []
        bn_6d_orientation = []
        for bn in self.skel.getBodyNodes():
            p = np.ones(4)
            p[:3] = bn.getCOM()
            bn_lin_pos.append(p)
            bn_6d_orientation.append(bn.getWorldTransform().rotation())
        
        
        bn_lin_pos = (self.cur_root_T_inv @ np.array(bn_lin_pos).transpose()).transpose()[:,:3].flatten()
        bn_lin_vel = (self.cur_root_T_inv[:3,:3] @ np.array([bn.getCOMLinearVelocity() for bn in self.skel.getBodyNodes()]).transpose()).transpose().flatten()        
        bn_6d_orientation = (self.cur_root_T_inv[:3,:3] @ np.array(bn_6d_orientation).transpose()).transpose().reshape(len(bn_6d_orientation), -1)[:,:6].flatten()
        bn_ang_vel = (self.cur_root_T_inv[:3,:3] @ np.array([w_bn_ang_vel * bn.getAngularVelocity() for bn in self.skel.getBodyNodes()]).transpose()).transpose().flatten()

        # Target        
        target_root_T = self.get_root_T(self.target_skel)
        target_root_T_inv = mat_inv(target_root_T)
        
        target_bn_pos = []
        target_6d_orientation = []

        for i in range(self.target_frames):
            self.target_skel.setPositions(self.target_positions[i])
            for bn in self.target_skel.getBodyNodes():
                p = np.ones(4)
                p[:3] = bn.getCOM()
                target_bn_pos.append(p)
                target_6d_orientation.append(bn.getWorldTransform().rotation())

        target_bn_pos = (target_root_T_inv @ np.array(target_bn_pos).transpose()).transpose()[:,:3].flatten()
        target_6d_orientation = (target_root_T_inv[:3,:3] @ np.array(target_6d_orientation).transpose()).transpose().reshape(len(target_6d_orientation), -1)[:,:6].flatten()

        self.target_skel.setPositions(self.target_pos)

        # Root Displacement

        cur_to_target = mat_inv(self.cur_root_T) @ target_root_T
        if self.onlyLocal:
            cur_to_target = cur_to_target[:3,:3].flatten()[:6] # np.concatenate([cur_to_target[:3,:3].flatten()[:6], cur_to_target[:3,3]])
        else:
            cur_to_target = np.concatenate([cur_to_target[:3,:3].flatten()[:6], cur_to_target[:3,3]])

        self.cur_obs = np.concatenate([bn_lin_pos , bn_lin_vel , bn_6d_orientation , bn_ang_vel , cur_to_target, target_bn_pos, target_6d_orientation], dtype=np.float32) 
        if self.prevNN: 
            initTarget = self.action_scale * self.prevNN.get_action(self.cur_obs)
            length = len(bn_lin_pos) + len(bn_lin_vel) + len(bn_6d_orientation) + len(bn_ang_vel)
            
            NotImplementedError("Not Implemented Yet observation for prevNN")
            # self.cur_obs = np.concatenate([initTarget, self.cur_obs])
            # self.cur_obs = np.concatenate([initTarget, self.cur_obs[:length], self.prevNN.policy.p_fc.quantizer.get_last_code().flatten()])
    def sampling_initial_state(self):
        if type(self.sampling_prob_values) == np.ndarray and self.sampling_strategy == "adaptive":
            dist_resolution = 100
            sampled_idx = np.random.choice(np.arange(len(self.sampling_prob_values)), p = self.sampling_prob_values)
            self.bvh_idx = sampled_idx // dist_resolution
            self.current_initial_time = (sampled_idx % dist_resolution) / dist_resolution * self.bvhs[self.bvh_idx].bvh_time - 1
            self.current_initial_time = max(0, self.current_initial_time)

        else: ## Uniform 
            self.bvh_idx = np.random.randint(0, len(self.bvhs))
            self.current_initial_time = (np.random.rand() % 1.0) * self.bvhs[self.bvh_idx].bvh_time
        
    def reset(self, time = None):
        # dynamics reset
        if self.current_sampling >= self.minimum_sampling \
            or self.current_sampling == -1: # First Time 
            self.current_sampling = 0
            self.sampling_initial_state()
        
        if time == None:
            time = self.current_initial_time
        self.update_target(time)

        solver = self.world.getConstraintSolver()
        solver.setCollisionDetector(dart.collision.BulletCollisionDetector())
        solver.clearLastCollisionResult()

        self.skel.setPositions(self.target_pos)
        self.skel.setVelocities(self.target_vel)

        self.skel.clearInternalForces()
        self.skel.clearExternalForces()
        self.skel.clearConstraintImpulses()

        self.world.setTime(time)
              
        self.update_obs()
        
        if self.muscles != None:
            self.muscles.update()
            self.muscle_pos = self.muscles.getMusclePositions()

        self.step_counter = 0
        self.pd_target = np.zeros(self.skel.getNumDofs())

        return self.get_obs()   

    def get_reward(self, render=False):
        r_q = 1.0
        r_dq = 1.0
        r_com = 1.0
        r_ee = 1.0
        
        # Joint angle reward
        q_diff = self.skel.getPositionDifferences(self.skel.getPositions(), self.target_pos)
        if self.onlyLocal:
            q_diff[3:6] = 0.0
        r_q = np.exp(-15.0 * np.inner(q_diff, q_diff) / len(q_diff))
        
        # Joint velocity reward
        dq_diff = self.skel.getVelocityDifferences(self.skel.getVelocities(), self.target_vel)
        if self.onlyLocal:
            dq_diff[3:6] = 0.0
        r_dq = np.exp(-0.2 * np.inner(dq_diff, dq_diff) / len(dq_diff))

        # COM reward 
        if not self.onlyLocal:
            com_diff = self.skel.getCOM() - self.target_skel.getCOM()
            r_com = np.exp(-5.0 * np.inner(com_diff, com_diff) / len(com_diff))

        # EE reward 
        ee_diff = np.concatenate([(self.skel.getBodyNode(ee).getCOM(self.skel.getRootBodyNode()) - self.target_skel.getBodyNode(ee).getCOM(self.target_skel.getRootBodyNode())) for ee in self.ees_name])
        r_ee = np.exp(-40 * np.inner(ee_diff, ee_diff) / len(ee_diff))

        w_alive = 0.01

        self.cur_reward = (w_alive + r_q * (1.0 - w_alive)) * (w_alive + r_ee * (1.0 - w_alive)) * (w_alive + r_com * (1.0 - w_alive)) * (w_alive + r_dq * (1.0 - w_alive))
        
        if not render:
            return self.cur_reward
        if render: 
            return (self.cur_reward, r_q, r_dq, r_com, r_ee)
        
    def step(self, action):
        # self.muscle_accuracy = []
        self.current_sampling += 1
        self.update_target(self.world.getTime())
        pd_target = np.zeros(self.skel.getNumDofs())
        if self.actuator_type.find("ref") != -1:
            pd_target = self.target_pos.copy()
        elif self.actuator_type.find("cur") != -1:
            pd_target = self.skel.getPositions().copy()

        if self.prevNN:
            prev_displacement = np.zeros(self.skel.getNumDofs())
            prev_displacement[6:] = self.get_obs()[:len(prev_displacement) - 6]
            pd_target = self.skel.getPositionDifferences(pd_target, -prev_displacement)
        displacement = np.zeros(self.skel.getNumDofs())
        displacement[6:] = self.action_scale * action[:len(displacement) - 6]

        kp = self.kp
        kv = self.kv
        if self.learning_gain:
            kp[6:] = self.kp[6:] + 0.01 * action[len(action)//3:2*len(action)//3] * self.kp[6:]
            kv[6:] = self.kv[6:] + 0.01 * action[2*len(action)//3:] * self.kv[6:]
        
        pd_target = self.skel.getPositionDifferences(pd_target, -displacement)
        
        self.pd_target = pd_target
        
        mt = None

        rand_idx = np.random.randint(0, int(self.simulationHz//self.controlHz))
        tau = np.zeros(self.skel.getNumDofs())
        ext = np.zeros(self.skel.getNumDofs())
        for i in range(int(self.simulationHz//self.controlHz)):
            if self.use_spd_nn and self.spd_nn:
                p = self.skel.getPositions().copy()
                root_inv_T = mat_inv(self.get_root_T(self.skel))
                p[3] = 0; p[5] = 0
                p[:6] = self.root_jn.convertToPositions(root_inv_T @ self.root_jn.convertToTransform(p[:6]).matrix())
                tau[6:] = self.spd_nn.get_torque(p, pd_target[6:])
            else:
                tau = self.skel.getSPDForce(pd_target, ext, kp, kv)
            
            if self.actuator_type.find("pd") != -1:
                self.skel.setForces(tau)
            elif self.actuator_type.find("mass") != -1:
                mt = self.muscles.getMuscleTuples()
                self.muscle_activation_levels = self.muscle_nn.get_activation(mt[0], tau[6:] - mt[1])
                self.muscles.setActivations(self.muscle_activation_levels)
                self.muscles.applyForceToBody()
                ## Temp
                # self.muscle_accuracy.append(np.linalg.norm(tau - self.skel.getExternalForces()))
            self.world.step()

            if self.muscles != None:
                self.muscles.update()

            ## Collecting Tuple for other learning
            if rand_idx == i:
                ## SPD Learning
                if self.spd_learning:
                    p = self.skel.getPositions().copy()
                    root_inv_T = mat_inv(self.get_root_T(self.skel))
                    p[3] = 0; p[5] = 0
                    p[:6] = self.root_jn.convertToPositions(root_inv_T @ self.root_jn.convertToTransform(p[:6]).matrix())
                    self.spd_buffer[0].append(p)
                    self.spd_buffer[1].append(pd_target[6:])
                    self.spd_buffer[2].append(tau[6:])
                ## Marinal Learning 
                if self.marginal_learning:
                    self.marginal_buffer[0].append(self.get_obs())
                    self.marginal_buffer[1].append((self.bvh_idx, self.bvh_phase))
                ## 2-Level Muscle Learning
                if self.actuator_type.find("mass") != -1:
                    self.muscle_buffer[0].append(mt[0]) # reduced_JtA
                    self.muscle_buffer[1].append(tau[6:] - mt[1]) # net_tau_des

        self.step_counter += 1        

        if self.muscles != None:
            self.muscle_pos = self.muscles.getMusclePositions()
        
        self.update_obs()
        self.get_reward()
        
        info = self.get_eoe_condition()

        return self.get_obs(), self.cur_reward, info["end"] != 0, info

    def get_eoe_condition(self):
        info = {}
        if self.skel.getCOM()[1] < 0.6:
            info["end"] = 1
        elif self.step_counter >= self.horizon: # elif self.world.getTime() > 10.0:
            info["end"] = 3
        else:
            info["end"] = 0
        return info

    def get_current_bvh(self):
        return self.bvhs[self.bvh_idx]

    # Learning Muscle
    def load_muscle_model_weight(self, w):
        self.muscle_nn.load_state_dict(convert_to_torch_tensor(ray.get(w)))
    def get_muscle_tuples(self, idx):
        res = np.array(self.muscle_buffer[idx])
        self.muscle_buffer[idx] = []
        return res
    
    # Learning SPD 
    def set_spd_learning(self, spd_learning):
        self.spd_learning = spd_learning
    def get_spd_tuples(self, idx):
        res = np.array(self.spd_buffer[idx])
        self.spd_buffer[idx] = []
        return res

    # Learning Marginal
    def set_marginal_learning(self, marginal_learning):
        self.marginal_learning = marginal_learning  
    def get_marginal_tuples(self, idx):
        res = np.array(self.marginal_buffer[idx])
        self.marginal_buffer[idx] = []
        return res
    def set_sampling_prob_values(self, sampling_prob_values):
        self.sampling_prob_values = ray.get(sampling_prob_values)