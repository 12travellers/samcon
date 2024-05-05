from isaacgym import gymtorch
from isaacgym import gymapi
import isaacgym as gym
import torch
import numpy as np
from ref_motion import compute_motion


class Simulation:
    def __init__(self, sim, asset, skeleton, idk):
        self.sim = sim
        self.idk = idk
        self.skeleton = skeleton
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = gym.create_env(sim, lower, upper, 8)
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(0.89))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.actor_handle = gym.create_actor(self.env, asset, start_pose)
        
        self.state = None
        self.trajectory = []

        
    def overlap(self, old_state, old_traj):

        self.state = old_state
        self.trajectory = old_traj   
        root_tensor, joint_tensor = old_state[0], old_state[1]
        
        actor_ids = [self.actor_handle]
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(root_tensor),
            actor_ids, n_actor_ids
        )
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(joint_tensor),
            actor_ids, n_actor_ids
        ) 
    
    def cost(self, joint_p, joint_q):
        
        len = self.state.numel()
        now_state = gym.acquire_dof_state_tensor(self.sim)[self.idk*len:self.idk*len+len]
        now_state = gymtorch.wrap_tensor(now_state)
        target_motion = compute_motion(30, self.skeleton, joint_q, joint_p)
        now_motion = self.compute_motion_from_state(now_state, joint_p, joint_q)
        return self.compute_total_cost(target_motion, now_motion)
        
    def act(self, target_state, record=0):
        if record:
            self.trajectory += [target_state]
        
        root_tensor, joint_tensor = target_state[0], target_state[1]        
        actor_ids = [self.actor_handle]
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        gym.set_dof_position_target_tensor_indexed(self.sim, joint_tensor, actor_ids, n_actor_ids)
        
        
    def history(self):
        return [self.state, self.trajectory]
        
        
        
    def compute_motion_from_state(self, state, candidate_p, candidate_q):
        skeleton = self.skeleton
        p = candidate_p.clone()
        q = candidate_q.clone()
        state = state.reshape(-1, 2)
        dofs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 16, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 31, 33, 34, 35]
        for i in range(0,len(dofs)):
            q [0,dofs[i]] = state[i][1]
            
        _root_tensor = gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        root_positions = root_tensor[:, 0:3]
        root_orientations = root_tensor[:, 3:7]
        
        for nid in range(len(skeleton.nodes)):
            pid = skeleton.parents[nid]
            if pid == -1:
                p[:, nid] = root_positions
                q[:, nid] = root_orientations
            else:
                p[:, nid] *= 0
            
        return compute_motion(30, self.skeleton, q, p)
        
        
        
        
        
        
        
        
        
        
        
        
    def compute_total_cost(self):
        pose_w, root_w, ee_w, balance_w, com_w = 0, 10, 60, 30, 10
        pose_cost = self.compute_pose_cost()
        root_cost = self.compute_root_cost()
        ee_cost = self.compute_ee_cost()
        balance_cost = self.compute_balance_cost()
        com_cost = self.compute_com_cost()
        total_cost = pose_w * pose_cost + \
                     root_w * root_cost + \
                     ee_w * ee_cost + \
                     balance_w * balance_cost + \
                     com_w * com_cost
        return total_cost, pose_cost, root_cost, ee_cost, balance_cost, com_cost

    def compute_pose_cost(self):
        """ pose + angular velocity of internal joints in local coordinate """
        error = 0.0
        joint_weights = self._cfg.joint_weights
        sim_joint_ps, sim_joint_vs = self._sim_agent.get_joint_pv(self._sim_agent._joint_indices_movable)
        kin_joint_ps, kin_joint_vs = self._kin_agent.get_joint_pv(self._kin_agent._joint_indices_movable)

        for i in range(len(joint_weights)):
            _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_joint_ps[i], kin_joint_ps[i])
            )
            diff_pos_vel = sim_joint_vs[i] - kin_joint_vs[i]
            error += joint_weights[i] * (1.0 * np.dot(diff_pose_pos, diff_pose_pos) + 0.1 * np.dot(diff_pos_vel, diff_pos_vel))
        error /= len(joint_weights)            
        return error

    def compute_root_cost(self):
        """ orientation + angular velocity of root in world coordinate """
        error = 0.0
        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = self._sim_agent.get_base_pQvw()
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = self._kin_agent.get_base_pQvw()

        diff_root_p = sim_root_p - kin_root_p
        diff_root_p = diff_root_p[:2] # only consider XY-component

        _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
            self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q)
        )
        diff_root_v = sim_root_v - kin_root_v
        diff_root_w = sim_root_w - kin_root_w
        error += 1.0 * np.dot(diff_root_Q, diff_root_Q) + \
                 0.1 * np.dot(diff_root_w, diff_root_w)
        return error

    def compute_ee_cost(self):
        """ end-effectors (height) in world coordinate """
        error = 0.0
        end_effectors = self._cfg.end_effectors
        sim_ps, _, _, _ = self._sim_agent.get_link_pQvw(end_effectors)
        kin_ps, _, _, _ = self._kin_agent.get_link_pQvw(end_effectors)
        
        for sim_p, kin_p in zip(sim_ps, kin_ps):
            diff_pos = sim_p - kin_p
            diff_pos = diff_pos[-1] # only consider Z-component (height)
            error += np.dot(diff_pos, diff_pos)

        error /= len(end_effectors)
        return error

    def compute_balance_cost(self):
        """ balance cost plz see the SamCon paper """
        error = 0.0
        sim_com_pos, sim_com_vel = self._sim_agent.compute_com_pos_vel()
        kin_com_pos, kin_com_vel = self._kin_agent.compute_com_pos_vel()
        end_effectors = self._cfg.end_effectors
        sim_ps, _, _, _ = self._sim_agent.get_link_pQvw(end_effectors)
        kin_ps, _, _, _ = self._kin_agent.get_link_pQvw(end_effectors)

        for i in range(len(end_effectors)):
            sim_planar_vec = sim_com_pos - sim_ps[i]
            kin_planar_vec = kin_com_pos - kin_ps[i]
            diff_planar_vec = sim_planar_vec - kin_planar_vec
            diff_planar_vec = diff_planar_vec[:2] # only consider XY-component
            error += np.dot(diff_planar_vec, diff_planar_vec)
        error /= len(end_effectors) * self._cfg.height

        # diff_com_vel = sim_com_vel - kin_com_vel
        # error += 1.0 * np.dot(diff_com_vel, diff_com_vel)

        return error
    
    def compute_com_cost(self):
        """ CoM (position linVel) in world coordinate """
        error = 0.0
        sim_com_pos, sim_com_vel = self._sim_agent.compute_com_pos_vel()
        kin_com_pos, kin_com_vel = self._kin_agent.compute_com_pos_vel()
        diff_com_pos = sim_com_pos - kin_com_pos
        diff_com_vel = sim_com_vel - kin_com_vel
        error += 1.0 * np.dot(diff_com_pos, diff_com_pos) + \
                 0.1 * np.dot(diff_com_vel, diff_com_vel)
        return error

        