from isaacgym import gymtorch
from isaacgym import gymapi
import isaacgym as gym
import torch
import numpy as np
from ref_motion import compute_motion
from scipy.spatial.transform import Rotation as sRot

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
    
    def cost(self, joint_p, joint_q, root_pos, root_orient, root_ang_vel, old_com_pos, old_com_vel):
        self.old_root_ang_vel = root_ang_vel
        self.old_root_orient = root_orient
        self.old_com_pos, self.old_com_vel = old_com_pos, old_com_vel
        
        self.com_pos, self.com_vel = self.compute_com_pos_vel()
        
        len = self.state.numel()
        skeleton = self.skeleton
        now_state = gym.acquire_dof_state_tensor(self.sim)[self.idk*len:self.idk*len+len]
        now_state = gymtorch.wrap_tensor(now_state)
        
        for nid in range(len(skeleton.nodes)):
            pid = skeleton.parents[nid]
            if pid == -1:
                joint_p[:, nid] = root_pos
                joint_q[:, nid] = root_orient
            else:
                joint_p[:, nid] *= 0
        
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
        n_links, controllable_links = 15, [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        dofs = [3, 3, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3]
        
        t = 0
        for i in range(0,len(controllable_links)):
            dcm = None
            if dofs[i] == 3:
                dcm = state[t:t+2,1]
            else:
                dcm = [0,state[t][1],0]
            dcm = np.asarray(dcm)
            q[0,controllable_links[i]] = \
                sRot.from_euler("xyz",dcm,degrees=False).as_quat()
            t+=dofs[i]
        _root_tensor = gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        root_positions = root_tensor[:, 0:3]
        root_orientations = root_tensor[:, 3:7]
        
        self.root_orient = root_orientations
        self.root_ang_vel = root_tensor[:, 10:13]
        
        for nid in range(len(skeleton.nodes)):
            pid = skeleton.parents[nid]
            if pid == -1:
                p[:, nid] = root_positions
                q[:, nid] = root_orientations
            else:
                p[:, nid] *= 0
            
        return compute_motion(30, self.skeleton, q, p)
        
        
        
        
        
        
        
        
        
        
        
        
    def compute_total_cost(self, target_motion, new_motion):
        pose_w, root_w, ee_w, balance_w, com_w = 0, 10, 60, 30, 10
        pose_cost = self.compute_pose_cost()
        root_cost = self.compute_root_cost()
        ee_cost = self.compute_ee_cost(target_motion, new_motion)
        balance_cost = self.compute_balance_cost(target_motion, new_motion)
        com_cost = self.compute_com_cost(target_motion, new_motion)
        total_cost = pose_w * pose_cost + \
                     root_w * root_cost + \
                     ee_w * ee_cost + \
                     balance_w * balance_cost + \
                     com_w * com_cost
        return total_cost, pose_cost, root_cost, ee_cost, balance_cost, com_cost

    def compute_pose_cost(self):
        # no need
        return 0
        # """ pose + angular velocity of internal joints in local coordinate """
        # error = 0.0
        # joint_weights = self._cfg.joint_weights
        # sim_joint_ps, sim_joint_vs = self._sim_agent.get_joint_pv(self._sim_agent._joint_indices_movable)
        # kin_joint_ps, kin_joint_vs = self._kin_agent.get_joint_pv(self._kin_agent._joint_indices_movable)

        # for i in range(len(joint_weights)):
        #     _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(
        #         self._pb_client.getDifferenceQuaternion(sim_joint_ps[i], kin_joint_ps[i])
        #     )
        #     diff_pos_vel = sim_joint_vs[i] - kin_joint_vs[i]
        #     error += joint_weights[i] * (1.0 * np.dot(diff_pose_pos, diff_pose_pos) + 0.1 * np.dot(diff_pos_vel, diff_pos_vel))
        # error /= len(joint_weights)            
        # return error

    def compute_root_cost(self):
        # done!
        """ orientation + angular velocity of root in world coordinate """
        error = 0.0
        
        diff_root_Q = self.old_root_orient - self.root_orient
        diff_root_w = self.old_root_ang_vel - self.root_ang_vel
        error += 1.0 * np.dot(diff_root_Q, diff_root_Q) + \
                 0.1 * np.dot(diff_root_w, diff_root_w)
        return error

    def compute_ee_cost(self, target_motion, new_motion):
        # done!
        """ end-effectors (height) in world coordinate """
        error = 0.0
        skeleton = self.skeleton
        not_ee = []
        for nid in range(len(skeleton.nodes)):
            pid = skeleton.parents[nid]
            not_ee.append(pid)
        
        ees = []
        for nid in range(len(skeleton.nodes)):
            if nid not in not_ee:
                diff_pos = target_motion.pos[0,nid] - new_motion.pos[0,nid]
                diff_pos = diff_pos[-1] # only consider Z-component (height)
                error += np.dot(diff_pos, diff_pos)
                ees += [nid]
        self.ees = ees
        error /= len(ees)
        return error

    def compute_com_pos_vel(self, _pos, _vel):
        properties = gym.get_actor_rigid_body_properties(self.env, self.actor_handle)
        com_pos = torch.tensor([0,0,0])
        com_vel = torch.tensor([0,0,0])
        for i in range(len(properties)):
            com_pos += properties[i].mass * _pos[i]
            com_vel += properties[i].mass * _vel[i]
        return com_pos, com_vel
    
    def compute_com(self):
        body_state = gym.get_actor_rigid_body_states(self.env, self.actor_handle)
        return self.compute_com_pos_vel(body_state['pose']['p'], body_state['vel']['linear'])
            
    def compute_balance_cost(self, target_motion, new_motion):
        """ balance cost plz see the SamCon paper """
        error = 0.0
        sim_com_pos, sim_com_vel = self.com_pos, self.com_vel
        kin_com_pos, kin_com_vel = self.old_com_pos, self.old_com_vel
        end_effectors = self._cfg.end_effectors
        sim_ps, _, _, _ = self._sim_agent.get_link_pQvw(end_effectors)
        kin_ps, _, _, _ = self._kin_agent.get_link_pQvw(end_effectors)

        for nid in self.ees:
            sim_planar_vec = sim_com_pos - target_motion.pos[0,nid] 
            kin_planar_vec = kin_com_pos - new_motion.pos[0,nid]
            diff_planar_vec = sim_planar_vec - kin_planar_vec
            diff_planar_vec = diff_planar_vec[:2] # only consider XY-component
            error += np.dot(diff_planar_vec, diff_planar_vec)
        error /= len(self.ees) 
        return error
    
    def compute_com_cost(self):
        """ CoM (position linVel) in world coordinate """
        error = 0.0
        sim_com_pos, sim_com_vel = self.com_pos, self.com_vel
        kin_com_pos, kin_com_vel = self.old_com_pos, self.old_com_vel
        
        diff_com_pos = sim_com_pos - kin_com_pos
        diff_com_vel = sim_com_vel - kin_com_vel
        
        error += 1.0 * np.dot(diff_com_pos, diff_com_pos) + \
                 0.1 * np.dot(diff_com_vel, diff_com_vel)
        return error

        