from isaacgym import gymtorch
from isaacgym import gymapi
import isaacgym
import torch
import numpy as np
from ref_motion import compute_motion
from scipy.spatial.transform import Rotation as sRot

class Simulation:
    UP_AXIS = 2 
    stiff = [
        600,600,600,50,50,50,
        200,200,200,150,
        200,200,200,150,
        300,300,300,300,200,200,200,
        300,300,300,300,200,200,200,
    ]
    ees = [14,8,11,5]
    # {'head': 2, 'left_foot': 14, 'left_hand': 8, 'left_lower_arm': 7, 
    #  'left_shin': 13, 'left_thigh': 12, 'left_upper_arm': 6, 
    #  'pelvis': 0, 'right_foot': 11, 'right_hand': 5, 'right_lower_arm': 4, 
    #  'right_shin': 10, 'right_thigh': 9, 'right_upper_arm': 3, 'torso': 1}
    
    def __init__(self, gym, sim, asset, skeleton, idk, param = [0, 10, 60, 30, 10]):
        self.param = param
        self.gym = gym
        self.sim = sim
        self.idk = idk
        self.skeleton = skeleton
        spacing = 100.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(sim, lower, upper, 8)
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(0.89))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.actor_handle = self.gym.create_actor(self.env, asset, start_pose)
        self.trajectory = []
        
        self.dof_len = gym.get_actor_dof_count(self.env, self.actor_handle) #(dof,2)

        props = gym.get_actor_dof_properties(self.env, self.actor_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"] = torch.tensor(self.stiff)
        props["damping"] = props["stiffness"] / 10
        assert(gym.set_actor_dof_properties(self.env, self.actor_handle, props))
        
    def build_tensor(self):
        _rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(_rigid_body_states)\
            [self.idk*15:self.idk*15+15]
        
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)[self.idk]
        
        _joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.joint_tensor = gymtorch.wrap_tensor(_joint_tensor)\
            [self.idk*self.dof_len:self.idk*self.dof_len+self.dof_len]
            
        # not changing quantitys
        self.properties = self.gym.get_actor_rigid_body_properties(self.env, self.actor_handle)
        
        
    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector

        
    def overlap(self, old_state, old_traj):
        self.trajectory = old_traj   
        
    
    def cost(self, candidate_p, candidate_q, root_pos, root_orient, root_ang_vel, old_com_pos, old_com_vel):
        self.old_root_ang_vel = root_ang_vel
        self.old_root_orient = root_orient
        self.old_com_pos, self.old_com_vel = old_com_pos, old_com_vel        

        
        self.com_pos, self.com_vel = self.compute_com()
        
        skeleton = self.skeleton
        # now_state = self.gym.acquire_dof_state_tensor(self.sim)
        # now_state = gymtorch.wrap_tensor(now_state).clone()[self.idk*self.dof_len:self.idk*self.dof_len+self.dof_len]
        now_state = self.joint_tensor
        for nid in range(len(skeleton.nodes)):
            pid = skeleton.parents[nid]
            if pid == -1:
                candidate_p[nid] = root_pos
                candidate_q[nid] = root_orient
            else:
                candidate_p[nid] *= 0
        
        target_motion = compute_motion(30, self.skeleton, candidate_q.unsqueeze(0), candidate_p.unsqueeze(0), early_stop=True)
        now_motion = self.compute_motion_from_state(now_state, candidate_p, candidate_q)
        

        # for i in range(0,15):
        #     print(self.rigid_body_states[i, 0:3], now_motion.pos[0,i])
        # print('---------------------------------')
        return self.compute_total_cost(target_motion, now_motion)
        
    def act(self, target_state, record=0):
        if record:
            self.trajectory += [target_state.cpu().numpy()]
        
 
        
    def history(self):
        return [[self.root_tensor.clone(), self.joint_tensor.clone()], self.trajectory.copy()]
        
        
        
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
                dcm = state[t:t+3,0].cpu()
            else:
                dcm = [0,state[t][0].cpu(),0]
            dcm = np.asarray(dcm)
            q[controllable_links[i]] = \
                torch.from_numpy(sRot.from_euler("xyz",dcm,degrees=False).as_quat())
            t+=dofs[i]
            
        root_tensor = self.root_tensor
        root_positions = root_tensor[0:3]
        root_orientations = root_tensor[3:7]
        
        self.root_orient = root_orientations
        self.root_ang_vel = root_tensor[10:13]
        
        for nid in range(len(skeleton.nodes)):
            pid = skeleton.parents[nid]
            if pid == -1:
                p[nid] = root_positions
                q[nid] = root_orientations
            else:
                p[nid] *= 0
            
        return compute_motion(30, self.skeleton, q.unsqueeze(0), p.unsqueeze(0), early_stop=True)
        
        
        
        
        
        
        
        
        
        
        
        
    def compute_total_cost(self, target_motion, new_motion):

        pose_w, root_w, ee_w, balance_w, com_w =\
            self.param[0], self.param[1], self.param[2], self.param[3], self.param[4]
        # pose_w, root_w, ee_w, balance_w, com_w = 0, 10, 60, 0, 0
        pose_cost = self.compute_pose_cost(target_motion, new_motion)
        root_cost = self.compute_root_cost()
        ee_cost = self.compute_ee_cost(target_motion, new_motion)
        balance_cost = self.compute_balance_cost(target_motion, new_motion)
        com_cost = self.compute_com_cost()
        total_cost = pose_w * pose_cost + \
                     root_w * root_cost + \
                     ee_w * ee_cost + \
                     balance_w * balance_cost + \
                     com_w * com_cost
        return total_cost, pose_cost, root_cost, ee_cost, balance_cost, com_cost

    def compute_pose_cost(self, target_motion, new_motion):
        # no need
        """ pose + angular velocity of internal joints in local coordinate """
        error = 0.0
        skeleton = self.skeleton

        for i in range(1,len(skeleton.nodes)):
            # _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(
            #     self._pb_client.getDifferenceQuaternion(sim_joint_ps[i], kin_joint_ps[i])
            # )
            # diff_pos_vel = sim_joint_vs[i] - kin_joint_vs[i]
            p = skeleton.parents[i]
            diff_pose_pos = (target_motion.pos[0,i]-target_motion.pos[0,p]) \
                - (new_motion.pos[0,i]-new_motion.pos[0,p])
            diff_pos_vel = 0 # fail to calulate local velocity, sorry
            error += np.dot(diff_pose_pos, diff_pose_pos) +\
                0.1 * np.dot(diff_pos_vel, diff_pos_vel)
        error /= len(skeleton)            
        return error

    def compute_root_cost(self):
        # done!
        """ orientation + angular velocity of root in world coordinate """
        error = 0.0
        
        diff_root_Q = self.old_root_orient - self.root_orient
        diff_root_w = self.old_root_ang_vel - self.root_ang_vel
        # if(np.isnan(self.root_orient[0]) and 0):
        #     print(self.idk)
        #     print( self.root_tensor)
        error += 1.0 * torch.dot(diff_root_Q, diff_root_Q) + \
                 0.1 * torch.dot(diff_root_w, diff_root_w)
        return error

    def compute_ee_cost(self, target_motion, new_motion):
        # done!
        """ end-effectors (height) in world coordinate """
        error = 0.0
        for nid in self.ees:
            diff_pos = target_motion.pos[0,nid] - new_motion.pos[0,nid]
            diff_pos = diff_pos[-1] # only consider Z-component (height)
            error += np.dot(diff_pos, diff_pos)
        error /= len(self.ees)
        return error

    def compute_com_pos_vel(self, _pos, _vel):
        properties = self.properties
        # for i in range(0,15):
        #     print(properties[i].com)
        com_pos = torch.tensor([.0,.0,.0])
        com_vel = torch.tensor([.0,.0,.0])
        total_mass = 0
        for i in range(len(properties)):
            for j in range(0,3):
                com_pos[j] += properties[i].mass * _pos[i][j].cpu()
                com_vel[j] += properties[i].mass * _vel[i][j].cpu()
            total_mass += properties[i].mass
        return com_pos / total_mass, com_vel / total_mass
    
    def compute_com(self):
        return self.compute_com_pos_vel(self.rigid_body_states[:,0:3], \
                        self.rigid_body_states[:,7:10])
            
    def compute_balance_cost(self, target_motion, new_motion):
        """ balance cost plz see the SamCon paper """
        error = 0.0
        sim_com_pos, sim_com_vel = self.com_pos, self.com_vel
        kin_com_pos, kin_com_vel = self.old_com_pos, self.old_com_vel

        for nid in self.ees:
            sim_planar_vec = sim_com_pos - new_motion.pos[0,nid] 
            kin_planar_vec = kin_com_pos - target_motion.pos[0,nid]
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

        