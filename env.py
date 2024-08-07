from isaacgym import gymtorch
from isaacgym import gymapi
import isaacgym
import torch
import numpy as np
import copy
from ref_motion import compute_motion
from scipy.spatial.transform import Rotation
import quaternion

from cfg import n_links, controllable_links, dofs, up_axis, simulation_dt

class Simulation:
    stiff = [
        600,600,600,50,50,50,
        200,200,200,150,
        200,200,200,150,
        300,300,300,300,200,200,200,
        300,300,300,300,200,200,200,
    ]
    ees_z = [14,8,11,5]
    ees_xy = [14,8,11,5,2][:-1]
    # {'head': 2, 'left_foot': 14, 'left_hand': 8, 'left_lower_arm': 7, 
    #  'left_shin': 13, 'left_thigh': 12, 'left_upper_arm': 6, 
    #  'pelvis': 0, 'right_foot': 11, 'right_hand': 5, 'right_lower_arm': 4, 
    #  'right_shin': 10, 'right_thigh': 9, 'right_upper_arm': 3, 'torso': 1}
    
    def __init__(self, gym, sim, asset, skeleton, idk, device='cpu', param = [0, 10, 60, 30, 10]):
        self.param = param
        self.gym = gym
        self.sim = sim
        self.skeleton = skeleton
        self.idk = idk
        self.device = device
        spacing = 8.0
        if up_axis==2:
            lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        else:
            lower = gymapi.Vec3(-spacing, 0.0,  -spacing,)
            
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
        self.com_pos = None
        assert(gym.set_actor_dof_properties(self.env, self.actor_handle, props))
        
    def init_skeleton(self, geoms, weights, total_weight):
        self.geoms = geoms
        self.weights = weights
        self.total_weight = total_weight
        
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
        # self.properties = self.gym.get_actor_rigid_body_properties(self.env, self.actor_handle)
        
        # self.properties_mass = torch.zeros(15)
        # self.total_mass = 0
        # for i in range(len(self.properties)):
        #     self.properties_mass[i] = self.properties[i].mass
        #     self.total_mass += self.properties[i].mass
        # self.properties_mass = self.properties_mass.to(self.device)
        
        
    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[up_axis] = val
        return base_vector

    def overlap(self, old_state, old_traj):
        self.trajectory = copy.deepcopy(old_traj)  
        
    def cost(self, another):
        return self.compute_total_cost(another)
        
    def act(self, target_state_id, record=0):
        if record:
            self.trajectory.append(target_state_id)
    
    def pos(self):
        return self.rigid_body_states[:,0:3]
    def orient(self):
        # w firsz
        return torch.concat([self.rigid_body_states[:,6:7],self.rigid_body_states[:,3:6]],axis=-1)
    def vel(self):
        return self.rigid_body_states[:,7:10]
    def angular(self):
        return self.rigid_body_states[:,10:13]
 
 
    def history(self):
        return [[self.root_tensor.clone(), self.joint_tensor.clone(), self.com_pos], copy.deepcopy(self.trajectory)]
        
        
        
    # def compute_motion_from_state(self, state, candidate_p, candidate_q):
    #     skeleton = self.skeleton
    #     p = candidate_p.clone()
    #     q = candidate_q.clone()
    #     state = state.reshape(-1, 2)

        
    #     t = 0
    #     for i in range(0,len(controllable_links)):
    #         dcm = None
    #         if dofs[i] == 3:
    #             dcm = state[t:t+3,0].cpu()
    #         else:
    #             dcm = [0,state[t][0].cpu(),0]
    #         dcm = np.asarray(dcm)
    #         q[controllable_links[i]] = \
    #             torch.from_numpy(sRot.from_euler("xyz",dcm,degrees=False).as_quat())
    #         t+=dofs[i]
            
    #     root_tensor = self.root_tensor
    #     root_positions = root_tensor[0:3]
    #     root_orientations = root_tensor[3:7]
        
    #     self.root_orient = root_orientations
    #     self.root_ang_vel = root_tensor[10:13]
        
    #     for nid in range(len(skeleton.nodes)):
    #         pid = skeleton.parents[nid]
    #         if pid == -1:
    #             p[nid] = root_positions
    #             q[nid] = root_orientations
    #         else:
    #             p[nid] *= 0
            
    #     return compute_motion(30, self.skeleton, q.unsqueeze(0), p.unsqueeze(0), early_stop=True)
    
    def compute_com_pos_vel(self, pos, orient, vel, angular):
        orient = Rotation.from_quat(orient.cpu())
        angular = Rotation.from_euler('xyz',angular.cpu()).as_rotvec() / (simulation_dt**0.5)
        orient2 = Rotation.from_rotvec(angular).inv() * orient
        
        orient[1:] = orient[self.skeleton.parents[1:]].inv() * orient[1:] 
        orient2[1:] = orient2[self.skeleton.parents[1:]].inv() * orient2[1:] 
        # R0 R1 R2 R3
        
        com_pos = torch.zeros(3, device = self.device)
        com_vel = torch.zeros(3, device = self.device)
        for i in range(0,len(orient)):
            # print(self.skeleton.nodes[i])
            # print(pos[i],self.weights[i])
            com_pos += self.weights[i] * \
                (pos[i]+torch.from_numpy(orient[i].apply(self.skeleton.geoms[i])).to(self.device))
            com_vel += self.weights[i] * \
                (pos[i]-vel[i]/simulation_dt+torch.from_numpy(orient2[i].apply(self.skeleton.geoms[i])).to(self.device))
        # exit(0)
        com_vel = (com_pos - com_vel) * simulation_dt
        
        # com_pos = (self.properties_mass.unsqueeze(-1) * _pos).sum(axis=0).squeeze(0)
        # com_vel = (self.properties_mass.unsqueeze(-1) * _vel).sum(axis=0).squeeze(0)
        self.com_pos, self.com_vel = com_pos / self.total_weight, com_vel / self.total_weight
    
    def compute_com(self):
        self.compute_com_pos_vel(self.pos(), self.orient(), self.vel(), self.angular())
        
        
        
    # def cost_provided(self, root_cost, ee_cost, balance_cost, com_cost):
    #     pose_w, root_w, ee_w, balance_w, com_w =\
    #         self.param[0], self.param[1], self.param[2], self.param[3], self.param[4]
    #     pose_cost = 0
    #     total_cost = pose_w * pose_cost + \
    #                  root_w * root_cost + \
    #                  ee_w * ee_cost + \
    #                  balance_w * balance_cost + \
    #                  com_w * com_cost
    #     return total_cost, pose_cost, root_cost, ee_cost, balance_cost, com_cost
    
    # def compute_total_cost(self, another):
    #     pose_w, root_w, ee_w, balance_w, com_w =\
    #         self.param[0], self.param[1], self.param[2], self.param[3], self.param[4]
    #     # pose_w, root_w, ee_w, balance_w, com_w = 0, 10, 60, 0, 0
    #     pose_cost = 0#self.compute_pose_cost(another)
    #     root_cost = self.compute_root_cost(another)
    #     ee_cost = self.compute_ee_cost(another)
    #     balance_cost = self.compute_balance_cost(another)
    #     com_cost = self.compute_com_cost(another)
    #     total_cost = pose_w * pose_cost + \
    #                  root_w * root_cost + \
    #                  ee_w * ee_cost + \
    #                  balance_w * balance_cost + \
    #                  com_w * com_cost
    #     return total_cost, pose_cost, root_cost, ee_cost, balance_cost, com_cost

    # def compute_pose_cost(self, another):
    #     # no need
    #     """ pose + angular velocity of internal joints in local coordinate """
    #     error = 0.0
    #     skeleton = self.skeleton

    #     for i in range(1,len(skeleton.nodes)):
    #         # _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(
    #         #     self._pb_client.getDifferenceQuaternion(sim_joint_ps[i], kin_joint_ps[i])
    #         # )
    #         # diff_pos_vel = sim_joint_vs[i] - kin_joint_vs[i]
    #         p = skeleton.parents[i]
    #         diff_pose_pos = (self.pos()[i]-self.pos()[p]) \
    #             - (another.pos()[i]-another.pos()[p])
    #         # diff_pos_vel = 0 # fail to calulate local velocity, sorry
    #         error += torch.dot(diff_pose_pos, diff_pose_pos) 
    #         # + 0.1 * torch.dot(diff_pos_vel, diff_pos_vel)
    #     error /= len(skeleton)            
    #     return error

    # def compute_root_cost(self, another):
    #     # done!
    #     """ orientation + angular velocity of root in world coordinate """
    #     error = 0.0
    #     diff_root_Q = another.root_tensor[0:3] - self.root_tensor[0:3]
    #     diff_root_w = another.root_tensor[10:13] - self.root_tensor[10:13]
    #     error = 1.0 * (diff_root_Q* diff_root_Q) + 0.1 * (diff_root_w* diff_root_w)
    #     return error.sum()

    # def compute_ee_cost(self, another):
    #     # done!
    #     """ end-effectors (height) in world coordinate """
    #     error = 0.0
    #     for nid in self.ees_z:
    #         diff_pos = another.pos()[nid] - self.pos()[nid]
    #         diff_pos = diff_pos[up_axis] # only consider Z-component (height)
    #         error += (diff_pos * diff_pos).item()
    #     error /= len(self.ees_z)
    #     return error

  
            
    # def compute_balance_cost(self, another):
    #     """ balance cost plz see the SamCon paper """
    #     error = 0.0
    #     sim_com_pos, sim_com_vel = self.com_pos, self.com_vel
    #     kin_com_pos, kin_com_vel = another.com_pos, another.com_vel

    #     for nid in self.ees_xy:
    #         sim_planar_vec = sim_com_pos - self.pos()[nid] 
    #         kin_planar_vec = kin_com_pos - another.pos()[nid]
    #         diff_planar_vec = sim_planar_vec - kin_planar_vec
    #         diff_planar_vec[up_axis] = 0
    #         # diff_planar_vec = diff_planar_vec[:up_axis] + diff_planar_vec[up_axis+1:] # only consider XY-component
    #         error += diff_planar_vec * diff_planar_vec
    #     error /= len(self.ees_xy) * 1.7
    #     return error.sum()
    
    # def compute_com_cost(self, another):
    #     """ CoM (position linVel) in world coordinate """
    #     error = 0.0
    #     sim_com_pos, sim_com_vel = self.com_pos, self.com_vel
    #     kin_com_pos, kin_com_vel = another.com_pos, another.com_vel
        
    #     diff_com_pos = sim_com_pos - kin_com_pos
    #     diff_com_vel = sim_com_vel - kin_com_vel
        
    #     error = 1.0 * (diff_com_pos * diff_com_pos) + 0.1 * (diff_com_vel * diff_com_vel)
    #     return error.sum()

        
def compute_full_ee_cost(BODY, another):
    diff_pos = another.pos()[another.ees_z,:].unsqueeze(0) - BODY[:,another.ees_z,0:3]
    diff_pos = diff_pos[:,:,up_axis] # only consider Z-component (height)
    error = (diff_pos * diff_pos).sum(axis=-1)
    error /= len(another.ees_z)
    return [error]

def compute_full_balance_cost(envs, BODY, another):
    def my_inverse(x):
        x = quaternion.as_float_array(x)

        x[...,1:] = -x[...,1:]
        x = quaternion.as_quat_array(x)
        return x
    POS, VEL = BODY[:,:,0:3], BODY[:,:,7:10]

    ORI = quaternion.as_quat_array(torch.concat([BODY[:,:,6:7],BODY[:,:,3:6]],axis=-1).cpu().numpy())
    
    angular = quaternion.as_rotation_vector(quaternion.from_euler_angles(BODY[:,:,10:13].cpu().numpy())) / (simulation_dt**0.5)
    ORI2 = my_inverse(quaternion.from_rotation_vector(angular)) * ORI

    ORI[:,1:] = my_inverse(ORI[:, another.skeleton.parents[1:]]) * ORI[:,1:] 
    ORI2[:,1:] = my_inverse(ORI2[:, another.skeleton.parents[1:]]) * ORI2[:,1:]
    
    
    offset = torch.from_numpy(quaternion.as_rotation_matrix(ORI) @ another.geoms.unsqueeze(0).unsqueeze(-1).cpu().numpy()).squeeze(-1).to(POS.device)
    offset2 = torch.from_numpy(quaternion.as_rotation_matrix(ORI2) @ another.geoms.unsqueeze(0).unsqueeze(-1).cpu().numpy()).squeeze(-1).to(POS.device)

    COM_POS = (another.weights.unsqueeze(0).unsqueeze(-1) * (POS + offset)).sum(axis=-2)
    COM_VEL = (another.weights.unsqueeze(0).unsqueeze(-1) * (POS - VEL/simulation_dt + offset2)).sum(axis=-2)
    COM_VEL = (COM_POS - COM_VEL) * simulation_dt 
    COM_POS, COM_VEL = COM_POS / another.total_weight, COM_VEL / another.total_weight
    
    
    # for i in range(0,len(envs)-1):
    #     envs[i].compute_com()
    # COM_POS = torch.concat([envs[i].com_pos.unsqueeze(0) for i in range(0,len(envs)-1)], axis=0)
    # COM_VEL = torch.concat([envs[i].com_vel.unsqueeze(0) for i in range(0,len(envs)-1)], axis=0)

    
    # print('debug com',COM_POS[0],COM_POS[1],COM_POS[2])
    # print('debug comvel',COM_VEL[0],COM_VEL[1],COM_VEL[2])
    diff_com_pos = COM_POS - another.com_pos.unsqueeze(0)
    diff_com_vel = COM_VEL - another.com_vel.unsqueeze(0)
    
    error_com = 2.0 * (diff_com_pos * diff_com_pos) + 0 * 0.1 * (diff_com_vel * diff_com_vel)
    error_com = error_com.sum(axis=-1)
        
    sim_planar_vec = COM_POS.unsqueeze(1) - POS[:,another.ees_xy,:]
    kin_planar_vec = another.com_pos.unsqueeze(0) - another.pos()[another.ees_xy,:] 
    diff_planar_vec = sim_planar_vec - kin_planar_vec
    diff_planar_vec[:,:,up_axis] = 0 # only consider XY-component
    error_balance = (diff_planar_vec * diff_planar_vec).sum(axis=-1).sum(axis=-1)
    error_balance /= len(another.ees_xy)
    
    return [error_balance/1.7, error_com]
    
def compute_full_root_cost(ROOT, another):
    # done!
    """ orientation + angular velocity of root in world coordinate """
    diff_root_Q = another.root_tensor[0:3].unsqueeze(0) - ROOT[:,0:3]
    diff_root_w = another.root_tensor[10:13].unsqueeze(0)- ROOT[:,10:13]
    error = 1.0 * (diff_root_Q* diff_root_Q) + 0.1 * (diff_root_w* diff_root_w)
    return [error.sum(axis=-1)]