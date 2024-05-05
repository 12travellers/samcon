from isaacgym import gymtorch
from isaacgym import gymapi
import isaacgym as gym
import torch

from env import Simulation
import numpy as np
from ref_motion import ReferenceMotion
from scipy.spatial.transform import Rotation as sRot

# ["torso", "head", 
#         "right_upper_arm", "right_lower_arm",
#         "left_upper_arm", "left_lower_arm", 
#         "right_thigh", "right_shin", "right_foot",
#         "left_thigh", "left_shin", "left_foot"]
limits = [.1,.1,.1,
          
          .2,.2,.2,
          
          .2,.2,.2,
          .0,
          
          .2,.2,.2,
          .0,
          
          .4,.4,.1,
          .0,
          .4,.2,.1,
          
          .4,.4,.1,
          .0,
          .4,.2,.1,
          ]
def get_noisy(joint_p, joint_q, reference):
    noise = np.random.random(joint_q.shape[0]) # sample from [0, 1) uniform distribution
    for i in range(len(limits)): # transform to [-limit, limit)
        noise[i] = 2 * limits[i] * noise[i] - limits[i]
    

    joint_q2 = joint_q + noise
    while np.any(joint_q2 > np.pi):
        joint_q2[joint_q2 > np.pi] -= 2 * np.pi
    while np.any(joint_q2 < -np.pi):
        joint_q2[joint_q2 < -np.pi] += 2 * np.pi

 
    return reference.state3(joint_p, joint_q2)



if __name__ == '__main__':
    device = 'gpu:3'
    compute_device_id, graphics_device_id = 3, 3
    num_envs = 50
    nSample, nSave = 1000, 100
    simulation_dt = 300
    sample_dt = 30
    
    nExtend = nSample // nSave
        
    sim_params = gymapi.SimParams()

    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    
    # get default set of parameters
    sim_params = gymapi.SimParams()

    # set common parameters
    sim_params.dt = 1 / simulation_dt
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    # set Flex-specific parameters
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20
    sim_params.flex.relaxation = 0.8
    sim_params.flex.warm_start = 0.5

    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    
    # configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0

    # create the ground plane
    gym.add_ground(sim, plane_params)
    
    n_links, controllable_links = 15, [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
    dofs = [3, 3, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3]
    init_pose = './assets/motions/clips_run.yaml'
    character_model = './assets/humanoid.xml'
    reference = ReferenceMotion(motion_file=init_pose, character_model=character_model,
            key_links=np.arange(n_links), controllable_links=controllable_links, dofs=dofs,
            device=device
        )
    
    asset_root = "/mnt/data/caoyuan/issac/samcon/assets/"
    asset_file = "humanoid.xml"
    asset_opt = gymapi.AssetOptions()
    asset_opt.angular_damping = 0.01
    asset_opt.max_angular_velocity = 100.0
    asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)    
    asset = gym.load_asset(sim, asset_root, asset_file, asset_opt)
    
    envs = []
    for i in range(num_envs):
        envs.append(Simulation(sim, asset, reference.skeleton, i))
    
    gym.prepare_sim(sim)
    
    
    rounds = simulation_dt // sample_dt
    root_tensor, link_tensor, joint_tensor = reference.state([0],0,0)
    root_tensor, link_tensor, joint_tensor = root_tensor[0], link_tensor[0], joint_tensor[0]
    best2 = [[root_tensor, joint_tensor], []]
    best = [best2 for i in range(nSave)]
    for i in range(0, reference.motion_length[0]):
        fid = i
        root_tensor, link_tensor, joint_tensor = reference.state([0],0,fid)
        root_tensor, link_tensor, joint_tensor = root_tensor[0], link_tensor[0], joint_tensor[0]
        joint_p, joint_q = reference.state2([0],0,fid)
        joint_p, joint_q = joint_p[0], joint_q[0]
        
        results = []
        for id in range(0, nSave, num_envs//nExtend):
            target_state = []
            for i in range(0, num_envs//nExtend):
                for j in range(0, nExtend):
                    envs[i*nExtend+j].overlap(best[id][j][0], best[id][j][1])
                    target_state2 = get_noisy(joint_p, joint_q)
                    target_state.append(target_state2)
                    envs[i*nExtend+j].act(target_state2, record=1)
                    
            for k in range(rounds):
                for i in range(0,num_envs):
                    envs[i].act(target_state[i])
                gym.simulate(sim)
                
            for i in range(0, num_envs):
                results.append([envs[i].cost(joint_tensor),envs[i].history()])
                
        best = sorted(results, lambda x:x[0])[:nSample]
        
    np.save('best.npy', np.asarray(best[0][1]))
        
    gym.destroy_sim(sim)