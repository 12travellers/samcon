from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from env import Simulation
import numpy as np
from ref_motion import ReferenceMotion
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
import argparse,time
from cfg import n_links, controllable_links, dofs, limits, param, up_axis
 

# ["torso", "head", 
#         "right_upper_arm", "right_lower_arm",
#         "left_upper_arm", "left_lower_arm", 
#         "right_thigh", "right_shin", "right_foot",
#         "left_thigh", "left_shin", "left_foot"]
simulation_dt = 30
save_file_name = 'debug.npy'
init_pose = './assets/motions/clips_walk.yaml'
character_model = './assets/humanoid.xml'
asset_root = "/mnt/data/caoyuan/issac/samcon/assets/"
asset_file = "humanoid.xml"
SSStart = 0

def get_noisy(joint_pos, joint_vel, reference):

    joint_pos2 = joint_pos.clone().reshape(-1)
    noise = np.random.random(joint_pos2.shape[0]) # sample from [0, 1) uniform distribution
    for i in range(len(limits)): # transform to [-limit, limit)
        noise[i] = 2 * limits[i] * noise[i] - limits[i]
        noise[i] *= np.pi

    
    joint_pos2 += torch.from_numpy(noise).to(joint_pos2.device)
    
    while torch.any(joint_pos2 > np.pi):
        joint_pos2[joint_pos2 > np.pi] -= 2 * np.pi
    while torch.any(joint_pos2 < -np.pi):
        joint_pos2[joint_pos2 < -np.pi] += 2 * np.pi

    joint_pos2 = joint_pos2.reshape(joint_pos.shape)
    
    return joint_pos2 #pos-driven pd control

def refresh(gym, sim):
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)


    
def read_data(gym, sim, device='cpu'):
    reference = ReferenceMotion(motion_file=init_pose, character_model=character_model,
            key_links=np.arange(n_links), controllable_links=controllable_links, dofs=dofs,
            device=device
        )
    asset_opt = gymapi.AssetOptions()
    asset_opt.angular_damping = 0.01
    asset_opt.max_angular_velocity = 100.0
    asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)    
    asset = gym.load_asset(sim, asset_root, asset_file, asset_opt)
    return asset, reference

def build_sim(gym, simulation_dt, use_gpu = False):
    sim_params = gymapi.SimParams()
    # set device
    if use_gpu:
        compute_device_id, graphics_device_id = 2, 2
    else:
        compute_device_id, graphics_device_id = 0, 0
    sim_params.use_gpu_pipeline = use_gpu
    sim_params.physx.use_gpu = use_gpu
    # set common parameters
    sim_params.dt = 1 / simulation_dt
    sim_params.substeps = 10
    if (up_axis==2):
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.up_axis = gymapi.UP_AXIS_Z
    else:
        sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
        sim_params.up_axis = gymapi.UP_AXIS_Y
    # set PhysX-specific parameters
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    #build simulation
    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    # configure the ground plane
    plane_params = gymapi.PlaneParams()
    if (up_axis==2):
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    else:
        plane_params.normal = gymapi.Vec3(0, 1, 0) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0
    # create the ground plane
    gym.add_ground(sim, plane_params)
    
    return sim



if __name__ == '__main__':
    all_target_states = []
    
    sample_dt = simulation_dt
    device = 'cuda:2'
    gym = gymapi.acquire_gym()
    num_envs = 5000
    nSample, nSave = 5000, 200
    nExtend = nSample // nSave
    sim = build_sim(gym, simulation_dt, use_gpu=True)
    asset, reference = read_data(gym, sim, device = device)
    # build each environments(extra is for normal calculation)
    envs = []
    for i in range(num_envs+1):
        envs.append(Simulation(gym, sim, asset, reference.skeleton, i, device, param))
    refresh(gym, sim)
    for ei in envs:
        ei.build_tensor()
    
    gym.prepare_sim(sim)
    

    
    rounds = simulation_dt // sample_dt
    root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),SSStart/simulation_dt)
    best2 = [[root_tensor, joint_tensor], []]
    best = [best2 for i in range(nSave)]
    
    TIME = reference.motion[0].pos.shape[0]
    TIME = 300
    for fid in tqdm(range(SSStart+1,TIME)):
        SS = time.time()
        
        root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),fid/simulation_dt)

        root_tensor_old, link_tensor_old, joint_tensor_old = reference.state(np.asarray([0]),(fid-1)/simulation_dt)
        
        joint_pos, joint_vel, root_pos, root_orient, root_lin_vel, root_ang_vel = \
            reference.state_partial(np.asarray([0]),fid/simulation_dt)
       
        results = []
        for id in range(0, nSave, num_envs//nExtend):
            SS = time.time()
            target_state = []
            # setting all to source status
            ROOT_TENSOR, JOINT_TENSOR = [], []
            for i in range(0, num_envs//nExtend):
                for j in range(0, nExtend):                  
                    envs[i*nExtend+j].overlap(best[id+i][0], best[id+i][1].copy())

                    root_tensor2, joint_tensor2 = best[id+i][0][0], best[id+i][0][1]
                    ROOT_TENSOR += [root_tensor2.unsqueeze(0)]
                    JOINT_TENSOR += [joint_tensor2]
                    
            ROOT_TENSOR += [root_tensor_old.unsqueeze(0)]
            JOINT_TENSOR += [joint_tensor_old]
            
            ROOT_TENSOR, JOINT_TENSOR = \
                torch.cat(ROOT_TENSOR, axis=0),torch.cat(JOINT_TENSOR, axis=0) 

            
            print(time.time()-SS)
            
            SS = time.time()
            assert(gym.set_actor_root_state_tensor(sim,
                gymtorch.unwrap_tensor(ROOT_TENSOR)))
            assert(gym.set_dof_state_tensor(sim,
                gymtorch.unwrap_tensor(JOINT_TENSOR)))
            print(time.time()-SS)
            
            
            # making pd-control's goal
            SS = time.time()
            for i in range(0, num_envs//nExtend):
                for j in range(0, nExtend):
                    target_state2 = get_noisy(joint_pos, joint_vel, reference)
                    envs[i*nExtend+j].act(len(target_state), record=1)
                    target_state.append(target_state2)
            target_state.append(joint_pos)
            all_target_states.append(target_state)
            target_state = torch.cat(target_state, axis=0)
            print(time.time()-SS)
            # simulating...
            for k in range(rounds):
                SS = time.time()
                assert(gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_state)))
                print(time.time()-SS)
                SS = time.time()
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                refresh(gym, sim)
                print(time.time()-SS)
            
            # calculating cost
                      
            for ie in envs:
                ie.compute_com()
            for i in range(num_envs):
                results.append([envs[i].cost(envs[num_envs])
                                ,envs[i].history()])
        # store nSample better ones
        best2 = sorted(results, key=lambda x:x[0][0])
        print()
        print('loss:', best2[0][0])
        
        
        best = [best2[i][1] for i in range(nSample)]
        print('root_pos compare,', best[0][0][0][0:3], root_tensor[0:3])
        print('com_pos compare,', best[0][0][2], envs[num_envs].com_pos)
        if fid % 10 == 0:
            saved_path = []
            for i in range(best[0][1]):
                saved_path+=[all_target_states[i][best[0][1][1]].cpu().unsqueeze(0).numpy()]
            np.save(save_file_name, np.concatenate([saved_path],axis=0))
    
    gym.destroy_sim(sim)