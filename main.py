from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from env import *
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
        noise[i] = (2 * noise[i] - 1 ) * limits[i] * np.pi

    
    joint_pos2 += torch.from_numpy(noise).to(joint_pos2.device)
    
    while torch.any(joint_pos2 > np.pi):
        joint_pos2[joint_pos2 > np.pi] -= 2 * np.pi
    while torch.any(joint_pos2 < -np.pi):
        joint_pos2[joint_pos2 < -np.pi] += 2 * np.pi

    joint_pos2 = joint_pos2.reshape(joint_pos.shape)
    
    return joint_pos2 #pos-driven pd control

def get_full_noisy(joint_pos, limits_gpu, device):
    # noise = np.random.random(joint_pos.shape[0]) # sample from [0, 1) uniform distribution
    noise = torch.rand(joint_pos.shape, device = device)
    noise = noise * limits_gpu
    
    joint_pos2 = joint_pos + noise
    while torch.any(joint_pos2 > np.pi):
        joint_pos2[joint_pos2 > np.pi] -= 2 * np.pi
    while torch.any(joint_pos2 < -np.pi):
        joint_pos2[joint_pos2 < -np.pi] += 2 * np.pi

    return joint_pos2
    


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

def build_sim(gym, simulation_dt, use_gpu = False, device = None):
    sim_params = gymapi.SimParams()
    # set device
    if use_gpu:
        compute_device_id, graphics_device_id = int(device[-1]),int(device[-1])
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
    gym = gymapi.acquire_gym()
    nSample, nSave = 10000, 200
    num_envs = nSample
    nExtend = [100 for i in range(0,50)] + [40 for i in range(0,100)] + [20 for i in range(0,50)]
    device = 'cuda:4'
    sim = build_sim(gym, simulation_dt, use_gpu=True, device=device)
    asset, reference = read_data(gym, sim, device = device)
    limits_gpu = torch.tensor(limits).to(device)
    param_gpu = torch.tensor(param).to(device)
    geoms_gpu = reference.skeleton.geoms.to(device)
    weights_gpu = reference.skeleton.weights.to(device)
    total_weight_gpu = reference.skeleton.total_weight.to(device)
    # build each environments(extra is for normal calculation)
    envs = []
    for i in range(num_envs+1):
        envs.append(Simulation(gym, sim, asset, reference.skeleton, i, device, param))
    refresh(gym, sim)
    for ei in envs:
        ei.build_tensor()
        # ei.init_skeleton(geoms_gpu, weights_gpu, total_weight_gpu)
        ei.init_skeleton(reference.skeleton.geoms, reference.skeleton.weights, reference.skeleton.total_weight)
    _rigid_body_states = gym.acquire_rigid_body_state_tensor(sim)
    BODY = gymtorch.wrap_tensor(_rigid_body_states)[:-15].reshape(-1,15,13)

    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    ROOT = gymtorch.wrap_tensor(_root_tensor)[:-1].reshape(-1,13)
    
    gym.prepare_sim(sim)
    

    
    root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),SSStart/simulation_dt)
    best2 = [[root_tensor, joint_tensor], []]
    best = [best2 for i in range(nSave)]
    
    TIME = reference.motion[0].pos.shape[0]
    TIME = 300
    
    

    
    
    for fid in tqdm(range(SSStart+1,TIME)):
        
        root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),fid/simulation_dt)

        root_tensor_old, link_tensor_old, joint_tensor_old = reference.state(np.asarray([0]),(fid-1)/simulation_dt)
        
        joint_pos, joint_vel, root_pos, root_orient, root_lin_vel, root_ang_vel = \
            reference.state_partial(np.asarray([0]),fid/simulation_dt)
       
        results = []
        results_for_sort = []
            
        SS = time.time()
        # setting all to source status
        ROOT_TENSOR, JOINT_TENSOR = [], []
        tot = 0
        for i in range(0, nSave):
            for j in range(0, nExtend[i]):                  
                envs[tot].overlap(best[i][0], best[i][1].copy())
                tot += 1

                root_tensor2, joint_tensor2 = best[i][0][0], best[i][0][1]
                ROOT_TENSOR += [root_tensor2.unsqueeze(0)]
                JOINT_TENSOR += [joint_tensor2]
                
        ROOT_TENSOR += [root_tensor_old.unsqueeze(0)]
        JOINT_TENSOR += [joint_tensor_old]
        
        ROOT_TENSOR, JOINT_TENSOR = \
            torch.cat(ROOT_TENSOR, axis=0),torch.cat(JOINT_TENSOR, axis=0) 

        
        print('load initial states', time.time()-SS)
        
        SS = time.time()
        assert(gym.set_actor_root_state_tensor(sim,
            gymtorch.unwrap_tensor(ROOT_TENSOR)))
        assert(gym.set_dof_state_tensor(sim,
            gymtorch.unwrap_tensor(JOINT_TENSOR)))
        print('GYM set initial state',time.time()-SS)
        
        
        # making pd-control's goal
        SS = time.time()
        target_state = []
        tot = 0
        for i in range(0, nSave):
            for j in range(0, nExtend[i]):      
                # target_state2 = get_noisy(joint_pos, joint_vel, reference)
                envs[tot].act(len(target_state), record=1)
                tot += 1
                target_state.append(joint_pos.unsqueeze(0))

        target_state = torch.cat(target_state, axis=0)
        target_state = get_full_noisy(target_state, limits_gpu, device)
        target_state = torch.cat((target_state,joint_pos.unsqueeze(0)), axis=0)
        all_target_states.append(target_state)
        print('getting target states',time.time()-SS)
        # simulating...
            
        SS = time.time()
        assert(gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_state)))
        print('GYM set target states', time.time()-SS)
        SS = time.time()
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        refresh(gym, sim)
        print('GYM stimulate and refresh states',time.time()-SS)
        
        # calculating cost
        SS = time.time()  
        # for ie in envs:
        #     ie.compute_com()
        envs[num_envs].compute_com()
        print('compute COM',time.time()-SS)
        SS = time.time()  
        full_cost = []
        full_cost += [torch.zeros(num_envs, device=device)]
        full_cost += compute_full_ee_cost(BODY, envs[num_envs])
        full_cost += compute_full_root_cost(ROOT, envs[num_envs])
        full_cost += compute_full_balance_cost(envs, BODY, envs[num_envs])
        
        full_cost = torch.stack(full_cost,axis=0)
        total_cost = (full_cost.transpose(1,0) * param_gpu.unsqueeze(0)).sum(-1)
        # print(full_cost,total_cost)
        # exit(0)
        print('compute cost',time.time()-SS)
        results_for_sort = total_cost
        SS = time.time()  
        
        for i in range(num_envs):
            results.append([total_cost[i],envs[i].history()])
        
        print('clean up ',time.time()-SS)
        # store nSample better ones
        SS = time.time()  
        ids = torch.argsort(results_for_sort)
        # best2 = sorted(results, key=lambda x:x[0][0])
        print('sort results',time.time()-SS)
        print('loss:', results[ids[0]][0])
        
        best = [ results[ids[i]][1] for i in range(nSave)]
        envs[ids[0]].compute_com()
        print('root_pos compare,', best[0][0][0][0:3], root_tensor[0:3])
        print('com_pos compare,',  envs[ids[0]].com_pos, envs[num_envs].com_pos)
        if fid % 10 == 0:
            saved_path = []
            for i in range(len(best[0][1])):
                saved_path+=[all_target_states[i][best[0][1][1]].cpu().unsqueeze(0).numpy()]
            np.save(save_file_name, np.concatenate([saved_path],axis=0))
    
    gym.destroy_sim(sim)