from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from env import Simulation
import numpy as np
from ref_motion import ReferenceMotion
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

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


def get_noisy(joint_pos, joint_vel, reference):

    joint_pos2 = joint_pos.clone().reshape(-1)
    noise = np.random.random(joint_pos2.shape[0]) # sample from [0, 1) uniform distribution
    for i in range(len(limits)): # transform to [-limit, limit)
        noise[i] = 2 * limits[i] * noise[i] - limits[i]
        # noise[i] *= np.pi

    
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

if __name__ == '__main__':
    device = 'cuda:3'
    gym = gymapi.acquire_gym()
    compute_device_id, graphics_device_id = 3, 3
    num_envs = 50
    nSample, nSave = 1000, 100
    simulation_dt = 30
    sample_dt = 30
    
    nExtend = nSample // nSave
        
    sim_params = gymapi.SimParams()

    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    


    # set common parameters
    sim_params.dt = 1 / simulation_dt
    sim_params.substeps = 10
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
    init_pose = './assets/motions/clips_walk.yaml'
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
        envs.append(Simulation(gym, sim, asset, reference.skeleton, i))
    refresh(gym, sim)
    for ei in envs:
        ei.build_tensor()
    
    gym.prepare_sim(sim)
    
    
    rounds = simulation_dt // sample_dt
    root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),0)
    root_tensor, link_tensor, joint_tensor = root_tensor[0], link_tensor[0], joint_tensor[0]
    best2 = [[root_tensor, joint_tensor], []]
    best = [best2 for i in range(nSave)]
    
    TIME = reference.motion[0].pos.shape[0]
    TIME = 300
    for fid in tqdm(range(1,TIME)):
        
        root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),fid/30)
        # root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),0)
        root_tensor, link_tensor, joint_tensor = root_tensor[0], link_tensor[0], joint_tensor[0]
        
        joint_pos, joint_vel, root_pos, root_orient, root_lin_vel, root_ang_vel = \
            reference.state_partial(np.asarray([0]),fid/30)
        # joint_pos, joint_vel, root_pos, root_orient, root_lin_vel, root_ang_vel = \
            # reference.state_partial(np.asarray([0]),0)
        joint_pos, joint_vel, root_pos, root_orient, root_lin_vel, root_ang_vel = \
            joint_pos[0], joint_vel[0], root_pos[0], root_orient[0], root_lin_vel[0], root_ang_vel[0]

        results = []
        for id in range(0, nSave, num_envs//nExtend):
            target_state = []
            # setting all to source status
            ROOT_TENSOR, JOINT_TENSOR = [], []
            for i in range(0, num_envs//nExtend):
                for j in range(0, nExtend):                  
                    envs[i*nExtend+j].overlap(best[id+i][0], best[id+i][1].copy())

                    root_tensor2, joint_tensor2 = best[id+i][0][0], best[id+i][0][1]
                    ROOT_TENSOR += [root_tensor2.unsqueeze(0).cpu().numpy()]
                    JOINT_TENSOR += [joint_tensor2.cpu().numpy()]
                
            ROOT_TENSOR, JOINT_TENSOR = \
                np.concatenate([ROOT_TENSOR], axis=0),np.concatenate([JOINT_TENSOR], axis=0) 

            ROOT_TENSOR, JOINT_TENSOR = \
                torch.from_numpy(ROOT_TENSOR).to(device), torch.from_numpy(JOINT_TENSOR).to(device)
            
            if(fid==1 or 1):
                assert(gym.set_actor_root_state_tensor(sim,
                    gymtorch.unwrap_tensor(ROOT_TENSOR)))
                assert(gym.set_dof_state_tensor(sim,
                    gymtorch.unwrap_tensor(JOINT_TENSOR)))
            
            # making pd-control's goal
            for i in range(0, num_envs//nExtend):
                for j in range(0, nExtend):
                    target_state2 = get_noisy(joint_pos, joint_vel, reference)
                    target_state.append(target_state2.cpu().numpy())
                    envs[i*nExtend+j].act(target_state2, record=1)
            target_state = np.concatenate([target_state], axis=0)
            target_state = torch.from_numpy(target_state).to(device)
                
            # print("target_state_info", target_state.max(),target_state.min())
            
            target_state = gymtorch.unwrap_tensor(target_state)
            # simulating...

            
            
            for k in range(rounds):
                assert(gym.set_dof_position_target_tensor(sim, target_state))
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                
                refresh(gym, sim)
            
            #not dof_only, every information is here(for p/q, except root)
            _pos, _vel = reference.motion[0].pos[fid],\
                reference.motion[0].lin_vel[fid]
            candidate_p, candidate_q = reference.motion[0].local_p[fid], \
                reference.motion[0].local_q[fid]
            
                
            com_pos, com_vel = envs[0].compute_com_pos_vel(_pos, _vel)
            
            # calculating cost
            for i in range(num_envs):
                results.append([envs[i].cost(candidate_p.clone(), candidate_q.clone(), 
                                             root_pos, root_orient,
                                             root_ang_vel, com_pos, com_vel)
                                ,envs[i].history()])
            # break
        # store nSample better ones
        best2 = sorted(results, key=lambda x:x[0])
        print('loss:', best2[0][0])
        
        best = [best2[i][1] for i in range(nSample)]
        print("showing........")
        print(best[0][0][0])
        print(root_tensor)
        np.save('best_no_control.npy', np.asarray(best[0][1]))
        
        
    #save history of targets in pd-control
        
    gym.destroy_sim(sim)