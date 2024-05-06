from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from env import Simulation
import numpy as np
from ref_motion import ReferenceMotion
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

def refresh(gym, sim):
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

if __name__ == '__main__':
    device = 'cpu'
    gym = gymapi.acquire_gym()
    compute_device_id, graphics_device_id = 0, 0
    num_envs = 1 
    nSample, nSave = 1000, 100
    simulation_dt = 30
    sample_dt = 30
    rounds = simulation_dt // sample_dt
    
    nExtend = nSample // nSave
        
    sim_params = gymapi.SimParams()

    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    


    # set common parameters
    sim_params.dt = 1 / simulation_dt
    sim_params.substeps = 10
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0


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
    
    
    # add viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    
    
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

    root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),0)
    root_tensor, link_tensor, joint_tensor = root_tensor[0], link_tensor[0], joint_tensor[0]

    
    
    history_target = np.load('best_no_control.npy')
    print('reading history', history_target.shape)
    TIME = history_target.shape[0]
    
    ROOT_TENSOR, JOINT_TENSOR = [], []
    for i in range(0, num_envs):
        ROOT_TENSOR += [root_tensor.cpu().unsqueeze(0).numpy()]
        JOINT_TENSOR += [joint_tensor.cpu().numpy()]        
    ROOT_TENSOR, JOINT_TENSOR = \
        np.concatenate([ROOT_TENSOR], axis=0),np.concatenate([JOINT_TENSOR], axis=0)
    ROOT_TENSOR, JOINT_TENSOR = \
        torch.from_numpy(ROOT_TENSOR), torch.from_numpy(JOINT_TENSOR)
    # ROOT_TENSOR, JOINT_TENSOR = gymtorch.unwrap_tensor(ROOT_TENSOR), gymtorch.unwrap_tensor(JOINT_TENSOR)
         
    

                
    for fid in tqdm(range(TIME)):
        
        if gym.query_viewer_has_closed(viewer):
            break
        
        if fid==0:
            assert(gym.set_actor_root_state_tensor(sim,
                gymtorch.unwrap_tensor(ROOT_TENSOR)))
            assert(gym.set_dof_state_tensor(sim,
                gymtorch.unwrap_tensor(JOINT_TENSOR)))

        target_state = []
        for i in range(0, num_envs):    
            target_state2 = torch.from_numpy(history_target[fid])
            target_state.append(target_state2.numpy())
            envs[i].act(target_state2, record=1)
        

                    
        target_state = np.concatenate([target_state], axis=0)
        target_state = torch.from_numpy(target_state)
                
        target_state = gymtorch.unwrap_tensor(target_state)
        # simulating...
        for k in range(rounds):

            # assert(gym.set_dof_position_target_tensor(sim, target_state) )
    
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            
            refresh(gym, sim)
            
            gym.step_graphics(sim)
            if(k==0):
                gym.draw_viewer(viewer, sim, True)
            # gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        
        print(envs[0].history()[0])


     
            
    
    
    print('ending')
    gym.destroy_viewer(viewer)    
    gym.destroy_sim(sim)