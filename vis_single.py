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
    
    while torch.any(joint_pos2 > np.pi):
        joint_pos2[joint_pos2 > np.pi] -= 2 * np.pi
    while torch.any(joint_pos2 < -np.pi):
        joint_pos2[joint_pos2 < -np.pi] += 2 * np.pi

    joint_pos2 = joint_pos2.reshape(joint_pos.shape)
    # return reference.state_joint_after_partial(joint_pos2, joint_vel)
    return joint_pos2 #pos-driven pd control



if __name__ == '__main__':
    device = 'cpu'
    gym = gymapi.acquire_gym()
    compute_device_id, graphics_device_id = 3, 3
    num_envs = 1 
    nSample, nSave = 1000, 100
    simulation_dt = 300
    sample_dt = 30
    
    nExtend = nSample // nSave
        
    sim_params = gymapi.SimParams()

    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    
    # get default set of parameters
    sim_params = gymapi.SimParams()

    # set common parameters
    sim_params.dt = 1 / simulation_dt
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Y
    sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)

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
    plane_params.normal = gymapi.Vec3(0, 1, 0) # y-up!
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
        envs.append(Simulation(gym, sim, asset, reference.skeleton, i))
    for ei in envs:
        ei.build_tensor()
    gym.prepare_sim(sim)
        
    
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    
    
    rounds = simulation_dt // sample_dt
    root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),0,0)
    root_tensor, link_tensor, joint_tensor = root_tensor[0], link_tensor[0], joint_tensor[0]
    best2 = [[root_tensor, joint_tensor], []]
    best = [best2 for i in range(nSave)]
    
    
    history_target = np.load('best.npy')
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
         
    gym.set_actor_root_state_tensor(sim,
        gymtorch.unwrap_tensor(JOINT_TENSOR))
    gym.set_dof_state_tensor(sim,
        gymtorch.unwrap_tensor(ROOT_TENSOR))

                
    for fid in tqdm(range(TIME)):
        
        if gym.query_viewer_has_closed(viewer):
            break

        target_state = []
        ROOT_TENSOR, JOINT_TENSOR = [], []
        for i in range(0, num_envs):
            
            target_state2 = torch.from_numpy(history_target[fid])
            target_state.append(target_state2.numpy())
            envs[i].act(target_state2, record=1)
        

                    
        target_state = np.concatenate([target_state], axis=0)
        target_state = torch.from_numpy(target_state)
                
        target_state = gymtorch.unwrap_tensor(target_state)
        # simulating...
        print('aaaaaaaaaaaaaaaaa')
        for k in range(rounds):

            print('aaaaaaaaaaaaaaaaa', k)
            gym.set_dof_position_target_tensor(sim, target_state) 
    
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            
            print('aaaaaaaaaaaaaaaaa')
            gym.step_graphics(sim)
            print('aaaaaaaaaaaaaaaaa')
            if(k==rounds-1):
                gym.draw_viewer(viewer, sim, True)
            print('aaaaaaaaaaaaaaaaa')
        
            gym.sync_frame_time(sim)
            # gym.viewer_camera_look_at(viewer, self.envs[tar_env], cam_pos, self.cam_target)

     
            
    
    
    print('ending')
    gym.destroy_viewer(viewer)    
    gym.destroy_sim(sim)