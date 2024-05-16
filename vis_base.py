from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from env import Simulation
import numpy as np
from ref_motion import ReferenceMotion
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

from main import read_data, build_sim, simulation_dt, save_file_name, SSStart


if __name__ == '__main__':
    device = 'cpu'
    gym = gymapi.acquire_gym()
    compute_device_id, graphics_device_id = 0, 0
    num_envs = 1 
    sample_dt = simulation_dt
    rounds = simulation_dt // sample_dt

    sim = build_sim(gym, simulation_dt)
    
    # add viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    
    asset, reference = read_data(gym, sim)
    
    envs = []
    for i in range(num_envs):
        envs.append(Simulation(gym, sim, asset, reference.skeleton, i))
    for ei in envs:
        ei.build_tensor()

    gym.prepare_sim(sim)

    root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),SSStart/simulation_dt)

    
    
    history_target = np.load(save_file_name)
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
        gymtorch.unwrap_tensor(ROOT_TENSOR))
    gym.set_dof_state_tensor(sim,
        gymtorch.unwrap_tensor(JOINT_TENSOR))

                
    for fid in tqdm(range(SSStart+1,TIME)):
        
        if gym.query_viewer_has_closed(viewer):
            break

        root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),fid/simulation_dt)

        joint_pos2 = joint_tensor.clone()
        # while torch.any(joint_pos2 > np.pi):
        #     joint_pos2[joint_pos2 > np.pi] -= 2 * np.pi
        # while torch.any(joint_pos2 < -np.pi):
        #     joint_pos2[joint_pos2 < -np.pi] += 2 * np.pi
        
        target_state = []
        ROOT_TENSOR, JOINT_TENSOR = [], []
        for i in range(0, num_envs):
            ROOT_TENSOR += [root_tensor.cpu().unsqueeze(0).numpy()]
            JOINT_TENSOR += [joint_pos2.cpu().numpy()]
            

        
        ROOT_TENSOR, JOINT_TENSOR = \
            np.concatenate([ROOT_TENSOR], axis=0),np.concatenate([JOINT_TENSOR], axis=0)
        ROOT_TENSOR, JOINT_TENSOR = \
            torch.from_numpy(ROOT_TENSOR), torch.from_numpy(JOINT_TENSOR)
                    
        for k in range(rounds):

            print('aaaaaaaaaaaaaaaaa', k)
            gym.set_actor_root_state_tensor(sim,
                gymtorch.unwrap_tensor(ROOT_TENSOR))
            gym.set_dof_state_tensor(sim,
                gymtorch.unwrap_tensor(JOINT_TENSOR))
    
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            
            gym.step_graphics(sim)
            if(k==rounds-1 or 1):
                gym.draw_viewer(viewer, sim, True)
            # gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

     
            
    
    
    print('ending')
    gym.destroy_viewer(viewer)    
    gym.destroy_sim(sim)