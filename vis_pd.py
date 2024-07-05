from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from env import Simulation
import numpy as np
from ref_motion import ReferenceMotion
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
from main import *
from cfg import *

def refresh(gym, sim):
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

if __name__ == '__main__':
    device = 'cpu'
    gym = gymapi.acquire_gym()
    num_envs = 1 
    sample_dt = simulation_dt
    
    sim = build_sim(gym, simulation_dt)    
    
    # add viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    
    
    asset, reference = read_data(gym, sim)
    
    envs = []
    for i in range(num_envs):
        envs.append(Simulation(gym, sim, asset, reference.skeleton, i))
    refresh(gym, sim)
    for ei in envs:
        ei.build_tensor()

    gym.prepare_sim(sim)

    root_tensor, link_tensor, joint_tensor = reference.state(np.asarray([0]),SSStart/simulation_dt)

    
    
    history_target = np.load(save_file_name.replace('.npy','_control.npy'))
    print('reading history', history_target.shape)
    TIME = history_target.shape[0]
    
    print(history_target.shape)
    
    for safdsvb in range(0,100):
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
                        
            # target_state = np.concatenate([target_state], axis=0)
            target_state = torch.from_numpy(history_target[fid])
          
            # simulating...

            assert(gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_state)) )
            
                
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            
            refresh(gym, sim)
            
            gym.step_graphics(sim)
                
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            
            # print(envs[0].history()[0])


        
            
    
    
    print('ending')
    gym.destroy_viewer(viewer)    
    gym.destroy_sim(sim)