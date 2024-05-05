from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from env import Simulation
import numpy as np
from ref_motion import ReferenceMotion
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm




device = 'cpu'
gym = gymapi.acquire_gym()
compute_device_id, graphics_device_id = 3,3
num_envs = 1 
nSample, nSave = 1000, 100
simulation_dt = 30
sample_dt = 30
rounds = simulation_dt // sample_dt

nExtend = nSample // nSave
    
sim_params = gymapi.SimParams()

sim_params.use_gpu_pipeline = False
sim_params.physx.use_gpu = False

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / simulation_dt
sim_params.substeps = 2
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



gym.prepare_sim(sim)
    


            
while not gym.query_viewer_has_closed(viewer):


    print('aaaaaaaaaaaaaaaaa')
    # gym.set_dof_position_target_tensor(sim, target_state) 

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    gym.step_graphics(sim)
    # if(k==rounds-1 or 1):
    #     gym.draw_viewer(viewer, sim, True)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    
        


# print('ending')
gym.destroy_viewer(viewer)    
gym.destroy_sim(sim)