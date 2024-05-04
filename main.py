from isaacgym import gymtorch
from isaacgym import gymapi
import isaacgym as gym
import torch

from env import Simulation
import numpy as np

def get_noisy(motion):
    


if __name__ == '__main__':
    compute_device_id, graphics_device_id = 3, 3
    num_envs = 50
    nSample, nSave = 1000, 100
    simulation_dt = 200
    sample_dt = 10
    
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
    
    
    asset_root = "./amass"
    asset_file = "human.urdf"
    asset = gym.load_asset(sim, asset_root, asset_file)
    
    envs = []
    for i in range(num_envs):
        envs.append(Simulation(sim, asset))
    
    
    gym.prepare_sim(sim)
    
    
    rounds = simulation_dt // sample_dt
    best2 = []
    best = [best2 for i in range(nSave)]
    for i in range(0, len(reference)):
        results = []
        for id in range(0, nSave, num_envs//nExtend):
            
            for i in range(0, num_envs//nExtend):
                for j in range(0, nExtend):
                    envs[i*nExtend+j].overlap(best[id][j][0], best[id][j][1])
                    
                    target_state = get_noisy(reference[i])
                    envs[i*nExtend+j].act(target_state)
                    
            for i in range(rounds):
                gym.simulate(sim)
                
            for i in range(0, num_envs):
                results.append([envs[i].reward(),envs[i].history()])
                
        best = sorted(results)[:nSample]
        
    np.save('best.npy', np.asarray(best[0][1]))
        
    gym.destroy_sim(sim)