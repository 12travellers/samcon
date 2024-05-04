from isaacgym import gymtorch
from isaacgym import gymapi
import isaacgym as gym
import torch


class Simulation:
    def __init__(self, sim, asset, height = 1):
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = gym.create_env(sim, lower, upper, 8)
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, height, 0.0)

        self.actor_handle = gym.create_actor(self.env, asset, pose, "MyActor", i, 1)
        
        self.state = None
        self.trajectory = []

        
    def overlap(self, old_state, old_traj):
        for step in old_traj:
            
        self.state = self.old_state
        self.trajectory = old_traj    
    
    def reward():
        NotImplementedError
        
    def act(self, target_state):
        self.trajectory += [target_state]
        
        
    def history():
        
        

        