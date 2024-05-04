# def smpl_to_bullet(pose, trans):
#     '''
#         Expect pose to be batch_size x 72
#         trans to be batch_size x 3
#     '''
#     if not torch.is_tensor(pose):
#         pose = torch.tensor(pose)
    
#     pose = pose.reshape(-1, 24, 3)[:, joints_to_use, :]
#     pose_quat = angle_axis_to_quaternion(pose.reshape(-1, 3)).reshape(pose.shape[0], -1, 4)
#     # switch quaternion order
#     # w,x,y,z -> x,y,z,w 
#     pose_quat = pose_quat[:, :, [1, 2, 3, 0]]
#     bullet = np.concatenate((trans, pose_quat.reshape(pose.shape[0], -1)), axis = 1)
#     return bullet


import pickle,joblib
 
with open('./amass/motion/amass_bullet.pkl', 'rb') as f:
    data = joblib.load(f)
    print(data.keys())
    print(data['ACCAD_Female1Running_c3d_C11 -  run turn left (90)_poses']['pose'].shape)
    print(data['ACCAD_Female1Running_c3d_C11 -  run turn left (90)_poses']['fr'])
    