
import torch
import numpy as np
import collections

c = np.load('data/amp_humanoid_cartwheel.npy', allow_pickle = True).item()

rotation, root = c['rotation']['arr'], c['root_translation']['arr']
dict2 = []

name = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']
for i in range(0,root.shape[0]):
    dict= {}
    dict['pelvis'] = [root[i].tolist(),rotation[i][0].tolist()]

    for j in range(1,len(name)):
        dict[name[j]] = rotation[i][j].tolist()
    dict2 += [dict]
all_dict = {}
all_dict['fps'] = 60
all_dict["frames"] = dict2

import json
file_obj=open('cartwheel.json','w',encoding='utf-8')
json.dump(all_dict, file_obj)