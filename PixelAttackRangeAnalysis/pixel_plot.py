# -*- coding: utf-8 -*-

import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = open('data/RtR1_analysis_2.json')
RtR_data = json.loads(f.read())

f = open('data/RtW1_analysis_2.json')
RtW_data = json.loads(f.read())

target_RtR_data = RtR_data['8447-0to0']
target_RtW_data = RtW_data['8447-0to7']

x_RtR_data = []
y_RtR_data = []
x_RtW_data = []
y_RtW_data = []
z_RtR_data = []
z_RtW_data = []

channel_index = 1


def get_val(data_str):
    spl = data_str.split(' ')
    s1 = spl[0][1: -1].split(',')
    s2 = spl[1]
    return [int(s1[0]), int(s1[1]), int(s1[2])], int(s2)


for item in target_RtR_data:
    pos, val = get_val(item)
    if pos[2] == channel_index:
        x_RtR_data.append(pos[0])
        y_RtR_data.append(pos[1])
        z_RtR_data.append(val)

for item in target_RtW_data:
    pos, val = get_val(item)
    if pos[2] == channel_index:
        x_RtW_data.append(pos[0])
        y_RtW_data.append(pos[1])
        z_RtW_data.append(val)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_RtR_data, y_RtR_data, z_RtR_data, c='r', marker='o')
ax.scatter(x_RtW_data, y_RtW_data, z_RtW_data, c='b', marker='^')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('channel-0')

plt.savefig('mix_1.jpg')
plt.show()
