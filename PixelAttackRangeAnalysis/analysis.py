# -*- coding: utf-8 -*-

import json
import os

PATH = 'data/'


def get_one_pixel_one_channel_with_channel_val_key(json_obj):
    """
    Get dict key from origin RtW(WtR) data.

    :param json_obj: data item
    :return: key value str
    """
    return '{}-[{},{},{}]-{}to{}'.format(json_obj['image_index'],
                                         json_obj['pixel_position'][0],
                                         json_obj['pixel_position'][1],
                                         json_obj['pixel_position'][2],
                                         json_obj['origin_predict'],
                                         json_obj['changed_predict'])

def get_attack_target_with_one_pixel_one_channel_key(json_obj):
    """
    Get dict key from origin RtW(WtR) data.

    :param json_obj: data item
    :return: key value str
    """
    return '{}-{}to{}'.format(json_obj['image_index'],
                              json_obj['origin_predict'],
                              json_obj['changed_predict'])


def save_one_pixel_one_channel_with_channel_val(data, file_name='analysis.json'):
    """
    Save data (one pixel one channel => channel val).

    :param data: origin data list
    :param file_name: file's name need to save
    :return: none
    """
    analysis_data = {}
    for item in data:
        key_str = get_one_pixel_one_channel_with_channel_val_key(item)
        if not analysis_data.__contains__(key_str):
            analysis_data[key_str] = [item['channel_val']]
        else:
            analysis_data[key_str].append(item['channel_val'])

    with open('data/' + file_name, 'w') as fw:
        fw.write(json.dumps(analysis_data))


def save_attack_target_with_one_pixel_one_channel(data, file_name='analysis.json'):
    """
    Save data (attack target => pixel position, channel val).

    :param data: origin data list
    :param file_name: file's name need to save
    :return: none
    """
    analysis_data = {}
    for item in data:
        key_str = get_attack_target_with_one_pixel_one_channel_key(item)
        value = '[{},{},{}] {}'.format(item['pixel_position'][0],
                                       item['pixel_position'][1],
                                       item['pixel_position'][2],
                                       item['channel_val'])
        if not analysis_data.__contains__(key_str):
            analysis_data[key_str] = [value]
        else:
            analysis_data[key_str].append(value)

    with open('data/' + file_name, 'w') as fw:
        fw.write(json.dumps(analysis_data))


f = open(PATH + '/RtW1.json', 'r')
data = json.loads(f.read())

save_one_pixel_one_channel_with_channel_val(data, file_name='RtW1_analysis_1.json')
save_attack_target_with_one_pixel_one_channel(data, file_name='RtW1_analysis_2.json')

# save_attack_target_with_one_pixel_one_channel(data, file_name='')
