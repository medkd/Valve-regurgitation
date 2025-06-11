import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorflow import keras
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
import sys
import shutil
from logging.handlers import RotatingFileHandler
from ensemble_model_for_MR import get_model_for_MR
from ensemble_model_for_TR import get_model_for_TR
import cv2
import copy
import asyncio
import json

LOG_LEVEL = logging.ERROR
root_logger = tf.get_logger()
root_logger.setLevel(LOG_LEVEL)

EXP_NAME = 'ensemble'


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
if os.path.isdir('log') == False:
    os.mkdir('log')


# root_logger = logging.getLogger('echo')
root_logger.setLevel(LOG_LEVEL)
handler = RotatingFileHandler(
    'log/error.log', 'a', encoding='utf-8', maxBytes=(10*1024*1024), backupCount=2)
handler.setFormatter(logging.Formatter(
    '[%(levelname)s]\t%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
root_logger.addHandler(handler)

# import tensorflow.keras.backend as K


label_2_index = {
    'Other': 0, 'A2C': 1,  'A3C': 2,   'A4C': 3,
    'Apical 2-chamber–color': 4, 'Apical 3-chamber–color': 5, 'Apical 4-chamber–color': 6,
    'RV inflow - color': 7, 'RV_inflow': 8, 'SAX_ao': 9, 'Short axis at aortic valve - color': 10}
index_2_label = {index: label for label, index in label_2_index.items()}

'''
MR_views以及TR_views目前還沒有列入a4c and a4c_color
這兩個set在"make_task_lst"會被修改，會被加入a4c, a4c_color真正使用的檔案名稱
例如:
    a4c的view是a4c_file或是a4c_mr_file(a4c_tr_file)
    a4c_color的view是a4c_color_file或是a4c_color_mr_file(a4c_color_tr_file)
'''
MR_views = {'a2c_file', 'a2c_color_file', 'a3c_file',
            'a3c_color_file'}

TR_views = {'rv_inflow_file', 'rv_inflow_color_file', 'sax_ao_file',
            'sax_ao_color_file'}

a4c_MR_views = {'a4c_file', 'a4c_mr_file'}
a4c_TR_views = {'a4c_file', 'a4c_tr_file'}

a4c_color_MR_views = {'a4c_color_file', 'a4c_color_mr_file'}
a4c_color_TR_views = {'a4c_color_file', 'a4c_color_tr_file'}


def creat_folder(path):
    if os.path.isdir(path) == False:
        os.makedirs(path, mode=0o777)
        os.chmod(path, 0o777)


def resize_128_and_adjust(imgs):
    new_imgs = []

    for i in range(min(imgs.shape[0], 20)):
        x = cv2.resize(imgs[i], (128, 128), interpolation=cv2.INTER_CUBIC)
        new_imgs.append(x)

    new_imgs = np.array(new_imgs)
    while True:
        if new_imgs.shape[0] < 20:  # 如果frame不足20
            needed_frames = 20 - new_imgs.shape[0]
            new_imgs = np.vstack((new_imgs, new_imgs[0:needed_frames]))
        else:
            break

    new_imgs = np.expand_dims(new_imgs, 0) / 255
    return new_imgs


def preprocessing(file_paths, MR_TR):
    videos = dict()

    for view, file_path in file_paths.items():
        x = np.load(file_path)['arr_0']
        videos[view] = x

    if MR_TR == 'MR':

        a2c = resize_128_and_adjust(videos['a2c_file'][0:20, :, :, :])
        a2c_color = resize_128_and_adjust(
            videos['a2c_color_file'][0:20, :, :, :])
        a3c = resize_128_and_adjust(videos['a3c_file'][0:20, :, :, :])
        a3c_color = resize_128_and_adjust(
            videos['a3c_color_file'][0:20, :, :, :])
        if 'a4c_file' in videos:
            a4c = resize_128_and_adjust(videos['a4c_file'][0:20, :, :, :])
        else:
            a4c = resize_128_and_adjust(videos['a4c_mr_file'][0:20, :, :, :])

        if 'a4c_color_file' in videos:
            a4c_color = resize_128_and_adjust(
                videos['a4c_color_file'][0:20, :, :, :])
        else:
            a4c_color = resize_128_and_adjust(
                videos['a4c_color_mr_file'][0:20, :, :, :])

        X = [a2c, a2c_color, a3c, a3c_color, a4c, a4c_color]
    else:
        rv = resize_128_and_adjust(videos['rv_inflow_file'][0:20, :, :, :])
        rv_color = resize_128_and_adjust(
            videos['rv_inflow_color_file'][0:20, :, :, :])
        sax_ao = resize_128_and_adjust(videos['sax_ao_file'][0:20, :, :, :])
        sax_ao_color = resize_128_and_adjust(
            videos['sax_ao_file'][0:20, :, :, :])
        if 'a4c_file' in videos:
            a4c = resize_128_and_adjust(videos['a4c_file'][0:20, :, :, :])
        else:
            a4c = resize_128_and_adjust(videos['a4c_tr_file'][0:20, :, :, :])

        if 'a4c_color_file' in videos:
            a4c_color = resize_128_and_adjust(
                videos['a4c_color_file'][0:20, :, :, :])
        else:
            a4c_color = resize_128_and_adjust(
                videos['a4c_color_tr_file'][0:20, :, :, :])

        X = [rv, rv_color, sax_ao, sax_ao_color, a4c, a4c_color]
    return X


def load_model(MR_TR=''):
    if MR_TR == 'MR':
        model = get_model_for_MR()
    elif MR_TR == 'TR':
        model = asyncio.run(get_model_for_TR())
    else:
        raise Exception(f'[load_model] MR_TR no matched option => {MR_TR}')
    return model


def classify(folder_name, MR_TR):

    model = load_model(MR_TR)
    views = MR_views if MR_TR == 'MR' else TR_views

    file_paths = dict()
    for view in views:
        file_paths[view] = f'target_files/{folder_name}/{view}.npz'
        if os.path.exists(file_paths[view]) == False:
            raise Exception(f'{file_paths[view]} not existing')

    root_logger.debug(f'Load npz file: {file_paths}')
    inputted_imgs = preprocessing(file_paths, MR_TR)

    results = model.predict(inputted_imgs)

    return results


def make_task_lst_and_modify_MR_TR_views(view_lst):
    '''
    這邊修改MR_views和TR_views主要目的是為了把真正用到的a4c和a4c_color的file name加進去
    例如:
        a4c的view是a4c_file或是a4c_mr_file(a4c_tr_file)
        a4c_color的view是a4c_color_file或是a4c_color_mr_file(a4c_color_tr_file)
    '''
    MR_task = True
    TR_task = True

    def get_used_view(view_lst, view_lst2):
        '''
            view_lst2裡面的項目至少要有一個在view_lst裡面
            如果有回傳True，否則回傳False
        '''
        for view in view_lst2:
            if view in view_lst:
                return view
        return ''

    # ========[ MR ]===============
    for MR_view in MR_views:
        if MR_view not in view_lst:
            MR_task = False
            break

    if MR_task:
        used_a4c_name = get_used_view(
            view_lst, a4c_MR_views)
        used_a4c_color_name = get_used_view(
            view_lst, a4c_color_MR_views)

        if used_a4c_name == '' or used_a4c_color_name == '':
            MR_task = False

        MR_views.add(used_a4c_name)
        MR_views.add(used_a4c_color_name)

    # ========[ TR ]===============
    for TR_view in TR_views:
        if TR_view not in view_lst:
            TR_task = False
            break

    if TR_task:
        used_a4c_name = get_used_view(
            view_lst, a4c_TR_views)
        used_a4c_color_name = get_used_view(
            view_lst, a4c_color_TR_views)

        if used_a4c_name == '' or used_a4c_color_name == '':
            TR_task = False

        TR_views.add(used_a4c_name)
        TR_views.add(used_a4c_color_name)

    return MR_task, TR_task


def main():
    try:
        parser = argparse.ArgumentParser(description='')
        parser.add_argument("-f", help='The target of folder name', type=str)
        parser.add_argument(
            "-v", help='The target of views', type=str, nargs='+')
        args = parser.parse_args()

        root_logger.debug(f'args.f = {args.f}')
        root_logger.debug(f'args.v = {args.v}')
        root_logger.debug(f'dir_path = {dir_path}')

        if args.f is None:
            root_logger.error(f'args.f is None')
            raise Exception(f'args.f is None')

        if args.v is None:
            root_logger.error(f'args.v is None')
            raise Exception(f'args.v is None')

        MR_task, TR_task = make_task_lst_and_modify_MR_TR_views(args.v)

        results = dict()
        if MR_task:
            MR_result = classify(args.f, 'MR')
            MR_determination = 'Positive' if MR_result[0][0] > 0.22689362 else 'Negative'
            results['MR'] = [float(MR_result[0][0]), MR_determination]
            print(f'MR results = {results["MR"]}')

        if TR_task:
            TR_result = classify(args.f, 'TR')
            TR_determination = 'Positive' if TR_result[0][0] > 0.21836957 else 'Negative'
            results['TR'] = [float(TR_result[0][0]), TR_determination]
            print(f'TR results = {results["TR"]}')

        creat_folder(f'output/{args.f}/ensemble')
        json.dump(results, open(
            f'output/{args.f}/ensemble/ensemble_results.json', 'w'))

    except Exception as e:
        root_logger.error(e)
        raise Exception(e)


if __name__ == '__main__':
    main()
