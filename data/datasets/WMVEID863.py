from __future__ import division, print_function, absolute_import
import glob
import warnings
import os
import os.path as osp
from .bases import BaseImageDataset
import re
import random
# random.seed(10)

class WMVEID863(BaseImageDataset):

    dataset_dir = 'WMVEID863'

    def __init__(self, root='', verbose=True, **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # 普通数据
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.gallery_dir = os.path.join(self.dataset_dir, 'test')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        
        # 困难数据
        self.train_hard_dir = os.path.join(self.dataset_dir, 'train_hard')
        # self.gallery_hard_dir = os.path.join(self.dataset_dir, 'test_hard')
        # self.query_hard_dir = os.path.join(self.dataset_dir, 'query_hard')

        # query contains the first version sample
        # query1 contains cyclegan generated sample

        self.check_dir_exist()

        # self.train , self.num_train_pids , self.train_cam_num= self.get_data(self.train_dir, relabel=True)
        # self.train_img_num = len(self.train)
        # self.gallery, self.num_gallery_pids, self.gallery_cam_num = self.get_data(self.gallery_dir, relabel=False)
        # self.gallery_img_num = len(self.gallery)
        # self.query, self.num_query_pids ,self.query_cam_num= self.get_data(self.query_dir, relabel=False, ratio = 1)
        # self.query_img_num = len(self.query)

        self.train , self.num_train_pids , self.num_train_cams = self.get_data(self.train_dir, relabel=True)
        # self.train_hard , self.num_train_pids , self.num_train_cams = self.get_data(self.train_hard_dir, relabel=True)
        # self.train = self.train + self.train_hard

        self.train_img_num = len(self.train)
        self.num_train_scenes = 0
        self.gallery, self.num_gallery_pids, self.num_gallery_cams = self.get_data(self.gallery_dir, relabel=False)
        self.gallery_img_num = len(self.gallery)
        self.query, self.num_query_pids ,self.num_query_cams = self.get_data(self.query_dir, relabel=False, ratio = 1)
        self.query_img_num = len(self.query)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

        if verbose:
            self.print_statistics_info()

    def get_data(self, folder, relabel=False, ratio = 1):
        vids = os.listdir(folder)
        # pdb.set_trace()
        
        if ratio != 1:
            print('randomly sample ',ratio, 'ids for ttt')
            vids = random.sample(vids, int(len(vids)*ratio))
        labels = [int(vid) for vid in vids]

        if relabel:
            label_map = dict()
            for i, lab in enumerate(labels):
                label_map[lab] = i
        cam_set = set()
        img_info = []
        for vid in vids:
            id_vimgs = os.listdir(os.path.join(folder, vid, "vis"))
            id_nimgs = os.listdir(os.path.join(folder, vid, "ni"))
            # print(vid)
            id_timgs = os.listdir(os.path.join(folder, vid, "th"))
            for i, img in enumerate(id_vimgs):
                vpath = os.path.join(folder, vid, "vis", id_vimgs[i])
                npath = os.path.join(folder, vid, "ni", id_nimgs[i])
                tpath = os.path.join(folder, vid, "th", id_timgs[i])
                label = label_map[int(vid)] if relabel else int(vid)


                night = re.search('n+\d',img).group(0)[1]
                cam = re.search('v+\d',img).group(0)[1]
                cam = int(cam)
                night = int(night)
                cam_set.add(cam)
                img_info.append(([vpath, npath, tpath], label, cam, -1))

                # img_info_R.append((vpath, label, cam))
                # img_info_N.append((npath, label, cam))
                # img_info_T.append((tpath, label, cam))

        return img_info, len(vids), len(cam_set)

    def check_dir_exist(self):
        if not os.path.exists(self.root):
            raise Exception('Error path: {}'.format(self.root))
        if not os.path.exists(self.train_dir):
            raise Exception('Error path:{}'.format(self.train_dir))
        if not os.path.exists(self.gallery_dir):
            raise Exception('Error path:{}'.format(self.gallery_dir))
        if not os.path.exists(self.query_dir):
            raise Exception('Error path:{}'.format(self.query_dir))

    def print_statistics_info(self):
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras ")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, len(self.train)*3, self.num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, len(self.query)*3, self.num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, len(self.gallery)*3, self.num_gallery_cams))
        print("  ----------------------------------------")


if __name__ == "__main__":
    root = './data'
    dataset = WMVEID863(root, True)
    # train_data = dataset.train_data
    # for i  in train_data:
    #     print(i)