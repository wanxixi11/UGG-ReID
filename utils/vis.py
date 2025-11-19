import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
from config import cfg
import os
from collections import Counter
from einops import rearrange
import matplotlib.image as mpimg
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import seaborn as sns
import torch.nn.functional as F
import random

diy_color_map= [
        (0.7, 0.9, 0.4, 1.0),#r, g, b
        (0.5, 0.5, 0.5, 1.0),
        (1.0, 0.6, 0.1, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 0.1, 0.7, 1.0),
        # (0.9, 0.2, 0.4, 1.0),
        (0.4, 0.2, 0.9, 1.0),
        (0.8, 0.2, 1.0, 1.0),
        # (0.2, 0.4, 0.8, 1.0),
        (0.8, 0.3, 0.2, 1.0),
        (0.7, 0.5, 0.3, 1.0),
        
        (0.1, 0.1, 0.1, 1.0),
        (0.7, 0.3, 0.8, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.4, 1.0, 0.6, 1.0),
        (0.3, 0.8, 0.5, 1.0),
        (0.1, 0.8, 1.0, 1.0),
        (0.5, 0.7, 0.9, 1.0),
        (0.4, 0.8, 0.3, 1.0),
        (0.5, 0.7, 0.4, 1.0),
        (0.2, 0.6, 0.8, 1.0),
        (0.1, 0.1, 1.0, 1.0),
        (0.3, 0.3, 0.9, 1.0),
        (0.6, 0.1, 0.4, 1.0),
        ]#R G B

diy_color_map_icpl= [
        (0.7, 0.9, 0.4, 1.0),#1  r, g, b
        (0.5, 0.5, 0.5, 1.0),#2
        (1.0, 0.6, 0.1, 1.0),#3
        (1.0, 0.0, 0.0, 1.0),#4
        (1.0, 0.1, 0.7, 1.0),#5
        # (0.9, 0.2, 0.4, 1.0),
        (0.4, 0.2, 0.9, 1.0),#6
        (0.8, 0.2, 1.0, 1.0),#7
        # (0.2, 0.4, 0.8, 1.0),
        (0.8, 0.3, 0.2, 1.0),#8
        (0.7, 0.5, 0.3, 1.0),#9
        (0.3, 0.3, 0.9, 1.0),#10
        (0.7, 0.3, 0.8, 1.0),#11
        (0.0, 0.8, 0.0, 0.8),#12
        (0.4, 1.0, 0.6, 1.0),
        (0.3, 0.8, 0.5, 1.0),
        (0.1, 0.8, 1.0, 1.0),
        (0.5, 0.7, 0.9, 1.0),
        (0.4, 0.8, 0.3, 1.0),
        (0.5, 0.7, 0.4, 1.0),
        (0.2, 0.6, 0.8, 1.0),
        (0.1, 0.1, 1.0, 1.0),
        (0.3, 0.3, 0.9, 1.0),
        (0.6, 0.1, 0.4, 1.0),
        ]#R G B
def tsne_vis(seed_idx, all_features, all_labels, 
                    epoch, modal, mAP, all_AP=None, q_pids=None,
                    stage=2, config=None, 
                    fixed_marker_shape=None, 
                    multi_modal=False, multi_text=False, modal_num=3, cls_num=30, filter_type='TopK', 
                    text_features=None, text_labels=None, 
                    tsne_global=True, dataset='train', selected_classes=None):
    '''
        在选定30个要可视化的类别之前做所有特征的tsne降维，这样来获取全局视角下的特征。
    '''
    if config.DATASETS.NAMES == 'RGBNT201':
        all_features = all_features[:len(all_AP)]
        all_labels = all_labels[:len(all_AP)]
        # 283, 279, 294, 286
        # 283, 279, 294, 295, 275, 258
        # 283, 294, 284, 279, 282, 291, 290, 285
        # 283, 294, 284, 279, 282, 291, 285, 278
        # 283, 294, 284, 279, 282, 291, 285, 278, 271, 274
        # 283, 284, 279, 282, 278, 271, 274
        # 283, 294, 284, 279, 282, 278, 271, 274
        # 283, 284, 279, 282, 285, 278, 271, 274
        # selected_classes = np.array([285, 284, 283, 279, 278, 277])
        selected_classes = np.array([282, 284, 279, 283, 285, 278, 271, 274]) # DEEP
        # selected_classes = np.array([292, 272, 283, 299, 278, 277, 280, 274, 258, 285, 260, 294]) # ICPL
        selected_classes = np.array([292, 272, 283, 278, 299, 280, 274, 258, 285, 260, 294, 277]) # ICPL
        # q_pids_ = set(q_pids)
        # selected_classes = random.sample(list(q_pids_), 12)
        
    elif config.DATASETS.NAMES == 'MSVR310':
        all_features = all_features[len(all_AP):]
        all_labels = all_labels[len(all_AP):]
        # 90, 253, 92, 69, 220, 145 
        # 92, 69, 220 24, 
        # 92, 69, 220, 90, 75, 24, 26, 20, 59, 75, 331, 327, 122
        # selected_classes = np.array([196, 116, 90, 160, 122, 127, 204, 241, 26, 253])
        # selected_classes = np.array([204, 241, 26, 253, 220, 22, 100, 228])
        # selected_classes = np.array([92, 69, 220, 90, 75, 24, 26, 20, 59, 122, 160, 220])
        q_pids_ = set(q_pids)
        selected_classes = random.sample(list(q_pids_), 12)
    logger = logging.getLogger("transreid.image_train")
    # Count the occurrences of each class label
    label_counter = Counter(all_labels)
    # filter_type = 'HardK' if all_AP is not None else 'TopK'
    
    # 定义所需的类别数量和每个类别的最大样本数
    max_samples_per_class = 15
    cls_num = 30
    filter_type = 'HardK'
    if selected_classes is None:
        if filter_type == 'TopK':
            # Get the top 30 most common classes with their occurrences
            top_30_classes = label_counter.most_common(cls_num)
            top_30_class_labels = [label for label, count in top_30_classes]
            selected_classes = np.array(top_30_class_labels)
        elif filter_type == 'RangeK':
            selected_labels = [label for label, count in label_counter.items() if 10 <= count <= 40]
            selected_classes = np.array(selected_labels[:cls_num])
        elif filter_type == 'HardID':
            # Step 1: Calculate AP for each unique label in all_labels
            ap_per_class = {}
            all_querys = all_labels[:len(all_AP)]
            for label in np.unique(all_querys):
                indices = np.where(all_querys == label)[0]
                ap_per_class[label] = np.mean([all_AP[i] for i in indices])
            sorted_ap_classes = sorted(ap_per_class.items(), key=lambda x: x[1])
            logger.info(sorted_ap_classes)
            selected_classes = [item[0] for item in sorted_ap_classes[:cls_num]]
        elif filter_type == 'HardK':
            # 对all_AP进行排序，并获取排序后的索引
            sorted_indices = np.argsort(all_AP)
            all_AP_ = np.array(all_AP)
            sorted_all_AP = all_AP_[sorted_indices]
            # 找出前cls_num个最小值对应的标签，并确保标签不重复
            selected_classes = []
            front_val = None
            for val in sorted_all_AP:
                if front_val is not None:
                    if front_val == val:
                        continue
                front_val = val
                index = np.where(all_AP == val)[0]
                if len(selected_classes) < cls_num:
                    if q_pids[index][0] not in selected_classes:
                        selected_classes.append(q_pids[index][0])
                        logger.info(f'HardK val: {q_pids[index][0]} {val}')
                else:
                    break
        elif filter_type == 'EasyK':
            # 对all_AP进行排序，并获取排序后的索引
            sorted_indices = np.argsort(all_AP)
            all_AP_ = np.array(all_AP)
            sorted_all_AP = all_AP_[sorted_indices]

            # 逆序遍历sorted_all_AP数组  
            reverse_sorted_indices = np.flip(sorted_indices)  
            reverse_sorted_all_AP = all_AP_[reverse_sorted_indices] 

            # 找出前cls_num个最小值对应的标签，并确保标签不重复
            selected_classes = []
            front_val = None
            for val in reverse_sorted_all_AP:
                if front_val is not None:
                    if front_val == val:
                        continue
                front_val = val
                index = np.where(all_AP == val)[0]
                if len(selected_classes) < cls_num:
                    if q_pids[index][0] not in selected_classes:
                        selected_classes.append(q_pids[index][0])
                else:
                    break
        # print(top_30_class_labels)
    logger.info(f'HardK: {selected_classes}')
    # 在filter之前降维能获得特征的全局分布视角
    if tsne_global:
        selected_features = np.array(all_features)
        selected_labels_all = np.array(all_labels)
        # 从文本模态中筛选出要可视化的特征向量和标签
        if text_features is not None:
            selected_text_features = np.array(text_features)
            selected_labels_text_all = np.array(text_labels)
    else:
        # 用于记录每个类别的样本数
        class_sample_count = {label: 0 for label in selected_classes}

        # 筛选出固定的20个类别对应的特征向量和标签
        selected_features = []
        selected_labels_all = []
        for label, feat in zip(all_labels, all_features):
            # 检查标签是否在所选类别中，并且样本数量是否未达到最大限制
            if label in selected_classes and class_sample_count[label] < max_samples_per_class:
                selected_features.append(feat)
                selected_labels_all.append(label)
                class_sample_count[label] += 1  # 增加类别样本计数
        
        if text_features is not None:
            selected_text_features = []
            selected_labels_text_all = []
            for label, feat in zip(text_labels, text_features):
                if label in selected_classes:
                    selected_text_features.append(feat)
                    selected_labels_text_all.append(label)

    # 如果是多模态的还要把特征向量拆开，并以送入Batch维的方式进行展示
    if multi_modal:
        selected_features = rearrange(selected_features, 'B (C D) -> (B C) D', C=modal_num)
    if multi_text:
        selected_text_features = rearrange(selected_text_features, 'B (C D) -> (B C) D', C=modal_num)
    
    # 把文本模态拼接到多模态特征的末端，两个不同尺度的模态一同降维时需要先进行模态的归一化
    if text_features is not None:
        # 假设 selected_features 和 selected_text_features 是您的两组特征数据
        scaler = StandardScaler()
        normalized_selected_features = scaler.fit_transform(selected_features)
        normalized_selected_text_features = scaler.fit_transform(selected_text_features)
        selected_features = np.concatenate((normalized_selected_features, normalized_selected_text_features), axis=0)
    
    # # 使用PCA进行降维
    # pca = PCA(n_components=50)  # 假设目标降维后维度为50，你可以根据需要调整该值
    # selected_features_pca = pca.fit_transform(selected_features)

    # # 使用T-SNE进行降维
    # tsne = TSNE(n_components=2, random_state=42)
    # selected_features_tsne = tsne.fit_transform(selected_features_pca)
    
    selected_features = np.array(selected_features)
    # tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    selected_features_tsne = tsne.fit_transform(selected_features)

    # 降维结束后，再把文本模态取出来
    if text_features is not None:
        text_feats_len = len(selected_text_features)
        selected_text_features_tsne = selected_features_tsne[-text_feats_len:,:]
        selected_features_tsne = selected_features_tsne[:-text_feats_len,:]

    # 同时多模态的特征拆分到C维(模态维)
    if multi_modal:
        selected_features_tsne = rearrange(selected_features_tsne, '(B C) D -> B C D', C=modal_num)
    if multi_text:
        selected_text_features_tsne = rearrange(selected_text_features_tsne, '(B C) D -> B C D', C=modal_num)
    
    if tsne_global:
        # 筛选出固定的20个类别对应的特征向量和标签
        selected_features = []
        selected_labels_all = []
        for label, feat in zip(all_labels, selected_features_tsne):
            if label in selected_classes:
                selected_features.append(feat)
                selected_labels_all.append(label)
        selected_features_tsne = np.array(selected_features)

        if text_features is not None:
            # 文本模态里筛选出来固定数量的特征
            selected_text_features = []
            selected_labels_text_all = []
            for label, feat in zip(text_labels, selected_text_features_tsne):
                if label in selected_classes:
                    selected_text_features.append(feat)
                    selected_labels_text_all.append(label)
            selected_text_features_tsne =np.array(selected_text_features)

    # 创建T-SNE可视化的散点图
    x_min, x_max = np.min(selected_features_tsne, 0), np.max(selected_features_tsne, 0)
    selected_features_tsne = (selected_features_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(5, 4),dpi=300)
    # plt.figure(figsize=(15, 12), dpi=100)
    num_classes = len(selected_classes)
    color_map = cm.get_cmap('tab20', num_classes)
    # 设置不同的marker shape
    # marker_shapes = ['o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', 'H', 'd', 'p', '8', 'h', '4', '_', '|', '.'][:5]
    if multi_modal:
        marker_shapes = ['o', 'x', '^']
    else:
        # marker_shapes = ['o', 'x']
        marker_shapes = ['*']
    text_marker_shapes = ['*', '+', 'x']
    text_marker_size = [200, 150, 100]

    for i, label in enumerate(selected_classes):
        mask = selected_labels_all == label
        # 如果是多模态的，就按照每个模态一种形状、每个ID一个颜色的格式进行展示
        if multi_modal:
            for j in range(modal_num):
                marker_shape = marker_shapes[j]
                modal_tag = ['rgb', 'nir', 'tir'][j]
                plt.scatter(selected_features_tsne[mask, j, 0], selected_features_tsne[mask, j, 1], label=modal_tag, color=diy_color_map[i%len(diy_color_map)], marker=marker_shape, alpha=0.7, s=10)
        else:
            # 如果是单模态的tsne，使用颜色和形状分辨不同的ID
            if fixed_marker_shape is None:
                marker_shape = marker_shapes[i % len(marker_shapes)]  # 循环使用不同的marker shape
            else:
                marker_shape = fixed_marker_shape
            # plt.scatter(selected_features_tsne[mask, 0], selected_features_tsne[mask, 1], label=f'ID: {label}', color=color_map(i), marker=marker_shape, alpha=0.7, s=20)
            # plt.scatter(selected_features_tsne[mask, 0], selected_features_tsne[mask, 1], label=f'ID: {label}', color=color_map(i), marker=marker_shape, s=10)
            # plt.scatter(selected_features_tsne[mask, 0], selected_features_tsne[mask, 1], label=f'ID: {label}', color=color_map(i), marker=marker_shape, alpha=0.7, s=10)
            plt.scatter(selected_features_tsne[mask, 0], selected_features_tsne[mask, 1], label=f'ID: {label}', color=diy_color_map_icpl[i%len(diy_color_map_icpl)], marker=marker_shape, alpha=0.6, s=80,edgecolor='none')
            # plt.scatter(selected_features_tsne[mask, 0], selected_features_tsne[mask, 1], label=f'ID: {label}', color=color_map(i), marker=marker_shape, alpha=0.7, s=40,edgecolor='none')
        
        # 有文本模态就对其进行可视化
        if text_features is not None:
            mask = selected_labels_text_all == label
            if multi_text:
                for j in range(modal_num):
                    text_marker_shape = text_marker_shapes[j]
                    text_modal_tag = ['rgb', 'nir', 'tir'][j]
                    original_color = color_map(i)
                    darkened_color = (max(original_color[0] - 0.12, 0.), max(original_color[1] - 0.12, 0.), max(original_color[2] - 0.12, 0.), original_color[3])
                    plt.scatter(selected_text_features_tsne[mask, j, 0], selected_text_features_tsne[mask, j, 1], label=text_modal_tag, color=darkened_color, marker=text_marker_shape, s=text_marker_size[j])
            else:
                original_color = color_map(i)
                darkened_color = (max(original_color[0] - 0.12, 0.), max(original_color[1] - 0.12, 0.), max(original_color[2] - 0.12, 0.), original_color[3])
                plt.scatter(selected_text_features_tsne[mask, 0], selected_text_features_tsne[mask, 1], label=f'Text ID: {label}', color=darkened_color, marker=text_marker_shapes[0], s=200)

    if multi_modal:
        legend_elements = [
            Line2D([], [], marker=marker_shapes[0], color='w', markerfacecolor=color_map(1), markersize=5, label='RGB', alpha=0.7),
            Line2D([], [], marker=marker_shapes[1], color='w', markerfacecolor=color_map(2), markersize=5, label='NIR', alpha=0.7),
        ]
        if modal_num >2:
            legend_elements.append(Line2D([], [], marker=marker_shapes[2], color='w', markerfacecolor=color_map(3), markersize=5, label='TIR', alpha=0.7))
        if text_features is not None:
            if multi_text:
                legend_elements.append(Line2D([], [], marker=text_marker_shapes[0], color='w', markerfacecolor=color_map(1), markersize=15, label='Text_rgb'))
                legend_elements.append(Line2D([], [], marker='P', color='w', markerfacecolor=color_map(2), markersize=10, label='Text_nir'))
                if modal_num >2:
                    legend_elements.append(Line2D([], [], marker='X', color='w', markerfacecolor=color_map(3), markersize=10, label='Text_tir'))
            else:
                legend_elements.append(Line2D([], [], marker=text_marker_shapes[0], color='w', markerfacecolor=color_map(1), markersize=15, label='Text'))
        # plt.legend(handles=legend_elements, loc='upper right')
    show_id_label = False
    if show_id_label:
        plt.legend(loc='upper right')
        plt.xlabel('T-SNE Dimension 1')
        plt.ylabel('T-SNE Dimension 2')
        plt.title('T-SNE Visualization of Features')
    plt.xticks([])
    plt.yticks([])
    if config is None:
        root = cfg.OUTPUT_DIR
    else:
        root = config.OUTPUT_DIR

    folder_path = os.path.join(root, 'tsne', dataset, f'stage_{stage}', 'global' if tsne_global else 'local', filter_type, 'text' if text_features is not None else 'no_text')
    os.makedirs(folder_path, exist_ok=True)  # 创建文件夹
    # output_pth_1 = os.path.join(folder_path, f'{modal}_{epoch}_{mAP:.1%}.png')
    output_pth_2 = os.path.join(folder_path, f'{modal}_{epoch}_{mAP:.1%}_{seed_idx}.pdf')
    # output_pth_3 = os.path.join(folder_path, f'0a_{modal}.png')
    output_pth_4 = os.path.join(folder_path, f'0a_{modal}_{seed_idx}.pdf')
    # plt.savefig(output_pth_1)
    # plt.savefig(output_pth_2, format='png')
    # plt.savefig(output_pth_3)
    plt.savefig(output_pth_4, format='pdf')
    # plt.show()
    plt.close()


def res_vis(q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids, distmat, all_AP, q_img_paths, g_img_paths, epoch, cfg):
    # 可视化样本结果
    sorted_indices = np.argsort(all_AP)
    # AP最低的20个样本
    # hard_idx = sorted_indices[:20]
    # # AP最高、最简单的20个样本
    # easy_idx = sorted_indices[-20:]
    # selected_idx = np.concatenate((hard_idx, easy_idx))
    # 可视化所有样本
    hard_idx = sorted_indices
    selected_idx = hard_idx

    num_q, num_g = distmat.shape
    max_rank = 20
    indices = np.argsort(distmat, axis=1) # 其中包含了对于每个查询样本，按照与其距离从近到远的顺序排列的画廊样本的索引
    
    visible_img = True

    # 使用花式索引获取路径
    # paths = np.array(g_img_paths)[indices]

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    output_path = 'res_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'res_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset_name = cfg.DATASETS.NAMES
    for j, q_idx in enumerate(selected_idx):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx] # 当前query所匹配的结果
        
        if dataset_name in ['RGBNT100', 'RGBN300', 'RGBNT201', 'WMVEID863']:
            # original protocol in RGBNT100 or RGBN300
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        elif dataset_name in ['MSVR310']:
            # new protocol in MSVR310
            q_sceneid = q_sceneids[q_idx]
            remove = (g_pids[order] == q_pid) & (g_sceneids[order] == q_sceneid)

        keep = np.invert(remove)
        v_id_labels = g_pids[order][keep][:max_rank] == q_pid
        paths = np.array(g_img_paths)[order][keep][:max_rank]
        
        # 设置每行显示的图片数量
        if dataset_name == 'RGBNT201':
            num_images_per_row = 5
        else:
            num_images_per_row = 10

        # 计算行数
        num_rows = len(paths) // num_images_per_row
        if len(paths) % num_images_per_row != 0:
            num_rows += 1

        q_img_path_vis = q_img_paths[q_idx]
        if dataset_name == 'RGBNT201':
            q_img_path_arr = q_img_path_vis.split('/')
            q_img_path_ni = os.path.join('/'.join(q_img_path_arr[:-2]),'NI','/'.join(q_img_path_arr[-1:]))
            q_img_path_th = os.path.join('/'.join(q_img_path_arr[:-2]),'TI','/'.join(q_img_path_arr[-1:]))
        else:
            q_img_path_ni = q_img_path_vis.replace('vis', 'ni')
            q_img_path_th = q_img_path_vis.replace('vis', 'th')

        if visible_img:
            if cfg.DATASETS.NAMES not in ['RGBNT100', 'RGBN300']:
                q_img_vis = mpimg.imread(q_img_path_vis)
                q_img_ni = mpimg.imread(q_img_path_ni)
                q_img_th = mpimg.imread(q_img_path_th)
            else:
                q_img = mpimg.imread(q_img_path_vis)

                # 裁剪图片的区域 (top:bottom, left:right)
                q_img_vis = q_img[0:128, 0:256]
                q_img_ni = q_img[0:128, 256:512]
                q_img_th = q_img[0:128, 512:768]

            # 获取最大的宽度和高度
            if dataset_name == 'RGBNT201':
                max_width = 128
                max_height = 256
            else:
                max_width = 256
                max_height = 128

            # 使用 cv2.resize 调整图像尺寸
            img_vis_resized = cv2.resize(q_img_vis, (max_width, max_height))
            img_ni_resized = cv2.resize(q_img_ni, (max_width, max_height))
            img_th_resized = cv2.resize(q_img_th, (max_width, max_height))
            # 将三张图片竖向拼接
            if dataset_name == 'RGBNT201':
                q_concatenated_img = np.concatenate((img_vis_resized, img_ni_resized, img_th_resized), axis=1)
            else:
                q_concatenated_img = np.concatenate((img_vis_resized, img_ni_resized, img_th_resized), axis=0)

        # 遍历每个路径，加载并显示图片
        fig, axes = plt.subplots(num_rows, num_images_per_row + 1, figsize=(15, 3 * num_rows))
        
        if visible_img:
            # 在第一列添加 q_concatenated_img
            axes[0, 0].imshow(q_concatenated_img)
        if dataset_name == 'WMVEID863':
            q_name = q_img_paths[q_idx].split('/')[-1][:-4].split('_')
            q_name = q_img_paths[q_idx].split('/')[-3]+'_'+q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif dataset_name == 'MSVR310':
            q_name = q_img_paths[q_idx].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif dataset_name == 'RGBNT201':
            q_name = q_img_paths[q_idx].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'\n'+q_name[1]+'_'+q_name[2]+'_'+q_name[3]
        elif dataset_name == 'RGBNT100':
            q_name = q_img_paths[q_idx].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'_'+q_name[1]+'\n'+q_name[2]
        axes[0, 0].set_title(q_name, color='brown')
        axes[0, 0].axis('off')

        # 竖向拼接
        for i, (path, label) in enumerate(zip(paths, v_id_labels)):
            
            if visible_img:
                g_img_path_vis = str(path)
                if dataset_name == 'RGBNT201':
                    g_img_path_arr = g_img_path_vis.split('/')
                    g_img_path_ni = os.path.join('/'.join(g_img_path_arr[:-2]),'NI','/'.join(g_img_path_arr[-1:]))
                    g_img_path_th = os.path.join('/'.join(g_img_path_arr[:-2]),'TI','/'.join(g_img_path_arr[-1:]))
                else:
                    g_img_path_ni = g_img_path_vis.replace('vis', 'ni')

                    g_img_path_th = g_img_path_vis.replace('vis', 'th')

                if cfg.DATASETS.NAMES not in ['RGBNT100', 'RGBN300']:
                    img_vis = mpimg.imread(g_img_path_vis)
                    img_ni = mpimg.imread(g_img_path_ni)
                    img_th = mpimg.imread(g_img_path_th)
                else:
                    g_img = mpimg.imread(g_img_path_vis)

                    # 裁剪图片的区域 (top:bottom, left:right)
                    img_vis = g_img[0:128, 0:256]
                    img_ni = g_img[0:128, 256:512]
                    img_th = g_img[0:128, 512:768]

                if dataset_name == 'RGBNT201':
                    max_width = 128
                    max_height = 256
                else:
                    max_width = 256
                    max_height = 128

                # 使用 cv2.resize 调整图像尺寸
                img_vis_resized = cv2.resize(img_vis, (max_width, max_height))
                img_ni_resized = cv2.resize(img_ni, (max_width, max_height))
                img_th_resized = cv2.resize(img_th, (max_width, max_height))

                # 将三张图片竖向拼接
                if dataset_name == 'RGBNT201':
                    concatenated_img = np.concatenate((img_vis_resized, img_ni_resized, img_th_resized), axis=1)
                else:
                    concatenated_img = np.concatenate((img_vis_resized, img_ni_resized, img_th_resized), axis=0)

            row, col = divmod(i, num_images_per_row)
            
            # 如果 v_id_labels 为 True，边框颜色为绿色，否则为红色
            border_color = 'green' if label else 'red'

            g_name = path.split('/')[-1][:-4].split('_')
            if dataset_name == 'WMVEID863':
                g_name = path.split('/')[-1][:-4].split('_')
                g_name = path.split('/')[-3]+'_'+g_name[0]+'_'+g_name[1]+'\n'+g_name[2]+'_'+g_name[3]
            elif dataset_name == 'MSVR310':
                g_name = path.split('/')[-1][:-4].split('_')
                g_name = g_name[0]+'_'+g_name[1]+'\n'+g_name[2]+'_'+g_name[3]
            elif dataset_name == 'RGBNT201':
                g_name = path.split('/')[-1][:-4].split('_')
                g_name = g_name[0]+'\n'+g_name[1]+'_'+g_name[2]+'_'+g_name[3]
            elif dataset_name == 'RGBNT100':
                g_name = path.split('/')[-1][:-4].split('_')
                g_name = g_name[0]+'_'+g_name[1]+'\n'+g_name[2]
            if visible_img:
                axes[row, col+1].imshow(concatenated_img)
            axes[row, col+1].set_title(g_name, color=border_color)
            axes[row, col+1].axis('off')
        
        # 删除第一列的多余 subplot
        for l in range(1, num_rows):
            fig.delaxes(axes[l, 0])

        # level_dir = 'hard' if j < 20 else 'easy'
        level_dir = 'all'
        out_dir = os.path.join(output_path, str(epoch), level_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        q_name = q_name.replace('\n','_')
        ap = all_AP[q_idx]
        # plt.savefig(os.path.join(out_dir, q_name+'_'+"{:.1%}".format(ap)+'.png'))
        plt.savefig(os.path.join(out_dir, q_name+'_'+"{:.1%}".format(ap)+'.pdf'), format='pdf')



def dist_vis_curve(q_pids, g_pids, qf, gf, epoch, cfg, eval_type='all'):
    query_feature = torch.FloatTensor(qf)
    query_label = torch.FloatTensor(q_pids)
    gallery_feature = torch.FloatTensor(gf)
    gallery_label = torch.FloatTensor(g_pids)

    query_feature = query_feature.detach().cpu().numpy()
    gallery_feature = gallery_feature.detach().cpu().numpy()

    # Normalize features
    query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)
    gallery_feature = gallery_feature / np.linalg.norm(gallery_feature, axis=1, keepdims=True)

    # Calculate mask and distance matrix
    mask = query_label.expand(len(gallery_label), len(query_label)).eq(
        gallery_label.expand(len(query_label), len(gallery_label)).t()).cuda()
    distmat = torch.FloatTensor(1 - np.matmul(gallery_feature, np.transpose(query_feature))).cuda() 

    # Extract intra-class and inter-class distances
    intra = distmat[mask]
    inter = distmat[mask == 0]

    ######################################################################
    fig, ax = plt.subplots()

    # Plot density curves instead of histograms
    sns.kdeplot(intra.detach().cpu().numpy(), color="blue", fill=True, alpha=0.6, label="Intra-class", ax=ax)
    sns.kdeplot(inter.detach().cpu().numpy(), color="green", fill=True, alpha=0.6, label="Inter-class", ax=ax)

    # Set labels and legend
    ax.set_xlabel('Feature Distance')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_xlim(0.01, 1.5)  # Adjust x-axis range as needed
    ax.set_ylim(0, 7)  # Adjust vertical limit if needed

    # Output path and saving
    output_path = 'dist_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'dist_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_dir = os.path.join(output_path, str(epoch))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, f'dist_vis_{eval_type}.pdf'), format='pdf')
    plt.show()


def dist_vis(q_pids, g_pids, qf, gf, epoch, cfg, eval_type='all'):
    query_feature = torch.FloatTensor(qf)
    query_label = torch.FloatTensor(q_pids)
    gallery_feature = torch.FloatTensor(gf)
    gallery_label = torch.FloatTensor(g_pids)

    query_feature = query_feature.detach().cpu().numpy()
    gallery_feature = gallery_feature.detach().cpu().numpy()

    query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)
    gallery_feature = gallery_feature / np.linalg.norm(gallery_feature, axis=1, keepdims=True)

    mask = query_label.expand(len(gallery_label), len(query_label)).eq(gallery_label.expand(len(query_label), len(gallery_label)).t()).cuda()

    #Cosine distance 余弦相似度的范围在[-1, 1]之间。此处 1 - matmul 之后，距离的范围则在[0, 2]之间。
    distmat = torch.FloatTensor(1 - np.matmul(gallery_feature, np.transpose(query_feature))).cuda() 
    
    intra = distmat[mask]
    inter = distmat[mask == 0]

    ######################################################################
    # plt.rcParams.update({'font.size': 14})


    fig, ax = plt.subplots()
    # b = np.linspace(0.001, 1.5, num=1000)
    b = np.linspace(0.01, 1.5, num=1000)

    ax.hist(intra.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color = 'blue', density=True, label='Intra-class')
    ax.hist(inter.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color = 'green', density=True, label='Inter-class')

    # # 计算类内距离的均值
    # intra_mean = intra.mean().item()
    # inter_mean = inter.mean().item()
    # # 在直方图上添加均值线
    # ax.axvline(intra_mean, color='blue', linestyle='--', linewidth=1.5, label='Intra-class Mean')
    # ax.axvline(inter_mean, color='green', linestyle='--', linewidth=1.5, label='Inter-class Mean')

    # # 计算均值间的距离
    # distance = abs(inter_mean - intra_mean)

    # # 在图上绘制均值之间的距离线和距离值
    # y_pos = ax.get_ylim()[1] * 0.9  # 设置距离线的高度，稍低于图的顶部
    # ax.annotate(f'Distance: {distance:.2f}', xy=((intra_mean + inter_mean) / 2, y_pos + 0.05), ha='center', color='red')
    # ax.plot([intra_mean, inter_mean], [y_pos, y_pos], color='red', linestyle='-', linewidth=1.5)


    ax.set_xlabel('Feature Distance')
    ax.set_ylabel('Density')
    ax.legend()

    ax.set_ylim(0, 7)  # 设置纵坐标范围，调整 0.1 为适合的最大值

    output_path = 'dist_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'dist_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_dir = os.path.join(output_path, str(epoch))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, f'dist_vis_{eval_type}.pdf'), format='pdf')
    plt.show()

def modal_sim_vis(all_old_rgb, all_old_nir, all_old_tir, all_rgb, all_nir, all_tir, cfg):
    split_idx = 836
    # all_old_rgb = torch.cat(all_old_rgb, dim=0)[:split_idx, 0].unsqueeze(1)
    # all_old_nir = torch.cat(all_old_nir, dim=0)[:split_idx, 1:]
    # all_old_tir = torch.cat(all_old_tir, dim=0)[:split_idx, 1:]

    # all_rgb = torch.cat(all_rgb, dim=0)[:split_idx, 0].unsqueeze(1)
    # all_nir = torch.cat(all_nir, dim=0)[:split_idx, 1:]
    # all_tir = torch.cat(all_tir, dim=0)[:split_idx, 1:]
    
    all_old_rgb = torch.cat(all_old_rgb, dim=0)[:split_idx, 0]
    all_old_nir = torch.cat(all_old_nir, dim=0)[:split_idx, 0]
    all_old_tir = torch.cat(all_old_tir, dim=0)[:split_idx, 0]

    all_rgb = torch.cat(all_rgb, dim=0)[:split_idx, 0]
    all_nir = torch.cat(all_nir, dim=0)[:split_idx, 0]
    all_tir = torch.cat(all_tir, dim=0)[:split_idx, 0]

    cos_rgb_nir_old = F.cosine_similarity(all_old_rgb, all_old_nir, dim=-1)  # 输出形状 [836, 128]
    cos_rgb_tir_old = F.cosine_similarity(all_old_rgb, all_old_tir, dim=-1)  # 输出形状 [836, 128]

    cos_rgb_nir = F.cosine_similarity(all_rgb, all_nir, dim=-1)  # 输出形状 [836, 128]
    cos_rgb_tir = F.cosine_similarity(all_rgb, all_tir, dim=-1)  # 输出形状 [836, 128]

    ######################################################################
    # 将相似度张量平铺为一维
    similarity_rgb_nir_old_flat = cos_rgb_nir_old.flatten().cpu().numpy()
    similarity_rgb_tir_old_flat = cos_rgb_tir_old.flatten().cpu().numpy()
    similarity_rgb_nir_flat = cos_rgb_nir.flatten().cpu().numpy()
    similarity_rgb_tir_flat = cos_rgb_tir.flatten().cpu().numpy()

    # 设置直方图参数
    bins = np.linspace(0.7, 1.0, num=1000)  # 设置直方图的柱数
    alpha = 0.5  # 设置透明度
    plt.figure(figsize=(10, 6))

    # 绘制 NIR 和 RGB class token 相似度的直方图
    plt.hist(similarity_rgb_nir_old_flat, bins=bins, alpha=alpha, color='blue', label='RGB-NIR OLD Similarity')
    # 绘制 TIR 和 RGB class token 相似度的直方图
    # plt.hist(similarity_rgb_tir_old_flat, bins=bins, alpha=alpha, color='orange', label='RGB-TIR Similarity')

    plt.hist(similarity_rgb_nir_flat, bins=bins, alpha=alpha, color='green', label='RGB-NIR Similarity')
    # plt.hist(similarity_rgb_tir_flat, bins=bins, alpha=alpha, color='green', label='RGB-TIR Similarity')

    # 添加图例和标题
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Cosine Similarity Distribution between RGB Class Token and NIR/TIR Patch Tokens")
    plt.legend()

    output_path = 'modal_sim_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'modal_sim_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_dir = os.path.join(output_path, str(0))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, f'modal_sim_vis_{0}.pdf'), format='pdf')
    plt.show()

def modal_sim_vis_cuver(all_old_rgb, all_old_nir, all_old_tir, all_rgb, all_nir, all_tir, cfg):
    if cfg.DATASETS.NAMES == 'RGBNT201':
        split_idx = 836
        all_old_rgb_patchs = torch.cat(all_old_rgb, dim=0)[:split_idx, 1:]
        all_old_rgb = torch.cat(all_old_rgb, dim=0)[:split_idx, 0].unsqueeze(1)
        all_old_nir = torch.cat(all_old_nir, dim=0)[:split_idx, 1:]
        all_old_tir = torch.cat(all_old_tir, dim=0)[:split_idx, 1:]

        all_rgb = torch.cat(all_rgb, dim=0)[:split_idx, 0].unsqueeze(1)
        all_nir = torch.cat(all_nir, dim=0)[:split_idx, 1:]
        all_tir = torch.cat(all_tir, dim=0)[:split_idx, 1:]
    else:
        split_idx = 591
        all_old_rgb = torch.cat(all_old_rgb, dim=0)[split_idx:, 0].unsqueeze(1)
        all_old_nir = torch.cat(all_old_nir, dim=0)[split_idx:, 1:]
        all_old_tir = torch.cat(all_old_tir, dim=0)[split_idx:, 1:]

        all_rgb = torch.cat(all_rgb, dim=0)[split_idx:, 0].unsqueeze(1)
        all_nir = torch.cat(all_nir, dim=0)[split_idx:, 1:]
        all_tir = torch.cat(all_tir, dim=0)[split_idx:, 1:]
    
    # if cfg.DATASETS.NAMES == 'RGBNT201':
    #     split_idx = 836
    #     all_old_rgb = torch.cat(all_old_rgb, dim=0)[:split_idx, 0]
    #     all_old_nir = torch.cat(all_old_nir, dim=0)[:split_idx, 0]
    #     all_old_tir = torch.cat(all_old_tir, dim=0)[:split_idx, 0]

    #     all_rgb = torch.cat(all_rgb, dim=0)[:split_idx, 0]
    #     all_nir = torch.cat(all_nir, dim=0)[:split_idx, 0]
    #     all_tir = torch.cat(all_tir, dim=0)[:split_idx, 0]
    # else:
    #     split_idx = 591
    #     all_old_rgb = torch.cat(all_old_rgb, dim=0)[split_idx:, 0]
    #     all_old_nir = torch.cat(all_old_nir, dim=0)[split_idx:, 0]
    #     all_old_tir = torch.cat(all_old_tir, dim=0)[split_idx:, 0]

    #     all_rgb = torch.cat(all_rgb, dim=0)[split_idx:, 0]
    #     all_nir = torch.cat(all_nir, dim=0)[split_idx:, 0]
    #     all_tir = torch.cat(all_tir, dim=0)[split_idx:, 0]

    cos_rgb_nir_old = F.cosine_similarity(all_old_rgb, all_old_nir, dim=-1)  # 输出形状 [836, 128]
    cos_rgb_tir_old = F.cosine_similarity(all_old_rgb, all_old_tir, dim=-1)  # 输出形状 [836, 128]
    cos_rgb_rgb_old = F.cosine_similarity(all_old_rgb, all_old_rgb_patchs, dim=-1)  # 输出形状 [836, 128]

    cos_rgb_nir = F.cosine_similarity(all_rgb, all_nir, dim=-1)  # 输出形状 [836, 128]
    cos_rgb_tir = F.cosine_similarity(all_rgb, all_tir, dim=-1)  # 输出形状 [836, 128]
    cos_rgb_rgb = F.cosine_similarity(all_rgb, all_old_rgb_patchs, dim=-1)  # 输出形状 [836, 128]

    # cos_rgb_nir_old = F.cosine_similarity(all_old_rgb, all_old_nir, dim=-1).mean(dim=-1)  # 输出形状 [836, 128]
    # cos_rgb_tir_old = F.cosine_similarity(all_old_rgb, all_old_tir, dim=-1).mean(dim=-1)  # 输出形状 [836, 128]

    # cos_rgb_nir = F.cosine_similarity(all_rgb, all_old_nir, dim=-1).mean(dim=-1)  # 输出形状 [836, 128]
    # cos_rgb_tir = F.cosine_similarity(all_rgb, all_old_tir, dim=-1).mean(dim=-1)  # 输出形状 [836, 128]

    ######################################################################
    # 将相似度张量平铺为一维
    similarity_rgb_nir_old_flat = cos_rgb_nir_old.flatten().cpu().numpy()
    similarity_rgb_tir_old_flat = cos_rgb_tir_old.flatten().cpu().numpy()
    similarity_rgb_rgb_old_flat = cos_rgb_rgb_old.flatten().cpu().numpy()
    similarity_rgb_nir_flat = cos_rgb_nir.flatten().cpu().numpy()
    similarity_rgb_tir_flat = cos_rgb_tir.flatten().cpu().numpy()
    similarity_rgb_rgb_flat = cos_rgb_rgb.flatten().cpu().numpy()

    # 设置直方图参数
    bins = np.linspace(0.7, 1.0, num=1000)  # 设置直方图的柱数
    alpha = 0.5  # 设置透明度
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()

    # Plot density curves instead of histograms
    sns.kdeplot(similarity_rgb_nir_old_flat, color="blue", fill=True, alpha=0.5, label="Before Propagation", ax=ax)
    sns.kdeplot(similarity_rgb_nir_flat, color="green", fill=True, alpha=0.5, label="After Propagation", ax=ax)
    plt.title("R2N")

    # sns.kdeplot(similarity_rgb_tir_old_flat, color="blue", fill=True, alpha=0.6, label="Before Propagation", ax=ax)
    # sns.kdeplot(similarity_rgb_tir_flat, color="green", fill=True, alpha=0.6, label="After Propagation", ax=ax)
    # plt.title("R2T")

    # sns.kdeplot(similarity_rgb_rgb_old_flat, color="blue", fill=True, alpha=0.6, label="Before Propagation", ax=ax)
    # sns.kdeplot(similarity_rgb_rgb_flat, color="green", fill=True, alpha=0.6, label="After Propagation", ax=ax)
    # plt.title("R2R")

    # 添加图例和标题
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()

    output_path = 'modal_sim_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'modal_sim_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_dir = os.path.join(output_path, str(0))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, f'modal_sim_vis_{0}.pdf'), format='pdf')
    plt.show()

def modal_sim_vis1(all_old_rgb, all_old_nir, all_old_tir, cfg):
    split_idx = 50
    all_old_rgb = torch.cat(all_old_rgb, dim=0)[:split_idx, 0].unsqueeze(1)
    all_old_nir = torch.cat(all_old_nir, dim=0)[:split_idx, 1:]
    all_old_tir = torch.cat(all_old_tir, dim=0)[:split_idx, 1:]

    # all_old_rgb = torch.cat(all_old_rgb, dim=0)[:split_idx, 0]
    # all_old_nir = torch.cat(all_old_nir, dim=0)[:split_idx, 0]
    # all_old_tir = torch.cat(all_old_tir, dim=0)[:split_idx, 0]
    
    # cos_rgb_nir_old = F.cosine_similarity(all_old_rgb, all_old_nir, dim=-1)  # 输出形状 [836, 128]
    # cos_rgb_tir_old = F.cosine_similarity(all_old_rgb, all_old_tir, dim=-1)  # 输出形状 [836, 128]
    
    cos_rgb_nir_old = F.cosine_similarity(all_old_nir, all_old_rgb, dim=-1)  # 输出形状 [836, 128]
    cos_rgb_tir_old = F.cosine_similarity(all_old_tir, all_old_rgb, dim=-1)  # 输出形状 [836, 128]

    ######################################################################
    # 将相似度张量平铺为一维
    similarity_rgb_nir_old_flat = cos_rgb_nir_old.flatten().cpu().numpy()
    similarity_rgb_tir_old_flat = cos_rgb_tir_old.flatten().cpu().numpy()

    # 设置直方图参数
    bins = np.linspace(0.01, 1.0, num=1000)  # 设置直方图的柱数
    alpha = 0.5  # 设置透明度
    plt.figure(figsize=(10, 6))

    # 绘制 NIR 和 RGB class token 相似度的直方图
    plt.hist(similarity_rgb_nir_old_flat, bins=bins, alpha=alpha, color='blue', label='RGB-NIR OLD Similarity')
    # 绘制 TIR 和 RGB class token 相似度的直方图
    plt.hist(similarity_rgb_tir_old_flat, bins=bins, alpha=alpha, color='orange', label='RGB-TIR Similarity')

    # 添加图例和标题
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Cosine Similarity Distribution between RGB Class Token and NIR/TIR Patch Tokens")
    plt.legend()

    output_path = 'modal_sim_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'modal_sim_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_dir = os.path.join(output_path, str(0))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, f'modal_sim_vis_{0}.pdf'), format='pdf')
    plt.show()

def prompt_img_sim_vis_cuver(
        all_rgb_old_patchs, all_nir_old_patchs, all_tir_old_patchs, 
        all_rgb_patchs, all_nir_patchs, all_tir_patchs, 
                             all_rgb_prompts, all_nir_prompts, all_tir_prompts, tmplt_prompt, cfg):

    if cfg.DATASETS.NAMES == 'RGBNT201':
        split_idx = 836
        
        all_rgb_patchs = torch.cat(all_rgb_patchs, dim=0)[:split_idx, 0]
        all_nir_patchs = torch.cat(all_nir_patchs, dim=0)[:split_idx, 0]
        all_tir_patchs = torch.cat(all_tir_patchs, dim=0)[:split_idx, 0]

        all_rgb_prompts = torch.cat(all_rgb_prompts, dim=0)[:split_idx]
        all_nir_prompts = torch.cat(all_nir_prompts, dim=0)[:split_idx]
        all_tir_prompts = torch.cat(all_tir_prompts, dim=0)[:split_idx]
        tmplt_prompt = tmplt_prompt.expand(836, 512)
    else:
        split_idx = 591

        all_rgb_patchs = torch.cat(all_rgb_patchs, dim=0)[split_idx:, 0]
        all_nir_patchs = torch.cat(all_nir_patchs, dim=0)[split_idx:, 0]
        all_tir_patchs = torch.cat(all_tir_patchs, dim=0)[split_idx:, 0]

        all_rgb_prompts = torch.cat(all_rgb_prompts, dim=0)[split_idx:]
        all_nir_prompts = torch.cat(all_nir_prompts, dim=0)[split_idx:]
        all_tir_prompts = torch.cat(all_tir_prompts, dim=0)[split_idx:]
        tmplt_prompt = tmplt_prompt.expand(1055, 512)

    # cos_rgb_rgb = F.cosine_similarity(all_rgb_patchs, all_rgb_prompts, dim=-1)  # 输出形状 [836, 128]
    # cos_rgb_nir = F.cosine_similarity(all_nir_patchs, all_rgb_prompts, dim=-1)  # 输出形状 [836, 128]
    # cos_rgb_tir = F.cosine_similarity(all_tir_patchs, all_rgb_prompts, dim=-1)  # 输出形状 [836, 128]

    # 测算 rgb的提示和 rgb的对齐相似度，以及nir与tir与光谱之间的相似度。
    cos_rgb_rgb = F.cosine_similarity(all_rgb_prompts, all_rgb_patchs, dim=-1)  # 输出形状 [836, 128]
    cos_nir_nir = F.cosine_similarity(all_nir_prompts, all_nir_patchs, dim=-1)  # 输出形状 [836, 128]
    cos_tir_tir = F.cosine_similarity(all_tir_prompts, all_tir_patchs, dim=-1)  # 输出形状 [836, 128]

    cos_tpl_rgb = F.cosine_similarity(tmplt_prompt, all_rgb_patchs, dim=-1)  # 输出形状 [836, 128]
    cos_tpl_nir = F.cosine_similarity(tmplt_prompt, all_nir_patchs, dim=-1)  # 输出形状 [836, 128]
    cos_tpl_tir = F.cosine_similarity(tmplt_prompt, all_tir_patchs, dim=-1)  # 输出形状 [836, 128]

    similarity_rgb_rgb_flat = cos_rgb_rgb.flatten().cpu().numpy()
    similarity_nir_nir_flat = cos_nir_nir.flatten().cpu().numpy()
    similarity_tir_tir_flat = cos_tir_tir.flatten().cpu().numpy()

    similarity_tpl_rgb_flat = cos_tpl_rgb.flatten().cpu().numpy()
    similarity_tpl_nir_flat = cos_tpl_nir.flatten().cpu().numpy()
    similarity_tpl_tir_flat = cos_tpl_tir.flatten().cpu().numpy()

    # 设置直方图参数
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()

    sns.kdeplot(similarity_rgb_rgb_flat, color="blue", fill=True, alpha=0.6, label="Prompt-to-RGB", ax=ax)
    sns.kdeplot(similarity_tpl_rgb_flat, color="green", fill=True, alpha=0.6, label="Tpl-to-RGB", ax=ax)
    plt.title("Prompt vs Tpl")

    # sns.kdeplot(similarity_nir_nir_flat, color="blue", fill=True, alpha=0.6, label="Prompt-to-Nir", ax=ax)
    # sns.kdeplot(similarity_tpl_nir_flat, color="green", fill=True, alpha=0.6, label="Tpl-to-Nir", ax=ax)
    # plt.title("Prompt vs Tpl")
    
    # sns.kdeplot(similarity_tir_tir_flat, color="blue", fill=True, alpha=0.6, label="Prompt-to-Tir", ax=ax)
    # sns.kdeplot(similarity_tpl_tir_flat, color="green", fill=True, alpha=0.6, label="Tpl-to-Tir", ax=ax)
    # plt.title("Prompt vs Tpl")

    # 添加图例和标题
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()

    output_path = 'modal_sim_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'modal_sim_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_dir = os.path.join(output_path, str(0))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, f'modal_sim_vis_{0}.pdf'), format='pdf')
    plt.show()

def vis_text_similarity(cfg, epoch, text_matrix, suffix=''):
    # 生成热图
    similarity_matrix = torch.matmul(text_matrix, text_matrix.t()).cpu()
    plt.imshow(similarity_matrix.numpy(), cmap='viridis', aspect='auto')
    plt.show()
    output_path = 'stage1_sim' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'stage1_sim')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, f'stage1_sim_{suffix}_{epoch}.png'))


def reshape_transform_vehicle(tensor, height=8, width=16):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_person(tensor, height=16, width=8):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_person_mim5(tensor, height=16, width=8):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def cam_vis(cfg, model, cam_loader, epoch):
    
    if cfg.DATASETS.NAMES in ['MSVR310','WMVEID863', 'RGBNT100']:
        reshape_transform = reshape_transform_vehicle
    elif cfg.DATASETS.NAMES in ['RGBNT201']:
        reshape_transform = reshape_transform_person
    else:
        raise NotImplementedError
    
    eigen_smooth = False
    aug_smooth = False
    # target_layers1 = [model.model1.image_encoder.transformer.resblocks[-1].ln_1,model.model1.image_encoder.transformer.resblocks[-2].ln_1,model.model1.image_encoder.transformer.resblocks[-3].ln_1,
    # model.model1.image_encoder.transformer.resblocks[-5].ln_1,model.model1.image_encoder.transformer.resblocks[-6].ln_1,model.model1.image_encoder.transformer.resblocks[-7].ln_1]
    # target_layers2 = [model.model2.image_encoder.transformer.resblocks[-1].ln_1,model.model2.image_encoder.transformer.resblocks[-2].ln_1,model.model2.image_encoder.transformer.resblocks[-3].ln_1,
    # model.model2.image_encoder.transformer.resblocks[-5].ln_1,model.model2.image_encoder.transformer.resblocks[-6].ln_1,model.model2.image_encoder.transformer.resblocks[-7].ln_1]
    # target_layers3 = [model.model3.image_encoder.transformer.resblocks[-1].ln_1,model.model3.image_encoder.transformer.resblocks[-2].ln_1,model.model3.image_encoder.transformer.resblocks[-3].ln_1,
    # model.model3.image_encoder.transformer.resblocks[-5].ln_1,model.model3.image_encoder.transformer.resblocks[-6].ln_1,model.model3.image_encoder.transformer.resblocks[-7].ln_1]

    target_layers1 = []
    target_layers2 = []
    target_layers3 = []
    # for i in range(1,2):
    #     target_layers1.append(model.model1.image_encoder.transformer.resblocks[-i].ln_1)
    #     target_layers2.append(model.model2.image_encoder.transformer.resblocks[-i].ln_1)
    #     target_layers3.append(model.model3.image_encoder.transformer.resblocks[-i].ln_1)
    target_layers1.append(model.model1.image_encoder.transformer.resblocks[-1].ln_1)
    target_layers2.append(model.model2.image_encoder.transformer.resblocks[-1].ln_1)
    target_layers3.append(model.model3.image_encoder.transformer.resblocks[-1].ln_1)

    cam1 = GradCAM(model=model.model1, target_layers=target_layers1, reshape_transform=reshape_transform)
    cam2 = GradCAM(model=model.model2, target_layers=target_layers2, reshape_transform=reshape_transform)
    cam3 = GradCAM(model=model.model3, target_layers=target_layers3, reshape_transform=reshape_transform)

    # cam1.uses_gradients = False
    # cam2.uses_gradients = False
    # cam3.uses_gradients = False

    output_path = 'cam_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'cam_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for n_iter, (img1, img2, img3, vid, camid, camids, sceneid, img_paths) in enumerate(cam_loader):
        img1 = img1.cuda()
        img2 = img2.cuda()
        img3 = img3.cuda()
        model.model1.cam_vis = True
        model.model2.cam_vis = True
        model.model3.cam_vis = True
        grayscale_cam1 = cam1(input_tensor=img1,
                            targets=None,
                            eigen_smooth=eigen_smooth,
                            aug_smooth=aug_smooth)
        grayscale_cam2 = cam2(input_tensor=img2,
                            targets=None,
                            eigen_smooth=eigen_smooth,
                            aug_smooth=aug_smooth)
        grayscale_cam3 = cam3(input_tensor=img3,
                            targets=None,
                            eigen_smooth=eigen_smooth,
                            aug_smooth=aug_smooth)
        model.model1.cam_vis = False
        model.model2.cam_vis = False
        model.model3.cam_vis = False
        # Here grayscale_cam has only one image in the batch
        grayscale_cam1 = grayscale_cam1[0, :]
        grayscale_cam2 = grayscale_cam2[0, :]
        grayscale_cam3 = grayscale_cam3[0, :]

        img_path_vis = img_paths[0]
        if cfg.DATASETS.NAMES in ['RGBNT201']:
            img_path_arr = img_path_vis.split('/')
            img_path_ni = os.path.join('/'.join(img_path_arr[:-2]),'NI','/'.join(img_path_arr[-1:]))
            img_path_th = os.path.join('/'.join(img_path_arr[:-2]),'TI','/'.join(img_path_arr[-1:]))
            max_width = 128
            max_height = 256
        else:
            img_path_ni = img_path_vis.replace('vis', 'ni')
            img_path_th = img_path_vis.replace('vis', 'th')
            max_width = 256
            max_height = 128
        img_vis = cv2.imread(img_path_vis, 1)[:, :, ::-1]
        img_ni = cv2.imread(img_path_ni, 1)[:, :, ::-1]
        img_th = cv2.imread(img_path_th, 1)[:, :, ::-1]

        # 使用 cv2.resize 调整图像尺寸
        img_vis = cv2.resize(img_vis, (max_width, max_height))
        img_ni = cv2.resize(img_ni, (max_width, max_height))
        img_th = cv2.resize(img_th, (max_width, max_height))

        img_vis = np.float32(img_vis) / 255 # (224, 224, 3)
        img_ni = np.float32(img_ni) / 255 # (224, 224, 3)
        img_th = np.float32(img_th) / 255 # (224, 224, 3)

        cam_image1 = show_cam_on_image(img_vis, grayscale_cam1)
        cam_image2 = show_cam_on_image(img_ni, grayscale_cam2)
        cam_image3 = show_cam_on_image(img_th, grayscale_cam3)

        if cfg.DATASETS.NAMES == 'WMVEID863':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = img_paths[0].split('/')[-3]+'_'+q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif cfg.DATASETS.NAMES == 'MSVR310':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif cfg.DATASETS.NAMES == 'RGBNT201':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'\n'+q_name[1]+'_'+q_name[2]+'_'+q_name[3]


        out_dir = os.path.join(output_path, str(epoch))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        q_name = q_name.replace('\n','_')
        cv2.imwrite(os.path.join(out_dir, q_name+'_vis.jpg'), cam_image1)
        cv2.imwrite(os.path.join(out_dir, q_name+'_nir.jpg'), cam_image2)
        cv2.imwrite(os.path.join(out_dir, q_name+'_tir.jpg'), cam_image3)

def cam_vis_sg(cfg, model, cam_loader, epoch):
    
    if cfg.DATASETS.NAMES in ['MSVR310','WMVEID863', 'RGBNT100']:
        reshape_transform = reshape_transform_vehicle
    elif cfg.DATASETS.NAMES in ['RGBNT201']:
        reshape_transform = reshape_transform_person_mim5
    else:
        raise NotImplementedError
    
    eigen_smooth = False
    aug_smooth = False
    # target_layers1 = [model.model1.image_encoder.transformer.resblocks[-1].ln_1,model.model1.image_encoder.transformer.resblocks[-2].ln_1,model.model1.image_encoder.transformer.resblocks[-3].ln_1,
    # model.model1.image_encoder.transformer.resblocks[-5].ln_1,model.model1.image_encoder.transformer.resblocks[-6].ln_1,model.model1.image_encoder.transformer.resblocks[-7].ln_1]
    # target_layers2 = [model.model2.image_encoder.transformer.resblocks[-1].ln_1,model.model2.image_encoder.transformer.resblocks[-2].ln_1,model.model2.image_encoder.transformer.resblocks[-3].ln_1,
    # model.model2.image_encoder.transformer.resblocks[-5].ln_1,model.model2.image_encoder.transformer.resblocks[-6].ln_1,model.model2.image_encoder.transformer.resblocks[-7].ln_1]
    # target_layers3 = [model.model3.image_encoder.transformer.resblocks[-1].ln_1,model.model3.image_encoder.transformer.resblocks[-2].ln_1,model.model3.image_encoder.transformer.resblocks[-3].ln_1,
    # model.model3.image_encoder.transformer.resblocks[-5].ln_1,model.model3.image_encoder.transformer.resblocks[-6].ln_1,model.model3.image_encoder.transformer.resblocks[-7].ln_1]

    target_layers1 = []
    target_layers2 = []
    target_layers3 = []
    # target_layers1.append(model.fusion_proj.attention1.norm1)
    # target_layers2.append(model.fusion_proj.attention2.norm1)
    # target_layers3.append(model.fusion_proj.attention3.norm1)

    # target_layers1.append(model.mim.corss_layer1.norm_k)
    # target_layers2.append(model.mim.corss_layer2.norm_k)
    # target_layers3.append(model.mim.corss_layer3.norm_k)

    # target_layers1.append(model.mim.msa2_rgb.ln_1)
    # target_layers2.append(model.mim.msa2_nir.ln_1)
    # target_layers3.append(model.mim.msa2_tir.ln_1)
    
    target_layers1.append(model.model1.image_encoder.transformer.resblocks[-1].ln_1)
    target_layers2.append(model.model2.image_encoder.transformer.resblocks[-1].ln_1)
    target_layers3.append(model.model3.image_encoder.transformer.resblocks[-1].ln_1)

    cam1 = GradCAM(model=model, target_layers=target_layers1, reshape_transform=reshape_transform)
    cam2 = GradCAM(model=model, target_layers=target_layers2, reshape_transform=reshape_transform)
    cam3 = GradCAM(model=model, target_layers=target_layers3, reshape_transform=reshape_transform)

    output_path = 'cam_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'cam_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # model.eval()
    for n_iter, (img1, img2, img3, vid, camid, camids, sceneid, img_paths) in enumerate(cam_loader):
        img1 = img1.cuda()
        img2 = img2.cuda()
        img3 = img3.cuda()
        model.cam_vis = True
        img = torch.concat((img1, img2, img3), dim=0)
        model.cam_vis_cls_type = 'RGB'
        grayscale_cam1 = cam1(input_tensor=img,
                            targets=None,
                            eigen_smooth=eigen_smooth,
                            aug_smooth=aug_smooth)
        grayscale_cam1 = grayscale_cam1[0, :]

        model.cam_vis_cls_type = 'NIR'
        grayscale_cam2 = cam2(input_tensor=img,
                            targets=None,
                            eigen_smooth=eigen_smooth,
                            aug_smooth=aug_smooth)
        grayscale_cam2 = grayscale_cam2[0, :]

        model.cam_vis_cls_type = 'TIR'
        grayscale_cam3 = cam3(input_tensor=img,
                            targets=None,
                            eigen_smooth=eigen_smooth,
                            aug_smooth=aug_smooth)
        grayscale_cam3 = grayscale_cam3[0, :]

        model.cam_vis = False
        # Here grayscale_cam has only one image in the batch
        img_path_vis = img_paths[0]
        if cfg.DATASETS.NAMES in ['RGBNT201']:
            img_path_arr = img_path_vis.split('/')
            img_path_ni = os.path.join('/'.join(img_path_arr[:-2]),'NI','/'.join(img_path_arr[-1:]))
            img_path_th = os.path.join('/'.join(img_path_arr[:-2]),'TI','/'.join(img_path_arr[-1:]))
            max_width = 128
            max_height = 256
        else:
            img_path_ni = img_path_vis.replace('vis', 'ni')
            img_path_th = img_path_vis.replace('vis', 'th')
            max_width = 256
            max_height = 128
        img_vis = cv2.imread(img_path_vis, 1)[:, :, ::-1]
        img_ni = cv2.imread(img_path_ni, 1)[:, :, ::-1]
        img_th = cv2.imread(img_path_th, 1)[:, :, ::-1]

        # 使用 cv2.resize 调整图像尺寸
        img_vis = cv2.resize(img_vis, (max_width, max_height))
        img_ni = cv2.resize(img_ni, (max_width, max_height))
        img_th = cv2.resize(img_th, (max_width, max_height))

        img_vis = np.float32(img_vis) / 255 # (224, 224, 3)
        img_ni = np.float32(img_ni) / 255 # (224, 224, 3)
        img_th = np.float32(img_th) / 255 # (224, 224, 3)

        cam_image1 = show_cam_on_image(img_vis, grayscale_cam1)
        cam_image2 = show_cam_on_image(img_ni, grayscale_cam2)
        cam_image3 = show_cam_on_image(img_th, grayscale_cam3)

        if cfg.DATASETS.NAMES == 'WMVEID863':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = img_paths[0].split('/')[-3]+'_'+q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif cfg.DATASETS.NAMES == 'MSVR310':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif cfg.DATASETS.NAMES == 'RGBNT201':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'\n'+q_name[1]+'_'+q_name[2]+'_'+q_name[3]


        out_dir = os.path.join(output_path, str(epoch))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        q_name = q_name.replace('\n','_')
        cv2.imwrite(os.path.join(out_dir, q_name+'_vis.jpg'), cam_image1)
        cv2.imwrite(os.path.join(out_dir, q_name+'_nir.jpg'), cam_image2)
        cv2.imwrite(os.path.join(out_dir, q_name+'_tir.jpg'), cam_image3)


def mim_attn_vis(cfg, model, cam_loader, epoch, device):
    output_path = 'mim_attn_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'mim_attn_vis', str(epoch))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for n_iter, (img1, img2, img3, vid, camid, camids, sceneid, img_paths) in enumerate(cam_loader):
        with torch.no_grad():
            img1 = img1.to(device)
            img2 = img2.to(device)
            if cfg.DATASETS.NAMES != 'RGBN300':
                img3 = img3.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.DATASETS.NAMES != 'RGBN300':
                feat, feat_bn, scores = model(img1, img2, img3, None, cam_label=camids, view_label=None)
            else:
                feat, feat_bn, scores = model(img1, img2, label = None, cam_label=camids, view_label=None)
            IMG_H = img1.shape[2]//16
            IMG_W = img1.shape[3]//16
            scores = scores[:,:,:,1:]
            scores_mm = torch.chunk(rearrange(scores, 'B C N (H W) -> B C N H W', H = IMG_H), 3, dim=1)
            scores_rgb = torch.nn.functional.interpolate(scores_mm[0].mean(dim=1), mode='bilinear', align_corners=False, size=(img1.shape[2], img1.shape[3])).squeeze(1).permute(1, 2, 0).cpu().numpy()
            scores_tir = torch.nn.functional.interpolate(scores_mm[1].mean(dim=1), mode='bilinear', align_corners=False, size=(img2.shape[2], img2.shape[3])).squeeze(1).permute(1, 2, 0).cpu().numpy()
            scores_nir = torch.nn.functional.interpolate(scores_mm[2].mean(dim=1), mode='bilinear', align_corners=False, size=(img3.shape[2], img3.shape[3])).squeeze(1).permute(1, 2, 0).cpu().numpy()
            
            scores_rgb = (scores_rgb - scores_rgb.min()) / (scores_rgb.max() - scores_rgb.min())
            scores_rgb = np.uint8(255 * scores_rgb)

            scores_tir = (scores_tir - scores_tir.min()) / (scores_tir.max() - scores_tir.min())
            scores_tir = np.uint8(255 * scores_tir)

            scores_nir = (scores_nir - scores_nir.min()) / (scores_nir.max() - scores_nir.min())
            scores_nir = np.uint8(255 * scores_nir)

            img_path_vis = img_paths[0]
            if cfg.DATASETS.NAMES in ['RGBNT201']:
                img_path_arr = img_path_vis.split('/')
                img_path_ni = os.path.join('/'.join(img_path_arr[:-2]),'NI','/'.join(img_path_arr[-1:]))
                img_path_th = os.path.join('/'.join(img_path_arr[:-2]),'TI','/'.join(img_path_arr[-1:]))
                max_width = 128
                max_height = 256
            else:
                img_path_ni = img_path_vis.replace('vis', 'ni')
                img_path_th = img_path_vis.replace('vis', 'th')
                max_width = 256
                max_height = 128
            img_vis = cv2.imread(img_path_vis, 1)[:, :, ::-1]
            img_ni = cv2.imread(img_path_ni, 1)[:, :, ::-1]
            img_th = cv2.imread(img_path_th, 1)[:, :, ::-1]

            # 使用 cv2.resize 调整图像尺寸
            img_vis = cv2.resize(img_vis, (max_width, max_height))
            img_ni = cv2.resize(img_ni, (max_width, max_height))
            img_th = cv2.resize(img_th, (max_width, max_height))

            # img_vis = np.float32(img_vis) / 255 # (224, 224, 3)
            # img_ni = np.float32(img_ni) / 255 # (224, 224, 3)
            # img_th = np.float32(img_th) / 255 # (224, 224, 3)
            
            heatmap_rgb = cv2.applyColorMap(scores_rgb, cv2.COLORMAP_JET)
            overlayed_img_rgb = cv2.addWeighted(img_vis, 0.4, heatmap_rgb, 0.6, 0)
            
            heatmap_nir = cv2.applyColorMap(scores_nir, cv2.COLORMAP_JET)
            overlayed_img_nir = cv2.addWeighted(img_ni, 0.4, heatmap_nir, 0.6, 0)
            
            heatmap_tir = cv2.applyColorMap(scores_tir, cv2.COLORMAP_JET)
            overlayed_img_tir = cv2.addWeighted(img_th, 0.4, heatmap_tir, 0.6, 0)
            # cam_image1 = show_cam_on_image(img_vis, scores_rgb.reshape(scores_rgb.shape[1],scores_rgb.shape[2], 1).cpu().numpy())
            # cam_image2 = show_cam_on_image(img_ni, scores_tir.reshape(scores_rgb.shape[1],scores_rgb.shape[2], 1).cpu().numpy())
            # cam_image3 = show_cam_on_image(img_th, scores_nir.reshape(scores_rgb.shape[1],scores_rgb.shape[2], 1).cpu().numpy())

            if cfg.DATASETS.NAMES == 'WMVEID863':
                q_name = img_paths[0].split('/')[-1][:-4].split('_')
                q_name = img_paths[0].split('/')[-3]+'_'+q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
            elif cfg.DATASETS.NAMES == 'MSVR310':
                q_name = img_paths[0].split('/')[-1][:-4].split('_')
                q_name = q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
            elif cfg.DATASETS.NAMES == 'RGBNT201':
                q_name = img_paths[0].split('/')[-1][:-4].split('_')
                q_name = q_name[0]+'\n'+q_name[1]+'_'+q_name[2]+'_'+q_name[3]

            q_name = q_name.replace('\n','_')
            cv2.imwrite(os.path.join(output_path, q_name+'_vis.jpg'), overlayed_img_rgb)
            cv2.imwrite(os.path.join(output_path, q_name+'_nir.jpg'), overlayed_img_nir)
            cv2.imwrite(os.path.join(output_path, q_name+'_tir.jpg'), overlayed_img_tir)

            # plt.imshow(scores_rgb.permute(1, 2, 0).cpu().numpy(), cmap='viridis')
            # plt.savefig(os.path.join(output_path, q_name+'_vis.jpg'))
            # plt.close()

            # plt.imshow(scores_tir.permute(1, 2, 0).cpu().numpy(), cmap='viridis')
            # plt.savefig(os.path.join(output_path, q_name+'_nir.jpg'))
            # plt.close()

            # plt.imshow(scores_nir.permute(1, 2, 0).cpu().numpy(), cmap='viridis')
            # plt.savefig(os.path.join(output_path, q_name+'_tir.jpg'))
            # plt.close()


def prompt_attn_vis(cfg, all_attn, all_paths):
    output_path = 'prompt_attn_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'prompt_attn_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for n_iter in range(len(all_paths)):
        attn = all_attn[n_iter]
        img_path = all_paths[n_iter]
        if cfg.DATASETS.NAMES in ['RGBNT201', 'Market-MM']:
            IMG_H = 256
            IMG_W = 128
        else:
            IMG_H = 128
            IMG_W = 256

        # attn_mm = attn

        attn_1 = torch.tensor(attn[:8,:,:])
        attn_2 = torch.tensor(attn[8:16,:,:])
        attn_3 = torch.tensor(attn[16::,:])
        attn_stack = torch.stack([attn_1, attn_2, attn_3], dim=0)  # shape: [64, 3, 8, 1, 129]
        # attn_softmax = attn_stack.mean(dim=0)
        # attn_softmax = F.softmax(attn_stack*0.001, dim=0)  # shape: [64, 3, 8, 1, 129]
        # attn_softmax = F.softmax(attn_stack*(64**-0.5), dim=0)  # shape: [64, 3, 8, 1, 129]
        attn_softmax = F.softmax(attn_stack/(64**-0.5), dim=0)  # shape: [64, 3, 8, 1, 129]
        # breakpoint()
        attn_mm = (attn_softmax * attn_stack).sum(dim=0)  # shape: [64, 8, 1, 129]
        attn_mm = attn_mm.detach().cpu().numpy()

        scores_rgb = rearrange(attn[:8,:,:].mean(0)[:, 1:], '1 (H W) -> 1 1 H W', H=IMG_H//16)
        scores_tir = rearrange(attn[8:16,:,:].mean(0)[:, 1:], '1 (H W) -> 1 1 H W', H=IMG_H//16)
        scores_nir = rearrange(attn[16::,:].mean(0)[:, 1:], '1 (H W) -> 1 1 H W', H=IMG_H//16)
        scores_all = rearrange(attn_mm.mean(0)[:, 1:], '1 (H W) -> 1 1 H W', H=IMG_H//16)

        scores_rgb = torch.from_numpy(scores_rgb).float()
        scores_tir = torch.from_numpy(scores_tir).float()
        scores_nir = torch.from_numpy(scores_nir).float()
        scores_all = torch.from_numpy(scores_all).float()

        scores_rgb = F.interpolate(scores_rgb, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
        scores_tir = F.interpolate(scores_tir, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
        scores_nir = F.interpolate(scores_nir, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
        scores_all = F.interpolate(scores_all, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()

        scores_rgb = (scores_rgb - scores_rgb.min()) / (scores_rgb.max() - scores_rgb.min())
        scores_rgb = np.uint8(255 * scores_rgb)

        scores_tir = (scores_tir - scores_tir.min()) / (scores_tir.max() - scores_tir.min())
        scores_tir = np.uint8(255 * scores_tir)

        scores_nir = (scores_nir - scores_nir.min()) / (scores_nir.max() - scores_nir.min())
        scores_nir = np.uint8(255 * scores_nir)

        scores_all = (scores_all - scores_all.min()) / (scores_all.max() - scores_all.min())
        scores_all = np.uint8(255 * scores_all)

        img_path_vis = img_path
        if cfg.DATASETS.NAMES in ['RGBNT201']:
            img_path_arr = img_path_vis.split('/')
            img_path_ni = os.path.join('/'.join(img_path_arr[:-2]),'NI','/'.join(img_path_arr[-1:]))
            img_path_th = os.path.join('/'.join(img_path_arr[:-2]),'TI','/'.join(img_path_arr[-1:]))
            max_width = 128
            max_height = 256
        else:
            img_path_ni = img_path_vis.replace('vis', 'ni')
            img_path_th = img_path_vis.replace('vis', 'th')
            max_width = 256
            max_height = 128
        img_vis = cv2.imread(img_path_vis, 1)[:, :, ::-1]
        img_ni = cv2.imread(img_path_ni, 1)[:, :, ::-1]
        img_th = cv2.imread(img_path_th, 1)[:, :, ::-1]

        # 使用 cv2.resize 调整图像尺寸
        img_vis = cv2.resize(img_vis, (max_width, max_height))
        img_ni = cv2.resize(img_ni, (max_width, max_height))
        img_th = cv2.resize(img_th, (max_width, max_height))

        # img_vis = np.float32(img_vis) / 255 # (224, 224, 3)
        # img_ni = np.float32(img_ni) / 255 # (224, 224, 3)
        # img_th = np.float32(img_th) / 255 # (224, 224, 3)
        
        heatmap_rgb = cv2.applyColorMap(scores_rgb, cv2.COLORMAP_JET)
        overlayed_img_rgb = cv2.addWeighted(img_vis, 0.4, heatmap_rgb, 0.6, 0)
        
        heatmap_nir = cv2.applyColorMap(scores_nir, cv2.COLORMAP_JET)
        overlayed_img_nir = cv2.addWeighted(img_ni, 0.4, heatmap_nir, 0.6, 0)
        
        heatmap_tir = cv2.applyColorMap(scores_tir, cv2.COLORMAP_JET)
        overlayed_img_tir = cv2.addWeighted(img_th, 0.4, heatmap_tir, 0.6, 0)

        heatmap_all = cv2.applyColorMap(scores_all, cv2.COLORMAP_JET)
        overlayed_img_all = cv2.addWeighted(img_vis, 0.4, heatmap_all, 0.6, 0)

        if cfg.DATASETS.NAMES == 'WMVEID863':
            q_name = img_path.split('/')[-1][:-4].split('_')
            q_name = img_path.split('/')[-3]+'_'+q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif cfg.DATASETS.NAMES == 'MSVR310':
            q_name = img_path.split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif cfg.DATASETS.NAMES == 'RGBNT201':
            q_name = img_path.split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'\n'+q_name[1]+'_'+q_name[2]+'_'+q_name[3]

        q_name = q_name.replace('\n','_')
        cv2.imwrite(os.path.join(output_path, q_name+'_vis.jpg'), overlayed_img_rgb)
        cv2.imwrite(os.path.join(output_path, q_name+'_nir.jpg'), overlayed_img_nir)
        cv2.imwrite(os.path.join(output_path, q_name+'_tir.jpg'), overlayed_img_tir)
        cv2.imwrite(os.path.join(output_path, q_name+'_all.jpg'), overlayed_img_all)


def feat_vis(cfg, model, cam_loader, epoch):
    output_path = 'feat_vis' if cfg is None else os.path.join(cfg.OUTPUT_DIR, 'feat_vis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for n_iter, (img1, img2, img3, vid, camid, camids, sceneid, img_paths) in enumerate(cam_loader):
        img1 = img1.cuda()
        img2 = img2.cuda()
        img3 = img3.cuda()

        feats_12, feats_11 = model(img1, img2, img3, None, cam_label=None, view_label=None)
        # rgb_feats, _, _ = torch.chunk(feats_12, 3, dim=1)
        # _, rgb_feats, _ = torch.chunk(feats_11, 3, dim=1)
        rgb_feats, nir_feats, tir_feats = torch.chunk(feats_12, 3, dim=1)

        # 平均聚合最后的特征
        # rgb_feats_mean = rgb_feats.mean(-1)[:, 1:].reshape(8, 16)  # Shape: [8, 16]
        if cfg.DATASETS.NAMES in ['RGBNT201']:
            rgb_feats_mean = rgb_feats.mean(-1)[:, 1:].reshape(16, 8)  # Shape: [8, 16]
            nir_feats_mean = nir_feats.mean(-1)[:, 1:].reshape(16, 8)  # Shape: [8, 16]
            tir_feats_mean = tir_feats.mean(-1)[:, 1:].reshape(16, 8)  # Shape: [8, 16]
            width_height = (256, 128)
        elif cfg.DATASETS.NAMES in ['MSVR310']:
            rgb_feats_mean = rgb_feats.mean(-1)[:, 1:].reshape(8, 16)  # Shape: [8, 16]
            nir_feats_mean = nir_feats.mean(-1)[:, 1:].reshape(8, 16)  # Shape: [8, 16]
            tir_feats_mean = tir_feats.mean(-1)[:, 1:].reshape(8, 16)  # Shape: [8, 16]
            width_height = (128, 256)

        img_path_vis = img_paths[0]
        if cfg.DATASETS.NAMES in ['RGBNT201']:
            img_path_arr = img_path_vis.split('/')
            img_path_ni = os.path.join('/'.join(img_path_arr[:-2]),'NI','/'.join(img_path_arr[-1:]))
            img_path_th = os.path.join('/'.join(img_path_arr[:-2]),'TI','/'.join(img_path_arr[-1:]))

        if cfg.DATASETS.NAMES == 'WMVEID863':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = img_paths[0].split('/')[-3]+'_'+q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif cfg.DATASETS.NAMES == 'MSVR310':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'_'+q_name[1]+'\n'+q_name[2]+'_'+q_name[3]
        elif cfg.DATASETS.NAMES == 'RGBNT201':
            q_name = img_paths[0].split('/')[-1][:-4].split('_')
            q_name = q_name[0]+'\n'+q_name[1]+'_'+q_name[2]+'_'+q_name[3]

        q_name = q_name.replace('\n','_')

        interpolated_feat_img(rgb_feats_mean, img_path_vis, output_path, width_height, q_name, 'rgb')
        interpolated_feat_img(nir_feats_mean, img_path_vis, output_path, width_height, q_name, 'nir')
        interpolated_feat_img(tir_feats_mean, img_path_vis, output_path, width_height, q_name, 'tir')

def interpolated_feat_img(rgb_feats_mean, img_path_vis, output_path, width_height, n_idx, img_type='rgb'):
    # 插值到图像大小 [256, 128]
    rgb_feats_interpolated = F.interpolate(
        rgb_feats_mean.unsqueeze(0).unsqueeze(0), 
        size=width_height, 
        mode='bilinear'
        # mode='bicubic'
        , align_corners=False
    ).squeeze().detach().cpu().numpy()

    # rgb_feats_interpolated = (rgb_feats_interpolated - rgb_feats_interpolated.min()) / (rgb_feats_interpolated.max() - rgb_feats_interpolated.min())  # 归一化
    

    # 归一化到 0-1 范围
    rgb_feats_norm = (rgb_feats_interpolated - rgb_feats_interpolated.min()) / \
                        (rgb_feats_interpolated.max() - rgb_feats_interpolated.min())
    
    rgb_feats_norm = np.power(rgb_feats_norm, 0.8)  # 增强对比度

    # 读取原始图像
    # img_path_vis = img_paths[0]
    img_vis = cv2.imread(img_path_vis, 1)[:, :, ::-1]  # BGR to RGB
    img_vis = cv2.resize(img_vis, (width_height[1], width_height[0]))  # 确保图像大小与特征匹配
    img_vis = np.float32(img_vis) / 255

    # 将特征叠加到原始图像
    heatmap = cv2.applyColorMap(np.uint8(255 * rgb_feats_norm), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255  # Heatmap 归一化
    overlay = heatmap * 0.6 + img_vis * 0.4  # 权重调整叠加

    # 保存结果
    out_dir = os.path.join(output_path, str(0))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    overlay_path = os.path.join(out_dir, f"{n_idx}_{img_type}.png")
    cv2.imwrite(overlay_path, np.uint8(overlay * 255)[:, :, ::-1])  # RGB to BGR for saving
