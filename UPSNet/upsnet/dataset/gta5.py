# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

from __future__ import print_function

import gzip
import json
import os
import pickle
import sys
from collections import Sequence, defaultdict

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.utils.data
from lib.utils.logging import logger
from PIL import Image, ImageDraw
from upsnet.bbox.sample_rois import sample_rois
from upsnet.config.config import config
from upsnet.dataset.base_dataset import BaseDataset
from upsnet.dataset.json_gta5_dataset import (JsonGTA5Dataset,
                                              add_bbox_regression_targets,
                                              extend_with_flipped_entries,
                                              filter_for_training)
from upsnet.rpn.assign_anchor import add_rpn_blobs


class GTA5(BaseDataset):

    def __init__(self, image_sets, flip=False, proposal_files=None, phase='train', result_path=''):

        super(GTA5, self).__init__()

        self.image_dirs = {
            'train': os.path.join(config.dataset.dataset_path, 'train'),
            'val': os.path.join(config.dataset.dataset_path, 'val'),
            'valsmall': os.path.join(config.dataset.dataset_path, 'valsmall'),
        }

        self.anno_files = {
            'train': os.path.join(config.dataset.dataset_path, 'train', 'inst.json'),
            'val': os.path.join(config.dataset.dataset_path, 'val', 'inst.json'),
            'valsmall': os.path.join(config.dataset.dataset_path, 'valsmall', 'inst.json'),
        }

        self.panoptic_json_file = os.path.join(config.dataset.dataset_path, 'valsmall', 'panoptic.json')
        self.panoptic_gt_folder = os.path.join(config.dataset.dataset_path, 'valsmall')

        self.flip = flip
        self.result_path = result_path
        self.num_classes = 11  # num of instance classes + 1
        self.phase = phase
        self.image_sets = image_sets

        # if image_sets[0] == 'demoVideo':
        #     assert len(image_sets) == 1
        #     assert phase == 'test'
        #     im_path = [_.strip() for _ in open('data/cityscapes/split/demoVideo_img.txt', 'r').readlines()]
        #     self.roidb = [{'image': _, 'flipped': False} for _ in im_path]
        #     return

        if proposal_files is None:
            proposal_files = [None] * len(image_sets)

        if phase == 'train' and len(image_sets) > 1:
            # combine multiple datasets
            roidbs = []
            for image_set, proposal_file in zip(image_sets, proposal_files):
                dataset = JsonGTA5Dataset('gta5_' + image_set,
                                          image_dir=self.image_dirs[image_set],
                                          anno_file=self.anno_files[image_set])
                roidb = dataset.get_roidb(gt=True, proposal_file=proposal_file,
                                          crowd_filter_thresh=config.train.crowd_filter_thresh)
                if flip:
                    if logger:
                        logger.info('Appending horizontally-flipped training examples...')
                    extend_with_flipped_entries(roidb, dataset)
                roidbs.append(roidb)
            roidb = roidbs[0]
            for r in roidbs[1:]:
                roidb.extend(r)
            roidb = filter_for_training(roidb)
            add_bbox_regression_targets(roidb)

        else:
            assert len(image_sets) == 1
            self.dataset = JsonGTA5Dataset('gta5_' + image_sets[0],
                                           image_dir=self.image_dirs[image_sets[0]],
                                           anno_file=self.anno_files[image_sets[0]])
            roidb = self.dataset.get_roidb(gt=True, proposal_file=proposal_files[0],
                                           crowd_filter_thresh=config.train.crowd_filter_thresh if phase != 'test' else 0)
            if flip:
                if logger:
                    logger.info('Appending horizontally-flipped training examples...')
                extend_with_flipped_entries(roidb, self.dataset)
            if phase != 'test':
                roidb = filter_for_training(roidb)
                add_bbox_regression_targets(roidb)

        self.roidb = roidb

    def __getitem__(self, index):
        blob = defaultdict(list)
        im_blob, im_scales = self.get_image_blob([self.roidb[index]])
        if config.network.has_rpn:
            if self.phase != 'test':
                add_rpn_blobs(blob, im_scales, [self.roidb[index]])
                data = {'data': im_blob,
                        'im_info': blob['im_info']}
                label = {'roidb': blob['roidb'][0]}
                for stride in config.network.rpn_feat_stride:
                    label.update({
                        'rpn_labels_fpn{}'.format(stride): blob['rpn_labels_int32_wide_fpn{}'.format(stride)].astype(
                            np.int64),
                        'rpn_bbox_targets_fpn{}'.format(stride): blob['rpn_bbox_targets_wide_fpn{}'.format(stride)],
                        'rpn_bbox_inside_weights_fpn{}'.format(stride): blob[
                            'rpn_bbox_inside_weights_wide_fpn{}'.format(stride)],
                        'rpn_bbox_outside_weights_fpn{}'.format(stride): blob[
                            'rpn_bbox_outside_weights_wide_fpn{}'.format(stride)]
                    })
            else:
                data = {'data': im_blob,
                        'im_info': np.array([[im_blob.shape[-2],
                                              im_blob.shape[-1],
                                              im_scales[0]]], np.float32),
                        }
                label = {'roidb': self.roidb[index]}
        else:
            if self.phase != 'test':
                frcn_blob = sample_rois(self.roidb[index], im_scales, 0)

                data = {'data': im_blob,
                        'im_info': np.array([[im_blob.shape[-2],
                                              im_blob.shape[-1],
                                              im_scales[0]]], np.float32)}
                label = {'rois': frcn_blob['rois'].astype(np.float32),
                         'cls_label': frcn_blob['labels_int32'].astype(np.int64),
                         'bbox_target': frcn_blob['bbox_targets'].astype(np.float32),
                         'bbox_inside_weight': frcn_blob['bbox_inside_weights'].astype(np.float32),
                         'bbox_outside_weight': frcn_blob['bbox_outside_weights'].astype(np.float32),
                         'mask_rois': frcn_blob['mask_rois'].astype(np.float32),
                         'mask_target': frcn_blob['mask_int32'].astype(np.float32)}
            else:
                data = {'data': im_blob,
                        'rois': np.hstack((np.zeros((self.roidb[index]['boxes'].shape[0], 1)), self.roidb[index]['boxes'])).astype(np.float32),
                        'im_info': np.array([[im_blob.shape[-2],
                                              im_blob.shape[-1],
                                              im_scales[0]]], np.float32),
                        'id': self.roidb[index]['id']}
                label = None
        if config.network.has_fcn_head:
            if self.phase != 'test':
                seg_gt = np.array(Image.open(self.roidb[index]['image'].replace('img', 'lblcls').replace('jpg', 'png')))
                if self.roidb[index]['flipped']:
                    seg_gt = np.fliplr(seg_gt)
                seg_gt = cv2.resize(seg_gt, None, None, fx=im_scales[0],
                                    fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                label.update({'seg_gt': seg_gt})
                label.update({'gt_classes': label['roidb']['gt_classes']})
                label.update({'mask_gt': np.zeros((len(label['gt_classes']), im_blob.shape[-2], im_blob.shape[-1]))})
                for i in range(len(label['gt_classes'])):
                    img = np.zeros((im_blob.shape[-2], im_blob.shape[-1]), dtype='uint8')
                    instances_img = np.array(Image.open(
                        self.roidb[index]['image'].replace('img', 'inst').replace('jpg', 'png')))
                    instances_img = cv2.resize(
                        instances_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img[(instances_img[:, :, 0] == label['roidb']['segms'][i][0]) &
                        (instances_img[:, :, 1] == label['roidb']['segms'][i][1]) &
                        (instances_img[:, :, 2] == label['roidb']['segms'][i][2])] = 1
                    label['mask_gt'][i] = img
                if config.train.fcn_with_roi_loss:
                    gt_boxes = label['roidb']['boxes'][np.where(label['roidb']['gt_classes'] > 0)[0]]
                    gt_boxes = np.around(gt_boxes * im_scales[0]).astype(np.int32)
                    label.update({'seg_roi_gt': np.zeros(
                        (len(gt_boxes), config.network.mask_size, config.network.mask_size), dtype=np.int64)})
                    for i in range(len(gt_boxes)):
                        if gt_boxes[i][3] == gt_boxes[i][1]:
                            gt_boxes[i][3] += 1
                        if gt_boxes[i][2] == gt_boxes[i][0]:
                            gt_boxes[i][2] += 1
                        label['seg_roi_gt'][i] = cv2.resize(seg_gt[gt_boxes[i][1]:gt_boxes[i][3], gt_boxes[i][0]:gt_boxes[i][2]], (
                            config.network.mask_size, config.network.mask_size), interpolation=cv2.INTER_NEAREST)
            else:
                pass

        return data, label, index

    def get_image_blob(self, roidb):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = len(roidb)
        # Sample random scales to use for each image in this batch
        if self.phase == 'train':
            scale_inds = np.random.randint(
                0, high=len(config.train.scales), size=num_images
            )
        else:
            scale_inds = np.random.randint(
                0, high=len(config.test.scales), size=num_images
            )
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(roidb[i]['image'])
            assert im is not None, \
                'Failed to read image \'{}\''.format(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            if self.phase == 'train':
                target_size = config.train.scales[scale_inds[i]]
                im, im_scale = self.prep_im_for_blob(
                    im, config.network.pixel_means, [target_size], config.train.max_size
                )
            else:
                target_size = config.test.scales[scale_inds[i]]
                im, im_scale = self.prep_im_for_blob(
                    im, config.network.pixel_means, [target_size], config.test.max_size
                )
            im_scales.append(im_scale[0])
            processed_ims.append(im[0].transpose(2, 0, 1))

        # Create a blob to hold the input images
        assert len(processed_ims) == 1
        blob = processed_ims[0]

        return blob, im_scales

    def vis_all_mask(self, all_boxes, all_masks, save_path=None):
        """
        visualize all detections in one image
        :param im_array: [b=1 c h w] in rgb
        :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        :param class_names: list of names in imdb
        :param scale: visualize the scaled image
        :return:
        """
        import matplotlib
        matplotlib.use('Agg')
        import random

        import cv2
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        palette = {
            'sky': (70, 130, 180),
            'road': (128, 64, 128),
            'sidewalk': (244, 35, 232),
            'terrain': (152, 251, 152),
            'tree': (87, 182, 35),
            'vegetation': (35, 142, 35),
            'building': (70, 70, 70),
            'infrastructure': (153, 153, 153),
            'fence': (190, 153, 153),
            'billboard': (150, 20, 20),
            'traffic light': (250, 170, 30),
            'traffic sign': (220, 220, 0),
            'mobile barrier': (180, 180, 100),
            'fire hydrant': (173, 153, 153),
            'chair': (168, 153, 153),
            'trash': (81, 0, 21),
            'trash can': (81, 0, 81),
            'person': (220, 20, 60),
            'motorcycle': (0, 0, 230),
            'car': (0, 0, 142),
            'van': (0, 80, 100),
            'bus': (0, 60, 100),
            'truck': (0, 0, 70)
        }
        name2id = {
            'sky': 0,
            'road': 1,
            'sidewalk': 2,
            'terrain': 3,
            'tree': 4,
            'vegetation': 5,
            'building': 6,
            'infrastructure': 7,
            'fence': 8,
            'billboard': 9,
            'traffic light': 10,
            'traffic sign': 11,
            'mobile barrier': 12,
            'fire hydrant': 13,
            'chair': 14,
            'trash': 15,
            'trash can': 16,
            'person': 17,
            'motorcycle': 18,
            'car': 19,
            'van': 20,
            'bus': 21,
            'truck': 22
        }

        self.classes = [
            '__background__',
            'traffic light',
            'fire hydrant',
            'chair',
            'trash can',
            'person',
            'motorcycle',
            'car',
            'van',
            'bus',
            'truck',
        ]

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        for i in range(len(self.roidb)):

            im = np.array(Image.open(self.roidb[i]['image']))
            fig = plt.figure(frameon=False)

            fig.set_size_inches(im.shape[1] / 200, im.shape[0] / 200)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(im)
            for j, name in enumerate(self.classes):
                if name == '__background__':
                    continue
                boxes = all_boxes[j][i]
                segms = all_masks[j][i]
                if segms == []:
                    continue
                masks = mask_util.decode(segms)
                for k in range(boxes.shape[0]):
                    score = boxes[k, -1]
                    mask = masks[:, :, k]
                    if score < 0.5:
                        continue
                    bbox = boxes[k, :]
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                      fill=False, edgecolor='g', linewidth=1, alpha=0.5)
                    )
                    ax.text(bbox[0], bbox[1] - 2, name + '{:0.2f}'.format(score).lstrip('0'), fontsize=5, family='serif',
                            bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'), color='white')
                    contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    color = (palette[name][0] / 255, palette[name][1] / 255, palette[name][2] / 255)
                    for c in contour:
                        ax.add_patch(
                            Polygon(
                                c.reshape((-1, 2)),
                                fill=True, facecolor=color, edgecolor='w', linewidth=0.8, alpha=0.5
                            )
                        )
            if save_path is None:
                plt.show()
            else:
                fig.savefig(os.path.join(save_path, '{}.png'.format(
                    self.roidb[i]['image'].split('/')[-1][:-4])), dpi=200)
            plt.close('all')

    def evaluate_masks(
            self,
            all_boxes,
            all_segms,
            output_dir,
    ):
        res_file = os.path.join(
            output_dir, 'segmentations_' + self.dataset.name + '_results')
        res_file += '.json'

        roidb = self.dataset.get_roidb()
        for i, entry in enumerate(roidb):
            im_name = entry['image']

            basename = os.path.splitext(os.path.basename(im_name))[0]
            txtname = os.path.join(output_dir, 'inst_seg', basename + 'pred.txt')
            os.makedirs(os.path.join(output_dir, 'inst_seg'), exist_ok=True)
            with open(txtname, 'w') as fid_txt:
                for j in range(1, len(all_segms)):
                    clss = self.dataset.classes[j]
                    segms = all_segms[j][i]
                    boxes = all_boxes[j][i]
                    if segms == []:
                        continue
                    masks = mask_util.decode(segms)

                    for k in range(boxes.shape[0]):
                        score = boxes[k, -1]
                        mask = masks[:, :, k]
                        pngname = os.path.join(
                            'seg_results', basename,
                            basename + '_' + clss + '_{}.png'.format(k))
                        # write txt
                        fid_txt.write('{} {}\n'.format(pngname, score))
                        # save mask
                        os.makedirs(os.path.join(output_dir, 'inst_seg', 'seg_results', basename), exist_ok=True)
                        cv2.imwrite(os.path.join(output_dir, 'inst_seg', pngname), mask * 255)
        return None

    def get_pallete(self):
        pallete = np.zeros((256, 3)).astype('uint8')

        pallete[0, :] = [70, 130, 180]
        pallete[1, :] = [128, 64, 128]
        pallete[2, :] = [244, 35, 232]
        pallete[3, :] = [152, 251, 152]
        pallete[4, :] = [87, 182, 35]
        pallete[5, :] = [35, 142, 35]
        pallete[6, :] = [70, 70, 70]
        pallete[7, :] = [153, 153, 153]
        pallete[8, :] = [190, 153, 153]
        pallete[9, :] = [150, 20, 20]
        pallete[10, :] = [250, 170, 30]
        pallete[11, :] = [220, 220, 0]
        pallete[12, :] = [180, 180, 100]
        pallete[13, :] = [173, 153, 153]
        pallete[14, :] = [168, 153, 153]
        pallete[15, :] = [81, 0, 21]
        pallete[16, :] = [81, 0, 81]
        pallete[17, :] = [220, 20, 60]
        pallete[18, :] = [0, 0, 230]
        pallete[19, :] = [0, 0, 142]
        pallete[20, :] = [0, 80, 100]
        pallete[21, :] = [0, 60, 100]
        pallete[22, :] = [0, 0, 70]

        pallete = pallete.reshape(-1)

        # return pallete_raw
        return pallete

    def evaluate_ssegs(self, pred_segmentations, res_file_folder):
        self.write_segmentation_result(pred_segmentations, res_file_folder)

        confusion_matrix = np.zeros((config.dataset.num_seg_classes, config.dataset.num_seg_classes))
        for i, roidb in enumerate(self.roidb):

            seg_gt = np.array(Image.open(roidb['image'].replace('img', 'lblcls').replace(
                'jpg', 'png'))).astype('float32')

            seg_pathes = os.path.split(roidb['image'].replace('img', 'lblcls').replace(
                'jpg', 'png'))
            res_image_name = seg_pathes[-1][:-len('.png')]
            res_save_path = os.path.join(res_file_folder, res_image_name + '.png')

            seg_pred = Image.open(res_save_path)

            seg_pred = np.array(seg_pred.resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST))
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, config.dataset.num_seg_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        evaluation_results = {'meanIU': mean_IU, 'IU_array': IU_array, 'confusion_matrix': confusion_matrix}

        def convert_confusion_matrix(confusion_matrix):
            cls_sum = confusion_matrix.sum(axis=1)
            confusion_matrix = confusion_matrix / cls_sum.reshape((-1, 1))
            return confusion_matrix

        logger.info('evaluate segmentation:')
        meanIU = evaluation_results['meanIU']
        IU_array = evaluation_results['IU_array']
        confusion_matrix = convert_confusion_matrix(evaluation_results['confusion_matrix'])
        logger.info('IU_array:')
        for i in range(len(IU_array)):
            logger.info('%.5f' % IU_array[i])
        logger.info('meanIU:%.5f' % meanIU)
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        import re
        confusion_matrix = re.sub('[\[\]]', '', np.array2string(confusion_matrix, separator='\t'))
        logger.info('confusion_matrix:')
        logger.info(confusion_matrix)

    def write_segmentation_result(self, segmentation_results, res_file_folder):
        """
        Write the segmentation result to result_file_folder
        :param segmentation_results: the prediction result
        :param result_file_folder: the saving folder
        :return: [None]
        """
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        pallete = self.get_pallete()
        for i, roidb in enumerate(self.roidb):

            seg_pathes = os.path.split(roidb['image'])
            res_image_name = seg_pathes[-1][:-len('.png')]
            res_save_path = os.path.join(res_file_folder, res_image_name + '.png')

            segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))
            segmentation_result = Image.fromarray(segmentation_result)
            segmentation_result.putpalette(pallete)
            segmentation_result.save(res_save_path)
