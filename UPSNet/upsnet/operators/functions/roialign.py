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

import torch
from torch.autograd import Function
from .._ext.roi_align import roi_align_cuda


class RoIAlignFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale, sampling_ratio=2):
        batch_size, num_channels, data_height, data_width = features.shape
        num_rois = rois.shape[0]

        if not features.is_cuda:
            raise Exception('not implemented')

        output = features.new().resize_(num_rois, num_channels, pooled_height, pooled_width).zero_()

        roi_align_cuda.roi_align_forward(pooled_height, pooled_width, sampling_ratio, spatial_scale,
                                         features, rois, output)
        feature_size = torch.tensor(features.size())
        ctx.save_for_backward(feature_size, torch.tensor(
            [pooled_height, pooled_width, sampling_ratio, spatial_scale]), rois)
        rois = rois
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size, args, rois = ctx.saved_tensors
        assert(feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = tuple(feature_size)
        pooled_height, pooled_width, sampling_ratio, spatial_scale = tuple(args)

        grad_input = grad_output.new().resize_(batch_size, num_channels, data_height, data_width).zero_()
        roi_align_cuda.roi_align_backward(pooled_height, pooled_width, sampling_ratio, spatial_scale,
                                          grad_output, rois, grad_input)

        return grad_input, None, None, None, None, None
