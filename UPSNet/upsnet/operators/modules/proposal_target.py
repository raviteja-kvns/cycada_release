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

from torch.nn.modules.module import Module
from ..functions.proposal_target import ProposalTargetFunction


class ProposalTarget(Module):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(ProposalTarget, self).__init__()
        self.num_classes = num_classes
        self.batch_images = batch_images
        self.batch_rois = batch_rois
        self.fg_fraction = fg_fraction

    def forward(self, rois, gt_boxes):
        return ProposalTargetFunction.apply(rois, gt_boxes, self.num_classes, self.batch_images, self.batch_rois, self.fg_fraction)
