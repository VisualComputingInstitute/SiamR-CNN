#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse
import itertools
import numpy as np
import os
import cv2
import six
import shutil

assert six.PY3, "FasterRCNN requires Python 3!"
import tensorflow as tf
import tqdm

import tensorpack.utils.viz as tpviz
from tensorpack import *
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_tuple, get_tensors_by_names
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.varreplace import freeze_variables

import model_frcnn
import model_mrcnn
from basemodel import image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone, backbone_scope
from dataset import DetectionDataset
from config import finalize_configs, config as cfg
from data import get_all_anchors, get_all_anchors_fpn, get_train_dataflow
from eval_utils import EvalCallback
from model_box import RPNAnchors, clip_boxes, crop_and_resize, roi_align
from model_cascade import CascadeRCNNHead, CascadeRCNNHeadWithHardExamples
from model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses
from model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs, fastrcnn_predictions, sample_fast_rcnn_targets
from model_mrcnn import maskrcnn_loss, maskrcnn_upXconv_head
from model_rpn import generate_rpn_proposals, rpn_head, rpn_losses

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


class DetectionModel(ModelDesc):
    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    @property
    def training(self):
        return get_current_tower_context().is_training

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate in the config is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        if cfg.MODE_THIRD_STAGE:
            out = ['output/boxes', 'output/scores', 'third_stage_features_out', 'ff_gt_tracklet_scores',
                   'sparse_tracklet_scores', 'tracklet_score_indices']
        else:
            out = ['output/boxes', 'output/scores', 'output/labels']
            if cfg.MODE_MASK:
                out.append('output/masks')
        if cfg.EXTRACT_GT_FEATURES:
            return ['image', 'roi_boxes'], ['boxes_for_extraction', 'features_for_extraction']
        else:
            return ['image'], out

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))

        image = self.preprocess(inputs['image'])  # 1CHW

        features = self.backbone(image)
        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        if cfg.EXTRACT_GT_FEATURES:
            anchor_inputs["roi_boxes"] = inputs["roi_boxes"]
        proposals, rpn_losses = self.rpn(image, features, anchor_inputs)  # inputs?

        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
        head_losses = self.roi_heads(image, features, proposals, targets)

        if self.training:
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            total_cost = tf.add_n(
                rpn_losses + head_losses + [wd_cost], 'total_cost')
            add_moving_summary(total_cost, wd_cost)
            return total_cost


class ResNetC4Model(DetectionModel):
    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')]  # all > 0
        if cfg.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
            )  # NR_GT x height x width
        return ret

    def backbone(self, image):
        return [resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS[:3])]

    def rpn(self, image, features, inputs):
        featuremap = features[0]
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)
        anchors = RPNAnchors(get_all_anchors(), inputs['anchor_labels'], inputs['anchor_boxes'])
        anchors = anchors.narrow_to(featuremap)

        image_shape2d = tf.shape(image)[2:]  # h,w
        pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes_decoded, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d,
            cfg.RPN.TRAIN_PRE_NMS_TOPK if self.training else cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TRAIN_POST_NMS_TOPK if self.training else cfg.RPN.TEST_POST_NMS_TOPK)

        if self.training:
            losses = rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(), rpn_label_logits, rpn_box_logits)
        else:
            losses = []

        return BoxProposals(proposal_boxes), losses

    def roi_heads(self, image, features, proposals, targets):
        image_shape2d = tf.shape(image)[2:]  # h,w
        featuremap = features[0]

        gt_boxes, gt_labels, *_ = targets

        if self.training:
            # sample proposal boxes in training
            proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)
        # The boxes to be used to crop RoIs.
        # Use all proposal boxes in inference

        boxes_on_featuremap = proposals.boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        feature_fastrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCKS[-1])  # nxcx7x7
        # Keep C5 feature to be shared with mask branch
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS)

        fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits, gt_boxes,
                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))

        if self.training:
            all_losses = fastrcnn_head.losses()

            if cfg.MODE_MASK:
                gt_masks = targets[2]
                # maskrcnn loss
                # In training, mask branch shares the same C5 feature.
                fg_feature = tf.gather(feature_fastrcnn, proposals.fg_inds())
                mask_logits = maskrcnn_upXconv_head(
                    'maskrcnn', fg_feature, cfg.DATA.NUM_CATEGORY, num_convs=0)  # #fg x #cat x 14x14

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(gt_masks, 1),
                    proposals.fg_boxes(),
                    proposals.fg_inds_wrt_gt, 14,
                    pad_border=False)  # nfg x 1x14x14
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                all_losses.append(maskrcnn_loss(mask_logits, proposals.fg_labels(), target_masks_for_fg))
            return all_losses
        else:
            decoded_boxes = fastrcnn_head.decoded_output_boxes()
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
            final_boxes, final_scores, final_labels = fastrcnn_predictions(
                decoded_boxes, label_scores, name_scope='output')

            if cfg.MODE_MASK:
                roi_resized = roi_align(featuremap, final_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE), 14)
                feature_maskrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCKS[-1])
                mask_logits = maskrcnn_upXconv_head(
                    'maskrcnn', feature_maskrcnn, cfg.DATA.NUM_CATEGORY, 0)  # #result x #cat x 14x14
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx14x14
                tf.sigmoid(final_mask_logits, name='output/masks')
            return []


class ResNetFPNModel(DetectionModel):

    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image')]
        num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
        for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
            ret.extend([
                tf.placeholder(tf.int32, (None, None, num_anchors),
                               'anchor_labels_lvl{}'.format(k + 2)),
                tf.placeholder(tf.float32, (None, None, num_anchors, 4),
                               'anchor_boxes_lvl{}'.format(k + 2))])
        ret.extend([
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')])  # all > 0
        if cfg.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
            )  # NR_GT x height x width
        if cfg.EXTRACT_GT_FEATURES:
            ret.append(tf.placeholder(tf.float32, (None, 4,), 'roi_boxes'))
        return ret

    def slice_feature_and_anchors(self, p23456, anchors):
        for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
            with tf.name_scope('FPN_slice_lvl{}'.format(i)):
                anchors[i] = anchors[i].narrow_to(p23456[i])

    def backbone(self, image):
        c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)
        p23456 = fpn_model('fpn', c2345)
        return p23456

    def rpn(self, image, features, inputs):
        if cfg.EXTRACT_GT_FEATURES:
            boxes = inputs['roi_boxes']
            return BoxProposals(boxes), tf.constant(0, dtype=tf.float32)

        assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)

        image_shape2d = tf.shape(image)[2:]  # h,w
        all_anchors_fpn = get_all_anchors_fpn()
        multilevel_anchors = [RPNAnchors(
            all_anchors_fpn[i],
            inputs['anchor_labels_lvl{}'.format(i + 2)],
            inputs['anchor_boxes_lvl{}'.format(i + 2)]) for i in range(len(all_anchors_fpn))]
        self.slice_feature_and_anchors(features, multilevel_anchors)

        # Multi-Level RPN Proposals
        rpn_outputs = [rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS))
                       for pi in features]
        multilevel_label_logits = [k[0] for k in rpn_outputs]
        multilevel_box_logits = [k[1] for k in rpn_outputs]
        multilevel_pred_boxes = [anchor.decode_logits(logits)
                                 for anchor, logits in zip(multilevel_anchors, multilevel_box_logits)]

        proposal_boxes, proposal_scores = generate_fpn_proposals(
            multilevel_pred_boxes, multilevel_label_logits, image_shape2d)

        if self.training:
            losses = multilevel_rpn_losses(
                multilevel_anchors, multilevel_label_logits, multilevel_box_logits)
        else:
            losses = []

        return BoxProposals(proposal_boxes), losses

    def roi_heads(self, image, features, proposals, targets):
        image_shape2d = tf.shape(image)[2:]  # h,w
        assert len(features) == 5, "Features have to be P23456!"
        gt_boxes, gt_labels, *_ = targets

        if self.training:
            proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)

        fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
        if not cfg.FPN.CASCADE:
            roi_feature_fastrcnn = multilevel_roi_align(features[:4], proposals.boxes, 7)

            head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn)
            fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
                'fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS)
            fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                         gt_boxes, tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))
        else:
            def roi_func(boxes):
                return multilevel_roi_align(features[:4], boxes, 7)

            fastrcnn_head = CascadeRCNNHead(
                proposals, roi_func, fastrcnn_head_func,
                (gt_boxes, gt_labels), image_shape2d, cfg.DATA.NUM_CLASS)

        if cfg.EXTRACT_GT_FEATURES:
            roi_feature_fastrcnn = multilevel_roi_align(features[:4], proposals.boxes, 7)
            tf.identity(roi_feature_fastrcnn, "rpn/feature")

        if self.training:
            all_losses = fastrcnn_head.losses()

            if cfg.MODE_MASK:
                gt_masks = targets[2]
                # maskrcnn loss
                roi_feature_maskrcnn = multilevel_roi_align(
                    features[:4], proposals.fg_boxes(), 14,
                    name_scope='multilevel_roi_align_mask')
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)  # #fg x #cat x 28 x 28

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(gt_masks, 1),
                    proposals.fg_boxes(),
                    proposals.fg_inds_wrt_gt, 28,
                    pad_border=False)  # fg x 1x28x28
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                all_losses.append(maskrcnn_loss(mask_logits, proposals.fg_labels(), target_masks_for_fg))
            return all_losses
        else:
            decoded_boxes = fastrcnn_head.decoded_output_boxes()
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
            final_boxes, final_scores, final_labels = fastrcnn_predictions(
                decoded_boxes, label_scores, name_scope='output')
            if cfg.MODE_MASK:
                # Cascade inference needs roi transform with refined boxes.
                roi_feature_maskrcnn = multilevel_roi_align(features[:4], final_boxes, 14)
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)  # #fg x #cat x 28 x 28
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx28x28
                tf.sigmoid(final_mask_logits, name='output/masks')
            return []


class ResNetFPNTrackModel(ResNetFPNModel):
    def inputs(self):
        ret = super().inputs()
        if cfg.USE_PRECOMPUTED_REF_FEATURES:
            ret.append(tf.placeholder(tf.float32, (256, 7, 7), 'ref_features'))
        else:
            ret.append(tf.placeholder(tf.float32, (None, None, 3), 'ref_image'))
            ret.append(tf.placeholder(tf.float32, (4,), 'ref_box'))
        if cfg.MODE_THIRD_STAGE:
            ret.append(tf.placeholder(tf.float32, (256, 7, 7), 'ff_gt_tracklet_feat'))
            ret.append(tf.placeholder(tf.float32, (None, 256, 7, 7), 'active_tracklets_feats'))
            ret.append(tf.placeholder(tf.float32, (None, 4), 'active_tracklets_boxes'))
            ret.append(tf.placeholder(tf.float32, (), 'tracklet_distance_threshold'))
        if cfg.MODE_HARD_MINING:
            ret.append(tf.placeholder(tf.float32, (None, 3, 256, 7, 7), 'hard_negative_features'))
            if cfg.MODE_IF_HARD_MINING_THEN_ALSO_POSITIVES:
                ret.append(tf.placeholder(tf.float32, (None, 3, 256, 7, 7), 'hard_positive_features'))
                ret.append(tf.placeholder(tf.float32, (None, 3), 'hard_positive_ious'))
                ret.append(tf.placeholder(tf.float32, (None, 4), 'hard_positive_gt_boxes'))
                ret.append(tf.placeholder(tf.float32, (None, 3, 4), 'hard_positive_jitter_boxes'))
        if cfg.EXTRACT_GT_FEATURES:
            ret.append(tf.placeholder(tf.float32, (None, 4,), 'roi_boxes'))
        return ret

    def backbone(self, image):
        c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)
        with backbone_scope(freeze=cfg.BACKBONE.FREEZE_AT > 3):
            p23456 = fpn_model('fpn', c2345)
        return p23456, c2345

    def rpn(self, image, features, inputs):
        if cfg.EXTRACT_GT_FEATURES:
            boxes = inputs['roi_boxes']
            return BoxProposals(boxes), tf.constant(0, dtype=tf.float32)

        if cfg.BACKBONE.FREEZE_AT > 3:
            with freeze_variables(stop_gradient=False, skip_collection=True):
                return super().rpn(image, features, inputs)
        else:
            return super().rpn(image, features, inputs)

    def roi_heads(self, image, ref_features, ref_box, features, proposals, targets, hard_negative_features=None,
                  hard_positive_features=None, hard_positive_ious=None, hard_positive_gt_boxes=None,
                  hard_positive_jitter_boxes=None, precomputed_ref_features=None):
        image_shape2d = tf.shape(image)[2:]  # h,w
        assert len(features) == 5, "Features have to be P23456!"
        gt_boxes, gt_labels, *_ = targets

        if self.training:
            proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)

        fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
        if precomputed_ref_features is None:
            roi_aligned_ref_features = multilevel_roi_align(ref_features[:4], ref_box[tf.newaxis], 7)
        else:
            roi_aligned_ref_features = precomputed_ref_features[tf.newaxis]

        if cfg.MODE_SHARED_CONV_REDUCE:
            scope = tf.get_variable_scope()
        else:
            scope = ""

        assert cfg.FPN.CASCADE

        def roi_func(boxes, already_aligned_features=None):
            if already_aligned_features is None:
                aligned_features = multilevel_roi_align(features[:4], boxes, 7)
            else:
                # for hard example mining
                aligned_features = already_aligned_features
            tiled = tf.tile(roi_aligned_ref_features, [tf.shape(aligned_features)[0], 1, 1, 1])
            concat_features = tf.concat((tiled, aligned_features), axis=1)

            with argscope(Conv2D, data_format='channels_first',
                          kernel_initializer=tf.variance_scaling_initializer(
                              scale=2.0, mode='fan_out',
                              distribution='untruncated_normal' if get_tf_version_tuple() >= (1, 12) else 'normal')):
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    reduced_features = Conv2D('conv_reduce', concat_features, 256, 1, activation=None)
            return reduced_features

        if cfg.MODE_HARD_MINING and self.training:
            fastrcnn_head = CascadeRCNNHeadWithHardExamples(
                proposals, roi_func, fastrcnn_head_func,
                (gt_boxes, gt_labels), image_shape2d, cfg.DATA.NUM_CLASS, hard_negative_features,
                hard_positive_features, cfg.HARD_NEGATIVE_LOSS_SCALING_FACTOR,
                cfg.HARD_POSITIVE_LOSS_SCALING_FACTOR, hard_positive_ious, hard_positive_gt_boxes,
                hard_positive_jitter_boxes)
        else:
            fastrcnn_head = CascadeRCNNHead(
                proposals, roi_func, fastrcnn_head_func,
                (gt_boxes, gt_labels), image_shape2d, cfg.DATA.NUM_CLASS)

        if cfg.EXTRACT_GT_FEATURES:
            # get boxes and features for each of the three cascade stages!
            b0 = proposals.boxes
            b1, b2, _ = fastrcnn_head._cascade_boxes
            f0 = multilevel_roi_align(features[:4], b0, 7)
            f1 = multilevel_roi_align(features[:4], b1, 7)
            f2 = multilevel_roi_align(features[:4], b2, 7)
            tf.concat([b0, b1, b2], axis=0, name="boxes_for_extraction")
            tf.concat([f0, f1, f2], axis=0, name="features_for_extraction")

        if self.training:
            all_losses = fastrcnn_head.losses()

            if cfg.MODE_MASK:
                gt_masks = targets[2]
                # maskrcnn loss
                roi_feature_maskrcnn = multilevel_roi_align(
                    features[:4], proposals.fg_boxes(), 14,
                    name_scope='multilevel_roi_align_mask')
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)  # #fg x #cat x 28 x 28

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(gt_masks, 1),
                    proposals.fg_boxes(),
                    proposals.fg_inds_wrt_gt, 28,
                    pad_border=False)  # fg x 1x28x28
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                all_losses.append(maskrcnn_loss(mask_logits, proposals.fg_labels(), target_masks_for_fg))

            if cfg.MEASURE_IOU_DURING_TRAINING:
                decoded_boxes = fastrcnn_head.decoded_output_boxes()
                decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
                label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
                final_boxes, final_scores, final_labels = fastrcnn_predictions(
                    decoded_boxes, label_scores, name_scope='output_train')
                # if predictions are empty, this might break...
                # to prevent, stack dummy box
                boxes_for_iou = tf.concat([final_boxes[:1], tf.constant([[0.0, 0.0, 1.0, 1.0]],
                                                                        dtype=tf.float32)], axis=0)
                from examples.FasterRCNN.utils.box_ops import pairwise_iou
                iou_at_1 = tf.identity(pairwise_iou(gt_boxes[:1], boxes_for_iou)[0, 0], name="train_iou_at_1")
                add_moving_summary(iou_at_1)

            return all_losses
        else:
            decoded_boxes = fastrcnn_head.decoded_output_boxes()
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
            final_boxes, final_scores, final_labels = fastrcnn_predictions(
                decoded_boxes, label_scores, name_scope='output')
            if cfg.MODE_MASK:
                # Cascade inference needs roi transform with refined boxes.
                roi_feature_maskrcnn = multilevel_roi_align(features[:4], final_boxes, 14)
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)  # #fg x #cat x 28 x 28
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx28x28
                tf.sigmoid(final_mask_logits, name='output/masks')
            return []

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        image = self.preprocess(inputs['image'])  # 1CHW

        fpn_features, backbone_features = self.backbone(image)

        if cfg.USE_PRECOMPUTED_REF_FEATURES:
            ref_features = None
            ref_box = None
        else:
            ref_image = self.preprocess(inputs['ref_image'])  # 1CHW
            ref_box = inputs['ref_box']
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                ref_features, _ = self.backbone(ref_image)

        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        if cfg.EXTRACT_GT_FEATURES:
            anchor_inputs["roi_boxes"] = inputs["roi_boxes"]
        proposals, rpn_losses = self.rpn(image, fpn_features, anchor_inputs)  # inputs?

        second_stage_features = fpn_features
        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]

        hard_negative_features = None
        hard_positive_features = None
        hard_positive_ious = None
        hard_positive_gt_boxes = None
        hard_positive_jitter_boxes = None
        if cfg.MODE_HARD_MINING:
            hard_negative_features = inputs['hard_negative_features']
            if cfg.MODE_IF_HARD_MINING_THEN_ALSO_POSITIVES:
                hard_positive_features = inputs['hard_positive_features']
                hard_positive_ious = inputs['hard_positive_ious']
                hard_positive_gt_boxes = inputs['hard_positive_gt_boxes']
                hard_positive_jitter_boxes = inputs['hard_positive_jitter_boxes']

        precomputed_ref_features = None
        if cfg.USE_PRECOMPUTED_REF_FEATURES:
            precomputed_ref_features = inputs['ref_features']

        # Extend proposals by previous frame detections
        if not self.training and cfg.MODE_THIRD_STAGE and cfg.EXTEND_PROPOSALS_BY_ACTIVE_TRACKLETS:
            proposal_boxes = proposals.boxes
            tracklet_boxes = inputs['active_tracklets_boxes']
            concat_boxes = tf.concat([proposal_boxes, tracklet_boxes], axis=0)
            proposals = BoxProposals(concat_boxes)

        head_losses = self.roi_heads(image, ref_features, ref_box, second_stage_features, proposals, targets,
                                     hard_negative_features, hard_positive_features, hard_positive_ious,
                                     hard_positive_gt_boxes, hard_positive_jitter_boxes,
                                     precomputed_ref_features=precomputed_ref_features)

        if cfg.MODE_THIRD_STAGE:
            self._run_third_stage(inputs, second_stage_features, tf.shape(image)[2:4])

        if self.training:
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            total_cost = tf.add_n(
                rpn_losses + head_losses + [wd_cost], 'total_cost')
            add_moving_summary(total_cost, wd_cost)
            return total_cost

    def _run_third_stage(self, inputs, second_stage_features, image_hw):
        boxes, scores = get_tensors_by_names(['output/boxes', 'output/scores'])
        # let's fix (as in finalize) the boxes, so we can roi align only one time
        aligned_features_curr = multilevel_roi_align(second_stage_features[:4], boxes, 7)
        # these also need to be extracted!
        aligned_features_curr = tf.identity(aligned_features_curr, name='third_stage_features_out')

        ff_gt_tracklet_scores, _ = self._score_for_third_stage(ref_feats=inputs['ff_gt_tracklet_feat'][tf.newaxis],
                                                               det_feats=aligned_features_curr)
        tf.identity(ff_gt_tracklet_scores, name='ff_gt_tracklet_scores')
        sparse_tracklet_scores, tracklet_score_indices = self._score_for_third_stage(
            ref_feats=inputs['active_tracklets_feats'], det_feats=aligned_features_curr,
            dense=False, ref_boxes=inputs['active_tracklets_boxes'], det_boxes=boxes, image_hw=image_hw,
            tracklet_distance_threshold=inputs['tracklet_distance_threshold'])
        tf.identity(sparse_tracklet_scores, name='sparse_tracklet_scores')
        tf.identity(tracklet_score_indices, name='tracklet_score_indices')

    def _score_for_third_stage(self, ref_feats, det_feats, dense=True, ref_boxes=None, det_boxes=None, image_hw=None,
                               tracklet_distance_threshold=0.08):
        # build all pairs
        n_refs = tf.shape(ref_feats)[0]
        n_dets = tf.shape(det_feats)[0]
        active_tracklets_tiled = tf.tile(ref_feats[:, tf.newaxis], multiples=[1, n_dets, 1, 1, 1])
        dets_tiled = tf.tile(det_feats[tf.newaxis], multiples=[n_refs, 1, 1, 1, 1])
        concated = tf.concat([active_tracklets_tiled, dets_tiled], axis=2)

        if not dense:
            # use boxes to prune the connectivity
            assert ref_boxes is not None
            assert det_boxes is not None
            assert image_hw is not None

            def xyxy_to_cxcywh(boxes_xyxy):
                wh = boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]
                c = boxes_xyxy[:, :2] + wh / 2
                boxes_cwh = tf.concat((c, wh), axis=1)
                return boxes_cwh

            active_tracklets_boxes_cxcywh = xyxy_to_cxcywh(ref_boxes)
            boxes_cxcywh = xyxy_to_cxcywh(det_boxes)
            # normalize by image size
            h = image_hw[0]
            w = image_hw[1]
            norm = tf.cast(tf.stack([w, h, w, h], axis=0), tf.float32)
            diffs = tf.abs(active_tracklets_boxes_cxcywh[:, tf.newaxis] - boxes_cxcywh[tf.newaxis]) / norm[
                tf.newaxis, tf.newaxis]

            # use distances of boxes, first frame scores ("scores") to prune
            thresholds = tf.stack([tracklet_distance_threshold] * 4, axis=0)
            keep_mask = tf.reduce_all(diffs < thresholds, axis=2)

            indices = tf.where(keep_mask)
            flattened = tf.boolean_mask(concated, keep_mask)
        else:
            indices = None
            flattened = tf.reshape(
                concated, [tf.shape(concated)[0] * tf.shape(concated)[1]] + [int(x) for x in concated.shape[2:]])

        fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
        if cfg.MODE_SHARED_CONV_REDUCE:
            scope = tf.get_variable_scope()
        else:
            scope = ""
        all_posteriors = []
        # do this for every cascade stage
        for idx in range(3):
            with tf.variable_scope('cascade_rcnn_stage{}'.format(idx + 1), reuse=True):
                with argscope(Conv2D, data_format='channels_first'):
                    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                        reduced_features = Conv2D('conv_reduce', flattened, 256, 1, activation=None)
                    head_feats = fastrcnn_head_func('head', reduced_features)
                    with tf.variable_scope('outputs_new', reuse=True):
                        classification = FullyConnected('class', head_feats, 2)
                        posteriors = tf.nn.softmax(classification)
                        all_posteriors.append(posteriors)
        posteriors = (all_posteriors[0] + all_posteriors[1] + all_posteriors[2]) / tf.constant(3.0, dtype=tf.float32)
        scores = posteriors[:, 1]
        return scores, indices

    def get_inference_tensor_names(self):
        inp, out = super().get_inference_tensor_names()
        if cfg.USE_PRECOMPUTED_REF_FEATURES:
            inp.append('ref_features')
        else:
            inp.append('ref_image')
            inp.append('ref_box')
        if cfg.MODE_THIRD_STAGE:
            inp.append('ff_gt_tracklet_feat')
            inp.append('active_tracklets_feats')
            inp.append('active_tracklets_boxes')
            inp.append('tracklet_distance_threshold')
        return inp, out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/siamrcnn')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetFPNTrackModel()
    DetectionDataset()  # initialize the config with information from our dataset

    is_horovod = cfg.TRAINER == 'horovod'
    if is_horovod:
        hvd.init()
        logger.info("Horovod Rank={}, Size={}".format(hvd.rank(), hvd.size()))

    if not is_horovod or hvd.rank() == 0:
        # keep the old log folder if already existing! (before it would just delete it)
        logger.set_logger_dir(args.logdir, 'k')
        # logger.set_logger_dir(args.logdir, 'd')

    finalize_configs(is_training=True)
    stepnum = cfg.TRAIN.STEPS_PER_EPOCH

    # warmup is step based, lr is epoch based
    init_lr = cfg.TRAIN.WARMUP_INIT_LR * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
    warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
    warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
    lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

    factor = 8. / cfg.TRAIN.NUM_GPUS
    for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule.append(
            (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
    logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
    logger.info("LR Schedule (epochs, value): " + str(lr_schedule))
    train_dataflow = get_train_dataflow()
    # This is what's commonly referred to as "epochs"
    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
    logger.info("Total passes of the training set is: {:.5g}".format(total_passes))

    callbacks = [
                    PeriodicCallback(
                        ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                        # every_k_epochs=1),
                        every_k_epochs=20),
                    # linear warmup
                    ScheduledHyperParamSetter(
                        'learning_rate', warmup_schedule, interp='linear', step_based=True),
                    ScheduledHyperParamSetter('learning_rate', lr_schedule),
                    PeakMemoryTracker(),
                    EstimatedTimeLeft(median=True),
                    SessionRunTimeout(60000).set_chief_only(True),  # 1 minute timeout
                ] + [
                    EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir)
                    for dataset in cfg.DATA.VAL
                ]
    if not is_horovod:
        callbacks.append(GPUUtilizationTracker())

    start_epoch = cfg.TRAIN.STARTING_EPOCH
    if is_horovod and hvd.rank() > 0:
        session_init = None
    else:
        # first try to find existing model
        checkpoint_path = os.path.join(args.logdir, "checkpoint")
        if os.path.exists(checkpoint_path):
            session_init = get_model_loader(checkpoint_path)
            start_step = int(session_init.path.split("-")[-1])
            start_epoch = start_step // stepnum
            logger.info(
                "initializing from existing model, " + session_init.path + ", starting from epoch " + str(start_epoch))
        else:
            if args.load:
                session_init = get_model_loader(args.load)
            else:
                session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

    max_epoch = min(cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum, cfg.TRAIN.MAX_NUM_EPOCHS)

    traincfg = TrainConfig(
        model=MODEL,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        # max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        max_epoch=max_epoch,
        session_init=session_init,
        starting_epoch=start_epoch
    )
    if is_horovod:
        trainer = HorovodTrainer(average=False)
    else:
        # nccl mode appears faster than cpu mode
        trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
    launch_train_with_config(traincfg, trainer)
