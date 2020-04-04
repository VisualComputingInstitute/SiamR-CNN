import tensorflow as tf

from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.summary import add_moving_summary

from config import config as cfg
from model_box import clip_boxes
from model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs
from utils.box_ops import pairwise_iou


class CascadeRCNNHead(object):
    def __init__(self, proposals,
                 roi_func, fastrcnn_head_func, gt_targets, image_shape2d, num_classes):
        """
        Args:
            proposals: BoxProposals
            roi_func (boxes -> features): a function to crop features with rois
            fastrcnn_head_func (features -> features): the fastrcnn head to apply on the cropped features
            gt_targets (gt_boxes, gt_labels):
        """
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.gt_boxes, self.gt_labels = gt_targets
        del self.gt_targets

        self.num_cascade_stages = len(cfg.CASCADE.IOUS)

        self.is_training = get_current_tower_context().is_training
        if self.is_training:
            @tf.custom_gradient
            def scale_gradient(x):
                return x, lambda dy: dy * (1.0 / self.num_cascade_stages)
            self.scale_gradient = scale_gradient
        else:
            self.scale_gradient = tf.identity

        ious = cfg.CASCADE.IOUS
        # It's unclear how to do >3 stages, so it does not make sense to implement them
        assert self.num_cascade_stages == 3, "Only 3-stage cascade was implemented!"
        with tf.variable_scope('cascade_rcnn_stage1'):
            H1, B1 = self.run_head(self.proposals, 0)

        with tf.variable_scope('cascade_rcnn_stage2'):
            B1_proposal = self.match_box_with_gt(B1, ious[1])
            H2, B2 = self.run_head(B1_proposal, 1)

        with tf.variable_scope('cascade_rcnn_stage3'):
            B2_proposal = self.match_box_with_gt(B2, ious[2])
            H3, B3 = self.run_head(B2_proposal, 2)
        self._cascade_boxes = [B1, B2, B3]
        self._heads = [H1, H2, H3]

    def run_head(self, proposals, stage):
        """
        Args:
            proposals: BoxProposals
            stage: 0, 1, 2

        Returns:
            FastRCNNHead
            Nx4, updated boxes
        """
        reg_weights = tf.constant(cfg.CASCADE.BBOX_REG_WEIGHTS[stage], dtype=tf.float32)
        pooled_feature = self.roi_func(proposals.boxes)  # N,C,S,S
        pooled_feature = self.scale_gradient(pooled_feature)
        head_feature = self.fastrcnn_head_func('head', pooled_feature)
        # changed by Paul
        label_logits, box_logits = fastrcnn_outputs(
            'outputs_new', head_feature, self.num_classes, class_agnostic_regression=True)
        head = FastRCNNHead(proposals, box_logits, label_logits, self.gt_boxes, reg_weights)

        refined_boxes = head.decoded_output_boxes_class_agnostic()
        refined_boxes = clip_boxes(refined_boxes, self.image_shape2d)
        return head, tf.stop_gradient(refined_boxes, name='output_boxes')

    def match_box_with_gt(self, boxes, iou_threshold):
        """
        Args:
            boxes: Nx4
        Returns:
            BoxProposals
        """
        if self.is_training:
            with tf.name_scope('match_box_with_gt_{}'.format(iou_threshold)):
                iou = pairwise_iou(boxes, self.gt_boxes)  # NxM
                max_iou_per_box = tf.reduce_max(iou, axis=1)  # N
                best_iou_ind = tf.argmax(iou, axis=1)  # N
                labels_per_box = tf.gather(self.gt_labels, best_iou_ind)
                fg_mask = max_iou_per_box >= iou_threshold
                fg_inds_wrt_gt = tf.boolean_mask(best_iou_ind, fg_mask)
                labels_per_box = tf.stop_gradient(labels_per_box * tf.cast(fg_mask, tf.int64))
                return BoxProposals(boxes, labels_per_box, fg_inds_wrt_gt)
        else:
            return BoxProposals(boxes)

    def losses(self):
        ret = []
        for idx, head in enumerate(self._heads):
            with tf.name_scope('cascade_loss_stage{}'.format(idx + 1)):
                ret.extend(head.losses())
        return ret

    def decoded_output_boxes(self):
        """
        Returns:
            Nx#classx4
        """
        ret = self._cascade_boxes[-1]
        ret = tf.expand_dims(ret, 1)     # class-agnostic
        return tf.tile(ret, [1, self.num_classes, 1])

    def output_scores(self, name=None):
        """
        Returns:
            Nx#class
        """
        scores = [head.output_scores('cascade_scores_stage{}'.format(idx + 1))
                  for idx, head in enumerate(self._heads)]
        return tf.multiply(tf.add_n(scores), (1.0 / self.num_cascade_stages), name=name)


class CascadeRCNNHeadWithHardExamples(CascadeRCNNHead):
    def __init__(self, proposals, roi_func, fastrcnn_head_func, gt_targets, image_shape2d, num_classes,
                 hard_negative_features, hard_positive_features, hard_negative_loss_scaling_factor,
                 hard_positive_loss_scaling_factor, hard_positive_ious, hard_positive_gt_boxes,
                 hard_positive_jitter_boxes):
        super().__init__(proposals, roi_func, fastrcnn_head_func, gt_targets, image_shape2d, num_classes)
        self._hard_negative_features = hard_negative_features
        self._hard_positive_features = hard_positive_features
        self._hard_negative_loss_scaling_factor = hard_negative_loss_scaling_factor
        self._hard_positive_loss_scaling_factor = hard_positive_loss_scaling_factor
        self._hard_positive_ious = hard_positive_ious
        self._hard_positive_gt_boxes = hard_positive_gt_boxes
        self._hard_positive_jitter_boxes = hard_positive_jitter_boxes

    def _hard_losses(self, negative=True):
        if negative:
            hard_features = self._hard_negative_features
            desc = "neg"
        else:
            hard_features = self._hard_positive_features
            desc = "pos"
        losses = []
        for cascade_idx, iou_thres in enumerate(cfg.CASCADE.IOUS):
            with tf.name_scope('cascade_loss_{}_stage{}'.format(desc, cascade_idx + 1)):
                with tf.variable_scope('cascade_rcnn_stage' + str(cascade_idx + 1), reuse=True):
                    pooled_feature = self.roi_func(None, hard_features[:, cascade_idx])
                    pooled_feature = self.scale_gradient(pooled_feature)
                    head_feature = self.fastrcnn_head_func('head', pooled_feature)
                    # changed by Paul
                    label_logits, box_logits = fastrcnn_outputs(
                        'outputs_new', head_feature, self.num_classes, class_agnostic_regression=True)
                    mean_label = None
                    box_loss = None
                    if negative:
                        labels = tf.zeros((tf.shape(label_logits)[0],), dtype=tf.int64)
                    else:
                        labels = tf.cast(tf.greater_equal(self._hard_positive_ious[:, cascade_idx], iou_thres),
                                         tf.int64)
                        mean_label = tf.reduce_mean(tf.cast(labels, tf.float32),
                                                    name='hard_{}_label_mean{}'.format(desc, cascade_idx + 1))
                        if cfg.USE_REGRESSION_LOSS_ON_HARD_POSITIVES:
                            labels_bool = tf.cast(labels, tf.bool)
                            valid = tf.reduce_any(labels_bool)

                            def make_box_loss():
                                gt_boxes = tf.boolean_mask(self._hard_positive_gt_boxes, labels_bool)
                                inp_boxes = tf.boolean_mask(self._hard_positive_jitter_boxes[:, cascade_idx],
                                                            labels_bool)
                                box_logits_masked = tf.boolean_mask(box_logits, labels_bool)
                                from examples.FasterRCNN.model_box import encode_bbox_target
                                reg_targets = encode_bbox_target(gt_boxes,
                                                                 inp_boxes) * cfg.CASCADE.BBOX_REG_WEIGHTS[cascade_idx]
                                _box_loss = tf.losses.huber_loss(
                                    reg_targets, tf.squeeze(box_logits_masked, axis=1),
                                    reduction=tf.losses.Reduction.SUM)
                                _box_loss = tf.truediv(
                                    _box_loss, tf.cast(tf.shape(reg_targets)[0], tf.float32))
                                return _box_loss

                            box_loss = tf.cond(valid, make_box_loss, lambda: tf.constant(0, dtype=tf.float32))
                            box_loss = tf.multiply(box_loss, cfg.HARD_POSITIVE_BOX_LOSS_SCALING_FACTOR,
                                                   name='hard_{}_box_loss{}'.format(desc, cascade_idx + 1))
                            losses.append(box_loss)
                    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=label_logits)
                    if negative:
                        label_loss *= self._hard_negative_loss_scaling_factor
                    else:
                        label_loss *= self._hard_positive_loss_scaling_factor
                    label_loss = tf.reduce_mean(label_loss, name='hard_{}_label_loss{}'.format(desc, cascade_idx + 1))
                    prediction = tf.argmax(label_logits, axis=1, name='label_prediction_hard_{}'.format(desc))
                    correct = tf.cast(tf.equal(prediction, labels), tf.float32)
                    accuracy = tf.reduce_mean(correct, name='hard_{}_label_accuracy{}'.format(desc, cascade_idx + 1))
                    losses.append(label_loss)
                if mean_label is not None:
                    add_moving_summary(mean_label)
                if box_loss is not None:
                    add_moving_summary(box_loss)
                add_moving_summary(accuracy)
                add_moving_summary(label_loss)
        return losses

    def losses(self):
        normal_losses = super().losses()
        if self.is_training:
            hnl = self._hard_losses(negative=True)
            if self._hard_positive_features is not None:
                hpl = self._hard_losses(negative=False)
            else:
                hpl = []
            return normal_losses + hnl + hpl
        else:
            return normal_losses
