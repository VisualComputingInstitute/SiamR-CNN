# -*- coding: utf-8 -*-
# File: data.py

import copy
import platform
import numpy as np
import bisect
import cv2
import glob
import random
import os
import PIL
from tabulate import tabulate
from termcolor import colored
import xmltodict

from tensorpack.dataflow import (
    DataFromList, MapDataComponent, MultiProcessMapDataZMQ, MultiThreadMapData, MapData, TestDataSpeed, imgaug)
from tensorpack.utils import logger
from tensorpack.utils.argtools import log_once, memoized

from common import (
    CustomResize, DataFromListOfDict, box_to_point8,
    filter_boxes_inside_shape, point8_to_box, segmentation_to_mask, np_iou)
from config import config as cfg
from dataset import DetectionDataset
from hard_example_utils import subsample_nns
from utils.generate_anchors import generate_anchors
from utils.np_box_ops import area as np_area, ioa as np_ioa

# import tensorpack.utils.viz as tpviz


class MalformedData(BaseException):
    pass


def print_class_histogram(roidbs):
    """
    Args:
        roidbs (list[dict]): the same format as the output of `load_training_roidbs`.
    """
    dataset = DetectionDataset()
    hist_bins = np.arange(dataset.num_classes + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((dataset.num_classes,), dtype=np.int)
    for entry in roidbs:
        # filter crowd?
        gt_inds = np.where(
            (entry['class'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['class'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    data = [[dataset.class_names[i], v] for i, v in enumerate(gt_hist)]
    data.append(['total', sum([x[1] for x in data])])
    table = tabulate(data, headers=['class', '#box'], tablefmt='pipe')
    logger.info("Ground-Truth Boxes:\n" + colored(table, 'cyan'))


@memoized
def get_all_anchors(stride=None, sizes=None):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

    """
    if stride is None:
        stride = cfg.RPN.ANCHOR_STRIDE
    if sizes is None:
        sizes = cfg.RPN.ANCHOR_SIZES
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array(sizes, dtype=np.float) / stride,
        ratios=np.array(cfg.RPN.ANCHOR_RATIOS, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    max_size = cfg.PREPROC.MAX_SIZE
    field_size = int(np.ceil(max_size / stride))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors


@memoized
def get_all_anchors_fpn(strides=None, sizes=None):
    """
    Returns:
        [anchors]: each anchors is a SxSx NUM_ANCHOR_RATIOS x4 array.
    """
    if strides is None:
        strides = cfg.FPN.ANCHOR_STRIDES
    if sizes is None:
        sizes = cfg.RPN.ANCHOR_SIZES
    assert len(strides) == len(sizes)
    foas = []
    for stride, size in zip(strides, sizes):
        foa = get_all_anchors(stride=stride, sizes=(size,))
        foas.append(foa)
    return foas


def get_anchor_labels(anchors, gt_boxes, crowd_boxes):
    """
    Label each anchor as fg/bg/ignore.
    Args:
        anchors: Ax4 float
        gt_boxes: Bx4 float, non-crowd
        crowd_boxes: Cx4 float

    Returns:
        anchor_labels: (A,) int. Each element is {-1, 0, 1}
        anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
    """
    # This function will modify labels and return the filtered inds
    def filter_box_label(labels, value, max_num):
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            disable_inds = np.random.choice(
                curr_inds, size=(len(curr_inds) - max_num),
                replace=False)
            labels[disable_inds] = -1    # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0  # empty images should have been filtered already
    box_ious = np_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

    # Setting NA labels: 1--fg 0--bg -1--ignore
    anchor_labels = -np.ones((NA,), dtype='int32')   # NA,

    # the order of setting neg/pos labels matter
    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[ious_max_per_anchor >= cfg.RPN.POSITIVE_ANCHOR_THRESH] = 1
    anchor_labels[ious_max_per_anchor < cfg.RPN.NEGATIVE_ANCHOR_THRESH] = 0

    # label all non-ignore candidate boxes which overlap crowd as ignore
    if crowd_boxes.size > 0:
        cand_inds = np.where(anchor_labels >= 0)[0]
        cand_anchors = anchors[cand_inds]
        ioas = np_ioa(crowd_boxes, cand_anchors)
        overlap_with_crowd = cand_inds[ioas.max(axis=0) > cfg.RPN.CROWD_OVERLAP_THRESH]
        anchor_labels[overlap_with_crowd] = -1

    # Subsample fg labels: ignore some fg if fg is too many
    target_num_fg = int(cfg.RPN.BATCH_PER_IM * cfg.RPN.FG_RATIO)
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
    # Keep an image even if there is no foreground anchors
    # if len(fg_inds) == 0:
    #     raise MalformedData("No valid foreground for RPN!")

    # Subsample bg labels. num_bg is not allowed to be too many
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0:
        # No valid bg in this image, skip.
        raise MalformedData("No valid background for RPN!")
    target_num_bg = cfg.RPN.BATCH_PER_IM - len(fg_inds)
    filter_box_label(anchor_labels, 0, target_num_bg)   # ignore return values

    # Set anchor boxes: the best gt_box for each fg anchor
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
    anchor_boxes[fg_inds, :] = fg_boxes
    # assert len(fg_inds) + np.sum(anchor_labels == 0) == cfg.RPN.BATCH_PER_IM
    return anchor_labels, anchor_boxes


def get_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        The anchor labels and target boxes for each pixel in the featuremap.
        fm_labels: fHxfWxNA
        fm_boxes: fHxfWxNAx4
        NA will be NUM_ANCHOR_SIZES x NUM_ANCHOR_RATIOS
    """
    boxes = boxes.copy()
    all_anchors = np.copy(get_all_anchors())
    # fHxfWxAx4 -> (-1, 4)
    featuremap_anchors_flatten = all_anchors.reshape((-1, 4))

    # only use anchors inside the image
    inside_ind, inside_anchors = filter_boxes_inside_shape(featuremap_anchors_flatten, im.shape[:2])
    # obtain anchor labels and their corresponding gt boxes
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # Fill them back to original size: fHxfWx1, fHxfWx4
    anchorH, anchorW = all_anchors.shape[:2]
    featuremap_labels = -np.ones((anchorH * anchorW * cfg.RPN.NUM_ANCHOR, ), dtype='int32')
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape((anchorH, anchorW, cfg.RPN.NUM_ANCHOR))
    featuremap_boxes = np.zeros((anchorH * anchorW * cfg.RPN.NUM_ANCHOR, 4), dtype='float32')
    featuremap_boxes[inside_ind, :] = anchor_gt_boxes
    featuremap_boxes = featuremap_boxes.reshape((anchorH, anchorW, cfg.RPN.NUM_ANCHOR, 4))
    return featuremap_labels, featuremap_boxes


def get_multilevel_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        [(fm_labels, fm_boxes)]: Returns a tuple for each FPN level.
        Each tuple contains the anchor labels and target boxes for each pixel in the featuremap.

        fm_labels: fHxfWx NUM_ANCHOR_RATIOS
        fm_boxes: fHxfWx NUM_ANCHOR_RATIOS x4
    """
    boxes = boxes.copy()
    anchors_per_level = get_all_anchors_fpn()
    flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
    all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)

    inside_ind, inside_anchors = filter_boxes_inside_shape(all_anchors_flatten, im.shape[:2])
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # map back to all_anchors, then split to each level
    num_all_anchors = all_anchors_flatten.shape[0]
    all_labels = -np.ones((num_all_anchors, ), dtype='int32')
    all_labels[inside_ind] = anchor_labels
    all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')
    all_boxes[inside_ind] = anchor_gt_boxes

    start = 0
    multilevel_inputs = []
    for level_anchor in anchors_per_level:
        assert level_anchor.shape[2] == len(cfg.RPN.ANCHOR_RATIOS)
        anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
        num_anchor_this_level = np.prod(anchor_shape)
        end = start + num_anchor_this_level
        multilevel_inputs.append(
            (all_labels[start: end].reshape(anchor_shape),
             all_boxes[start: end, :].reshape(anchor_shape + (4,))
             ))
        start = end
    assert end == num_all_anchors, "{} != {}".format(end, num_all_anchors)
    return multilevel_inputs


def get_bbox_from_segmentation_mask_np(mask):
    object_locations = (np.stack(np.where(np.equal(mask, 1))).T[:, :2]).astype(np.int32)
    y0 = np.min(object_locations[:, 0])
    x0 = np.min(object_locations[:, 1])
    y1 = np.max(object_locations[:, 0]) + 1
    x1 = np.max(object_locations[:, 1]) + 1
    bbox = np.stack([x0, y0, x1, y1])
    return bbox


def _augment_boxes(boxes, aug, params):
    points = box_to_point8(boxes)
    points = aug.augment_coords(points, params)
    boxes = point8_to_box(points)
    #assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"
    if np.min(np_area(boxes)) <= 0:
        return None
    return boxes


def _preprocess_common(ref_box, target_box, ref_im, target_im, aug):
    ref_boxes = np.array([ref_box], dtype=np.float32)
    target_boxes = np.array([target_box], dtype=np.float32)
    klass = np.array([1], dtype=np.int32)

    # augmentation:
    target_im, target_params = aug.augment_return_params(target_im)
    ref_im, ref_params = aug.augment_return_params(ref_im)
    ref_boxes = _augment_boxes(ref_boxes, aug, ref_params)
    target_boxes = _augment_boxes(target_boxes, aug, target_params)
    if ref_boxes is None or target_boxes is None:
        return None

    # additional augmentations:
    # motion blur
    if cfg.DATA.MOTION_BLUR_AUGMENTATIONS:
        do_motion_blur_ref = np.random.rand() < 0.25
        if do_motion_blur_ref:
            # generating the kernel
            kernel_size = np.random.randint(5, 15)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur = kernel_motion_blur / kernel_size
            # applying the kernel
            ref_im = cv2.filter2D(ref_im, -1, kernel_motion_blur)
        do_motion_blur_target = np.random.rand() < 0.25
        if do_motion_blur_target:
            # generating the kernel
            kernel_size = np.random.randint(5, 15)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur = kernel_motion_blur / kernel_size
            # applying the kernel
            target_im = cv2.filter2D(target_im, -1, kernel_motion_blur)

    # grayscale
    if cfg.DATA.GRAYSCALE_AUGMENTATIONS:
        do_grayscale = np.random.rand() < 0.25
        if do_grayscale:
            grayscale_aug = imgaug.Grayscale()
            ref_im = np.tile(grayscale_aug.augment(ref_im), [1, 1, 3])
            target_im = np.tile(grayscale_aug.augment(target_im), [1, 1, 3])

    if cfg.DATA.DEBUG_VIS:
        import matplotlib.pyplot as plt
        ref_im_vis = ref_im.copy()
        #ref_im_vis[int(ref_boxes[0][1]):int(ref_boxes[0][3]), int(ref_boxes[0][0]):int(ref_boxes[0][2]), 0] = 255
        ref_im_vis[int(ref_boxes[0][1]):int(ref_boxes[0][3]), int(ref_boxes[0][0]):int(ref_boxes[0][2]), 2] = \
            (0.5 * ref_im_vis[int(ref_boxes[0][1]):int(ref_boxes[0][3]), int(ref_boxes[0][0]):int(ref_boxes[0][2]), 2] + 120).astype(np.uint8)
        plt.imshow(ref_im_vis[..., ::-1])
        plt.show()
        target_im_vis = target_im.copy()
        target_im_vis[int(target_boxes[0][1]):int(target_boxes[0][3]), int(target_boxes[0][0]):int(target_boxes[0][2]), 2] = \
            (0.5 * target_im_vis[int(target_boxes[0][1]):int(target_boxes[0][3]), int(target_boxes[0][0]):int(target_boxes[0][2]), 2] + 120).astype(np.uint8)
        plt.imshow(target_im_vis[..., ::-1])
        plt.show()

    is_crowd = np.array([0], dtype=np.int32)
    ret = {'ref_image': ref_im, 'ref_box': ref_boxes[0], 'image': target_im}
    if cfg.DATA.DEBUG_VIS:
        return ret

    # rpn anchor:
    try:
        if cfg.MODE_FPN:
            multilevel_anchor_inputs = get_multilevel_rpn_anchor_input(target_im, target_boxes, is_crowd)
            for i, (anchor_labels, anchor_boxes) in enumerate(multilevel_anchor_inputs):
                ret['anchor_labels_lvl{}'.format(i + 2)] = anchor_labels
                ret['anchor_boxes_lvl{}'.format(i + 2)] = anchor_boxes
        else:
            # anchor_labels, anchor_boxes
            ret['anchor_labels'], ret['anchor_boxes'] = get_rpn_anchor_input(target_im, target_boxes, is_crowd)
        ret['gt_boxes'] = target_boxes
        ret['gt_labels'] = klass
        if not len(target_boxes):
            raise MalformedData("No valid gt_boxes!")
    except MalformedData as e:
        log_once("Input is filtered for training: {}".format(str(e)), 'warn')
        return None
    return ret


def _preprocess_imagenet_vid(roidb, aug, hard_example_index, hard_example_names):
    vid_name = roidb
    ann_path = os.path.join(cfg.DATA.IMAGENET_VID_ROOT, "Annotations/VID/train/", vid_name)
    ann_files = sorted(glob.glob(ann_path + "/*.xml"))
    # randomly select two files
    ref_idx = np.random.randint(len(ann_files))
    target_idx = np.random.randint(1, len(ann_files))
    ref_ann_file = ann_files[ref_idx]
    target_ann_file = ann_files[target_idx]

    def get_id_to_data(ann):
        id_to_data = {}
        if "object" in ann:
            obj_anns = ann["object"]
            if not isinstance(obj_anns, list):
                obj_anns = [obj_anns]
            for obj_ann in obj_anns:
                id_ = obj_ann["trackid"]
                id_to_data[id_] = obj_ann
        return id_to_data

    ref_ann = xmltodict.parse(open(ref_ann_file).read())["annotation"]
    target_ann = xmltodict.parse(open(target_ann_file).read())["annotation"]
    ref_id_to_data = get_id_to_data(ref_ann)
    target_id_to_data = get_id_to_data(target_ann)
    ref_obj_ids = set(ref_id_to_data.keys())
    target_obj_ids = set(target_id_to_data.keys())
    obj_ids = ref_obj_ids & target_obj_ids
    obj_ids = list(obj_ids)
    if len(obj_ids) == 0:
        # this happens quite often, do not print it for now
        #log_once("Inputs {},{} filtered for training because of no common objects".format(ref_fname, target_fname),
        #         'warn')
        return None
    random.shuffle(obj_ids)
    obj_id = obj_ids[0]

    def obj_data_to_bbox(obj_ann):
        bbox = obj_ann['bndbox']
        x1 = bbox['xmin']
        y1 = bbox['ymin']
        x2 = bbox['xmax']
        y2 = bbox['ymax']
        box = [x1, y1, x2, y2]
        return box

    ref_ann = ref_id_to_data[obj_id]
    target_ann = target_id_to_data[obj_id]
    ref_box = obj_data_to_bbox(ref_ann)
    target_box = obj_data_to_bbox(target_ann)
    ref_fname = ref_ann_file.replace("/Annotations/", "/Data/").replace(".xml", ".JPEG")
    ref_im = cv2.imread(ref_fname, cv2.IMREAD_COLOR)
    target_fname = target_ann_file.replace("/Annotations/", "/Data/").replace(".xml", ".JPEG")
    target_im = cv2.imread(target_fname, cv2.IMREAD_COLOR)
    data = _preprocess_common(ref_box, target_box, ref_im, target_im, aug)
    vid_name = roidb.replace("/", "_") + "_" + str(obj_id)
    return _maybe_add_hard_example_data(data, ref_fname, vid_name, hard_example_index,
                                        hard_example_names, dataset_name="ImageNetVID")


def _preprocess_davis_like(roidb, aug, ann_folder, dataset_name="YouTubeVOS", hard_example_index=None,
                           hard_example_names=None):
    vid_name = roidb
    ann_path = os.path.join(ann_folder, vid_name)
    ann_files = sorted(glob.glob(ann_path + "/*.png"))
    if len(ann_files) == 0:
        logger.info("no annotations found, skipping {}...".format(ann_path))
        return None
    # randomly select two files
    ref_idx = np.random.randint(len(ann_files))
    target_idx = np.random.randint(1, len(ann_files))
    ref_ann_file = ann_files[ref_idx]
    target_ann_file = ann_files[target_idx]

    ref_masks = np.array(PIL.Image.open(ref_ann_file))
    target_masks = np.array(PIL.Image.open(target_ann_file))
    ref_obj_ids = set(np.setdiff1d(np.unique(ref_masks), [0]))
    target_obj_ids = set(np.setdiff1d(np.unique(target_masks), [0]))
    obj_ids = ref_obj_ids & target_obj_ids
    obj_ids = list(obj_ids)
    if len(obj_ids) == 0:
        # this happens quite often, do not print it for now
        # log_once("Inputs {},{} filtered for training because of no common objects".format(ref_fname, target_fname),
        #         'warn')
        return None
    random.shuffle(obj_ids)
    obj_id = obj_ids[0]
    ref_mask = ref_masks == obj_id
    target_mask = target_masks == obj_id
    # convert mask to bbox!
    ref_box = get_bbox_from_segmentation_mask_np(ref_mask)
    target_box = get_bbox_from_segmentation_mask_np(target_mask)
    ref_fname = ref_ann_file.replace("/Annotations/", "/JPEGImages/").replace("/Annotations/", "/JPEGImages/").replace(".png", ".jpg")
    ref_im = cv2.imread(ref_fname, cv2.IMREAD_COLOR)
    target_fname = target_ann_file.replace("/Annotations/", "/JPEGImages/").replace("/Annotations/", "/JPEGImages/").replace(".png", ".jpg")
    target_im = cv2.imread(target_fname, cv2.IMREAD_COLOR)
    data = _preprocess_common(ref_box, target_box, ref_im, target_im, aug)
    vid_name = roidb + "_" + str(obj_id)
    return _maybe_add_hard_example_data(data, ref_fname, vid_name, hard_example_index,
                                        hard_example_names, dataset_name=dataset_name)


def _preprocess_lasot(roidb, aug, hard_example_index, hard_example_names):
    category = roidb.split("-")[0]
    data_path = os.path.join(cfg.DATA.LASOT_ROOT, category, roidb)
    gt_file = os.path.join(data_path, "groundtruth.txt")
    oov_file = os.path.join(data_path, "out_of_view.txt")
    full_occ_file = os.path.join(data_path, "full_occlusion.txt")
    boxes = []
    with open(gt_file) as f:
        for l in f:
            sp = l.strip().split(",")
            box = [float(x) for x in sp]
            box[2] += box[0]
            box[3] += box[1]
            boxes.append(box)
    with open(oov_file) as f:
        sp = f.read().strip().split(",")
        oovs = [int(x) for x in sp]
    with open(full_occ_file) as f:
        sp = f.read().strip().split(",")
        full_occs = [int(x) for x in sp]
    n_frames = len(boxes)
    assert len(boxes) == len(oovs) == len(full_occs)
    data = list(zip(range(n_frames), boxes, oovs, full_occs))
    data = [x for x in data if x[2] == 0 and x[3] == 0]
    assert len(data) > 0
    n_frames = len(data)
    ref_idx = np.random.randint(n_frames)
    target_idx = np.random.randint(n_frames)
    ref_box = data[ref_idx][1]
    target_box = data[target_idx][1]
    ref_time_idx = data[ref_idx][0]
    target_time_idx = data[target_idx][0]
    ref_fname = os.path.join(data_path, "img", "%08d.jpg" % (ref_time_idx + 1))
    ref_im = cv2.imread(ref_fname, cv2.IMREAD_COLOR)
    target_fname = os.path.join(data_path, "img", "%08d.jpg" % (target_time_idx + 1))
    target_im = cv2.imread(target_fname, cv2.IMREAD_COLOR)
    data = _preprocess_common(ref_box, target_box, ref_im, target_im, aug)
    return _maybe_add_hard_example_data(data, ref_fname, roidb, hard_example_index,
                                        hard_example_names, dataset_name="LaSOT")


def _preprocess_got10k(roidb, aug, hard_example_index, hard_example_names):
    vid_name = roidb
    data_path = os.path.join(cfg.DATA.GOT10K_ROOT, "train", vid_name)
    gt_file = os.path.join(data_path, "groundtruth.txt")
    absent_file = os.path.join(data_path, "absence.label")
    boxes = []
    absence = []
    with open(gt_file) as f:
        for l in f:
            sp = l.strip().split(",")
            box = [float(x) for x in sp]
            box[2] += box[0]
            box[3] += box[1]
            boxes.append(box)
    with open(absent_file) as f:
        for l in f:
            absent = int(l.strip())
            absence.append(absent)
    n_frames = len(boxes)
    assert len(boxes) == len(absence)
    data = list(zip(range(n_frames), boxes, absence))
    data = [x for x in data if x[2] == 0]
    assert len(data) > 0
    n_frames = len(data)
    ref_idx = np.random.randint(n_frames)
    target_idx = np.random.randint(n_frames)
    ref_box = data[ref_idx][1]
    target_box = data[target_idx][1]
    ref_time_idx = data[ref_idx][0]
    target_time_idx = data[target_idx][0]
    ref_fname = os.path.join(data_path, "%08d.jpg" % (ref_time_idx + 1))
    ref_im = cv2.imread(ref_fname, cv2.IMREAD_COLOR)
    target_fname = os.path.join(data_path, "%08d.jpg" % (target_time_idx + 1))
    target_im = cv2.imread(target_fname, cv2.IMREAD_COLOR)
    data = _preprocess_common(ref_box, target_box, ref_im, target_im, aug)
    return _maybe_add_hard_example_data(data, ref_fname, vid_name, hard_example_index, hard_example_names,
                                        dataset_name="GOT10k")


def _maybe_add_hard_example_data(data, ref_fname, vid_name, hard_example_index, hard_example_names, dataset_name):
    if not cfg.MODE_HARD_MINING:
        return data
    data = data.copy()
    name_for_idx = dataset_name + "/" + vid_name + "/"

    if dataset_name == "GOT10k":
        name_for_idx += ref_fname.split("/")[-1].replace(".jpg", "")
        this_fmt = "%08d"
    elif dataset_name == "ImageNetVID":
        name_for_idx += str(int(ref_fname.split("/")[-1].replace(".JPEG", "")))
        this_fmt = "%06d"
    elif dataset_name == "LaSOT":
        name_for_idx += str(int(ref_fname.split("/")[-1].replace(".jpg", "")))
        this_fmt = "%08d"
    elif dataset_name == "YouTubeVOS":
        name_for_idx += str(int(ref_fname.split("/")[-1].replace(".jpg", "")))
        this_fmt = "%05d"
    else:
        assert False, ("unknown dataset", dataset_name)

    try:
        idx = hard_example_names["all"].index(name_for_idx)
    except ValueError:
        log_once("Not found in index: {}".format(name_for_idx), 'warn')
        return None
    if dataset_name == "LaSOT":
        nns = hard_example_index.get_nns_by_item(idx, cfg.HARD_MINING_KNN_LASOT)
    else:
        nns = hard_example_index.get_nns_by_item(idx, cfg.HARD_MINING_KNN)
    if cfg.MODE_HARD_NEGATIVES_ONLY_CROSSOVER or \
          (cfg.MODE_HARD_NEGATIVES_ONLY_CROSSOVER_YOUTUBEVOS and dataset_name == "YouTubeVOS"):
        nn_names = [hard_example_names["all"][nn] for nn in nns]
        nn_datasets = [x.split("/")[0] for x in nn_names]
        nns = [nn for nn, ds_ in zip(nns, nn_datasets) if ds_ != dataset_name]
        remove_query = False
    else:
        remove_query = True
    nns = subsample_nns(vid_name, nns, hard_example_names["all"], cfg.N_HARD_NEGATIVES_TO_SAMPLE,
                        remove_query=remove_query)

    feats = []
    for nn in nns:
        sp = hard_example_names["all"][nn].split("/")
        if sp[0] == "GOT10k":
            fmt = "%08d"
        elif sp[0] == "ImageNetVID":
            fmt = "%06d"
        elif sp[0] == "LaSOT":
            fmt = "%08d"
        elif sp[0] == "YouTubeVOS":
            fmt = "%05d"
        else:
            assert False, ("unknown dataset", sp[0])

        feat_fn = os.path.join(cfg.HARD_MINING_DATA_PATH, sp[0], "det_feats_compressed", sp[1],
                               fmt % int(sp[2]) + ".npz")
        feat = np.load(feat_fn)
        feat = feat["f"]
        feats.append(feat)
    feats = np.stack(feats, axis=0)
    data['hard_negative_features'] = feats

    if cfg.MODE_IF_HARD_MINING_THEN_ALSO_POSITIVES:
        hard_example_names_dataset = hard_example_names[dataset_name]

        #hpens_oldversion = [x for x in hard_example_names_dataset if x.startswith(vid_name)]
        left = right = bisect.bisect_left(hard_example_names_dataset, vid_name)
        while left > 0:
            if hard_example_names_dataset[left - 1].startswith(vid_name):
                left -= 1
            else:
                break
        while right < len(hard_example_names_dataset):
            if hard_example_names_dataset[right].startswith(vid_name):
                right += 1
            else:
                break
        hpens = hard_example_names_dataset[left:right]

        assert len(hpens) > 0, vid_name
        random.shuffle(hpens)
        hpens = hpens[:cfg.N_HARD_POS_TO_SAMPLE]
        feats = []
        ious = []
        gt_boxes = []
        jitter_boxes = []
        for hpen in hpens:
            sp = hpen.split("/")
            feat_fn = os.path.join(cfg.HARD_MINING_DATA_PATH, dataset_name, "det_feats_compressed", sp[0],
                                   this_fmt % int(sp[1]) + ".npz")
            npz_data = np.load(feat_fn)
            feat = npz_data["f"]
            iou_data = npz_data["i"]

            feats.append(feat)
            iou = [float(x) for x in iou_data[-3:]]
            ious.append(iou)
            box_xyxy = [float(x) for x in iou_data[:4]]
            gt_boxes.append(box_xyxy)
            jitter_box_xyxy = np.array([float(x) for x in iou_data[4:16]]).reshape(3, 4)
            jitter_boxes.append(jitter_box_xyxy)
        feats = np.stack(feats, axis=0)
        # atm just sample from same sequence, does not need to be hard
        data['hard_positive_features'] = feats
        data['hard_positive_ious'] = np.stack(ious, axis=0)
        data['hard_positive_gt_boxes'] = np.stack(gt_boxes, axis=0)
        data['hard_positive_jitter_boxes'] = np.stack(jitter_boxes, axis=0)
    return data


def _preprocess_youtube_bb(roidb, aug):
    ann_path = os.path.join(cfg.DATA.YOUTUBE_BB_ROOT, "annotations", roidb)
    ann_files = glob.glob(os.path.join(ann_path, "*.xml"))
    random.shuffle(ann_files)

    def ann_to_bbox(ann):
        if 'object' not in ann:
            return None
        if 'bndbox' not in ann['object']:
            return None
        bbox = ann['object']['bndbox']
        width = float(ann["size"]["width"])
        height = float(ann["size"]["height"])
        x1 = int(round(float(bbox['xmin']) * width))
        y1 = int(round(float(bbox['ymin']) * height))
        x2 = int(round(float(bbox['xmax']) * width))
        y2 = int(round(float(bbox['ymax']) * height))
        box = [x1, y1, x2, y2]
        return box

    # randomly select two files with bounding box
    sampled = []
    idx = 0
    while len(sampled) < 2 and idx < len(ann_files):
        ann = xmltodict.parse(open(ann_files[idx]).read())["annotation"]
        bbox = ann_to_bbox(ann)
        if bbox is not None:
            sampled.append((ann_files[idx], bbox))
        idx += 1
    if len(sampled) < 2:
        #print("did not find 2 bounding boxes in", roidb)
        return None

    ref_ann_file = sampled[0][0]
    target_ann_file = sampled[1][0]
    ref_box = sampled[0][1]
    target_box = sampled[1][1]
    ref_fname = ref_ann_file.replace("/annotations/", "/frames/").replace(".xml", ".jpg")
    ref_im = cv2.imread(ref_fname, cv2.IMREAD_COLOR)
    target_fname = target_ann_file.replace("/annotations/", "/frames/").replace(".xml", ".jpg")
    target_im = cv2.imread(target_fname, cv2.IMREAD_COLOR)
    return _preprocess_common(ref_box, target_box, ref_im, target_im, aug)


def _preprocess_trackingnet(roidb, aug):
    part, vid_name = roidb.split("____")
    data_path = os.path.join(cfg.DATA.TRACKINGNET_ROOT, part)
    gt_file = os.path.join(data_path, "anno", vid_name + ".txt")
    boxes = []
    with open(gt_file) as f:
        for l in f:
            sp = l.strip().split(",")
            box = [float(x) for x in sp]
            box[2] += box[0]
            box[3] += box[1]
            # there are negative coordinates in the data... should we compensate for that?
            if box[0] < 0:
                box[0] = 0.0
            if box[1] < 0:
                box[1] = 0.0
            boxes.append(box)
    n_frames = len(boxes)
    ref_idx = np.random.randint(n_frames)
    target_idx = np.random.randint(n_frames)
    ref_box = boxes[ref_idx]
    target_box = boxes[target_idx]
    ref_fname = os.path.join(data_path, "frames", vid_name, str(ref_idx) + ".jpg")
    ref_im = cv2.imread(ref_fname, cv2.IMREAD_COLOR)
    target_fname = os.path.join(data_path, "frames", vid_name, str(target_idx) + ".jpg")
    target_im = cv2.imread(target_fname, cv2.IMREAD_COLOR)
    return _preprocess_common(ref_box, target_box, ref_im, target_im, aug)


def get_train_dataflow():
    roidbs = DetectionDataset().load_training_roidbs(cfg.DATA.TRAIN)
    ds = DataFromList(roidbs, shuffle=True)
    # for now let's not do flipping to keep things simple
    aug = imgaug.AugmentorList(
        [CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)])#,
         #imgaug.Flip(horiz=True)])

    if cfg.MODE_HARD_MINING:
        from annoy import AnnoyIndex
        hard_mining_index = AnnoyIndex(128, 'euclidean')
        hard_mining_index.load(cfg.HARD_MINING_DATA_PATH + "/index_all/index.ann")
        names_path = cfg.HARD_MINING_DATA_PATH + "index_all/names.txt"
        hard_mining_names_all = []
        with open(names_path) as f:
            for l in f:
                hard_mining_names_all.append(l.strip())
            hard_example_names_got = [x[7:] for x in hard_mining_names_all if x.startswith("GOT10k/")]
            hard_example_names_vid = [x[12:] for x in hard_mining_names_all if x.startswith("ImageNetVID/")]
            hard_example_names_ytbvos = [x[11:] for x in hard_mining_names_all if x.startswith("YouTubeVOS/")]
            hard_example_names_lasot = [x[6:] for x in hard_mining_names_all if x.startswith("LaSOT/")]
            assert len(hard_example_names_got) > 0
            assert len(hard_example_names_vid) > 0
            assert len(hard_example_names_ytbvos) > 0
            assert len(hard_example_names_lasot) > 0
            hard_example_names_got.sort()
            hard_example_names_vid.sort()
            hard_example_names_ytbvos.sort()
            hard_example_names_lasot.sort()
            hard_mining_names = {"all": hard_mining_names_all, "GOT10k": hard_example_names_got,
                                 "ImageNetVID": hard_example_names_vid, "YouTubeVOS": hard_example_names_ytbvos,
                                 "LaSOT": hard_example_names_lasot}
    else:
        hard_mining_index = None
        hard_mining_names = None

    def preprocess(roidb):
        if roidb.startswith("VID/"):
            return _preprocess_imagenet_vid(roidb[4:], aug, hard_mining_index, hard_mining_names)
        elif roidb.startswith("DAVIS/"):
            return _preprocess_davis_like(roidb[6:], aug, os.path.join(cfg.DATA.DAVIS2017_ROOT, "Annotations",
                                                                       "480p"))
        elif roidb.startswith("YouTubeVOS/"):
            return _preprocess_davis_like(roidb[11:], aug, os.path.join(cfg.DATA.YOUTUBE_VOS_ROOT, "train",
                                                                        "Annotations"),
                                          "YouTubeVOS", hard_mining_index, hard_mining_names)
        elif roidb.startswith("GOT10K/"):
            return _preprocess_got10k(roidb[7:], aug, hard_mining_index, hard_mining_names)
        elif roidb.startswith("LaSOT/"):
            return _preprocess_lasot(roidb[6:], aug, hard_mining_index, hard_mining_names)
        elif roidb.startswith("YouTube-BB/"):
            return _preprocess_youtube_bb(roidb[11:], aug)
        elif roidb.startswith("TrackingNet/"):
            return _preprocess_trackingnet(roidb[12:], aug)
        else:
            assert False

    #ds = MultiProcessMapDataZMQ(ds, 10, preprocess)
    #ds = MapData(ds, preprocess)
    if cfg.DATA.DEBUG_VIS or not cfg.DATA.MULTITHREAD:
        ds = MapData(ds, preprocess)
    else:
        #ds = MultiThreadMapData(ds, 6, preprocess)
        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
    return ds


def get_eval_dataflow(name, shard=0, num_shards=1):
    seqs = []
    with open("davis2017_fast_val_ids.txt") as f:
        for l in f:
            seqs.append(l.strip())

    seqs_timesteps = []
    for seq in seqs:
        files = sorted(glob.glob(cfg.DATA.DAVIS2017_ROOT + "/JPEGImages/480p/" + seq.split("__")[0] + "/*.jpg"))[1:-1]
        timesteps = [f.split('/')[-1].replace(".jpg", "") for f in files]

        for timestep in timesteps:
          ann_fn = cfg.DATA.DAVIS2017_ROOT + "/Annotations/480p/" + seq.split("__")[0] + '/' + timestep + ".png"
          ann = np.array(PIL.Image.open(ann_fn))
          ann_mask = ann == int(seq.split("__")[1])
          if ann_mask.any():
            seqs_timesteps.append((seq.split('__')[0], seq.split('__')[1], timestep))

        # seqs_timesteps += [(seq.split('__')[0], seq.split('__')[1], timestep) for timestep in timesteps]

    num_seqs_timesteps = len(seqs_timesteps)
    seqs_timesteps_per_shard = num_seqs_timesteps // num_shards
    seqs_timesteps_range = (shard * seqs_timesteps_per_shard, (shard + 1) * seqs_timesteps_per_shard if shard + 1 < num_shards else num_seqs_timesteps)
    ds = DataFromList(seqs_timesteps[seqs_timesteps_range[0]: seqs_timesteps_range[1]])

    def preprocess(seq_timestep):

        seq, obj_id, timestep = seq_timestep
        ann_fn = cfg.DATA.DAVIS2017_ROOT + "/Annotations/480p/" + seq + '/' + timestep + ".png"
        ann = np.array(PIL.Image.open(ann_fn))
        ann_mask = ann == int(obj_id)
        if not ann_mask.any():
            return None, None, None, None, None
            # ann_box = np.array([-1000000, -1000000, 100000, 100000])
        else:
            ann_box = get_bbox_from_segmentation_mask_np(ann_mask)

        ff_fn = cfg.DATA.DAVIS2017_ROOT + "/Annotations/480p/" + seq + '/' + str(0).zfill(5) + ".png"
        ff = np.array(PIL.Image.open(ff_fn))
        ff_mask = ff == int(obj_id)
        ff_box = get_bbox_from_segmentation_mask_np(ff_mask)

        x1, y1, x2, y2 = [float(x) for x in ann_box]
        target_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        x1, y1, x2, y2 = [float(x) for x in ff_box]
        ref_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        target_img_fn = cfg.DATA.DAVIS2017_ROOT + "/JPEGImages/480p/" + seq + "/" + timestep + ".jpg"
        ref_img_fn = cfg.DATA.DAVIS2017_ROOT + "/JPEGImages/480p/" + seq + "/" + str(0).zfill(5) + ".jpg"
        target_img = cv2.imread(target_img_fn, cv2.IMREAD_COLOR)
        ref_img = cv2.imread(ref_img_fn, cv2.IMREAD_COLOR)
        return ref_img, ref_bbox, target_img, target_bbox, "__".join(seq_timestep)
    ds = MapData(ds, preprocess)
    return ds
