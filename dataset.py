# -*- coding: utf-8 -*-
# File: coco.py

import numpy as np
import random
import os
import tqdm
import json
import glob

from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation

from config import config as cfg

__all__ = ['COCODetection', 'DetectionDataset']


class COCODetection(object):
    # handle the weird (but standard) split of train and val
    _INSTANCE_TO_BASEDIR = {
        'valminusminival2014': 'val2014',
        'minival2014': 'val2014',
    }

    COCO_id_to_category_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}  # noqa
    """
    Mapping from the incontinuous COCO category id to an id in [1, #category]
    For your own dataset, this should usually be an identity mapping.
    """

    # 80 names for COCO
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]  # noqa

    def __init__(self, basedir, name):
        self.name = name
        self._imgdir = os.path.realpath(os.path.join(
            basedir, self._INSTANCE_TO_BASEDIR.get(name, name)))
        assert os.path.isdir(self._imgdir), self._imgdir
        annotation_file = os.path.join(
            basedir, 'annotations/instances_{}.json'.format(name))
        assert os.path.isfile(annotation_file), annotation_file

        from pycocotools.coco import COCO
        self.coco = COCO(annotation_file)
        logger.info("Instances loaded from {}.".format(annotation_file))

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    def print_coco_metrics(self, json_file):
        """
        Args:
            json_file (str): path to the results json file in coco format
        Returns:
            dict: the evaluation metrics
        """
        from pycocotools.cocoeval import COCOeval
        ret = {}
        cocoDt = self.coco.loadRes(json_file)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
        for k in range(6):
            ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

        json_obj = json.load(open(json_file))
        if len(json_obj) > 0 and 'segmentation' in json_obj[0]:
            cocoEval = COCOeval(self.coco, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k in range(6):
                ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
        return ret

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'height', 'width', 'id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        if add_mask:
            assert add_gt
        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):
            img_ids = self.coco.getImgIds()
            img_ids.sort()
            # list of dict, each has keys: height,width,id,file_name
            imgs = self.coco.loadImgs(img_ids)

            for img in tqdm.tqdm(imgs):
                self._use_absolute_file_name(img)
                if add_gt:
                    self._add_detection_gt(img, add_mask)
            return imgs

    def _use_absolute_file_name(self, img):
        """
        Change relative filename to abosolute file name.
        """
        img['file_name'] = os.path.join(
            self._imgdir, img['file_name'])
        assert os.path.isfile(img['file_name']), img['file_name']

    def _add_detection_gt(self, img, add_mask):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        # objs = self.coco.loadAnns(ann_ids)
        objs = self.coco.imgToAnns[img['id']]  # equivalent but faster than the above two lines

        # clean-up boxes
        valid_objs = []
        width = img['width']
        height = img['height']
        for objid, obj in enumerate(objs):
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = obj['bbox']
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel

            x1 = np.clip(float(x1), 0, width)
            y1 = np.clip(float(y1), 0, height)
            w = np.clip(float(x1 + w), 0, width) - x1
            h = np.clip(float(y1 + h), 0, height) - y1
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 1 and w > 0 and h > 0 and w * h >= 4:
                obj['bbox'] = [x1, y1, x1 + w, y1 + h]
                valid_objs.append(obj)

                if add_mask:
                    segs = obj['segmentation']
                    if not isinstance(segs, list):
                        assert obj['iscrowd'] == 1
                        obj['segmentation'] = None
                    else:
                        valid_segs = [np.asarray(p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6]
                        if len(valid_segs) == 0:
                            logger.error("Object {} in image {} has no valid polygons!".format(objid, img['file_name']))
                        elif len(valid_segs) < len(segs):
                            logger.warn("Object {} in image {} has invalid polygons!".format(objid, img['file_name']))

                        obj['segmentation'] = valid_segs

        # all geometrically-valid boxes are returned
        boxes = np.asarray([obj['bbox'] for obj in valid_objs], dtype='float32')  # (n, 4)
        cls = np.asarray([
            self.COCO_id_to_category_id[obj['category_id']]
            for obj in valid_objs], dtype='int32')  # (n,)
        is_crowd = np.asarray([obj['iscrowd'] for obj in valid_objs], dtype='int8')

        # add the keys
        img['boxes'] = boxes        # nx4
        img['class'] = cls          # n, always >0
        img['is_crowd'] = is_crowd  # n,
        if add_mask:
            # also required to be float32
            img['segmentation'] = [
                obj['segmentation'] for obj in valid_objs]

    @staticmethod
    def load_many(basedir, names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.

        Returns the same format as :meth:`COCODetection.load`.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            coco = COCODetection(basedir, n)
            ret.extend(coco.load(add_gt, add_mask=add_mask))
        return ret


if cfg.DATA.IMAGENET_VID or cfg.DATA.DAVIS2017 or cfg.DATA.GOT10K or cfg.DATA.TRACKINGNET or cfg.DATA.COCO \
        or cfg.DATA.YOUTUBE_BB or cfg.DATA.DAVIS_LUCID or cfg.DATA.LASOT:

    def calculate_ious(bboxes1, bboxes2):
        # assume layout (x0, y0, x1, y1)
        min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        I = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        U = area1[:, np.newaxis] + area2[np.newaxis, :] - I
        assert (U > 0).all()
        IOUs = I / U
        assert (IOUs >= 0).all()
        assert (IOUs <= 1).all()
        return IOUs

    class DetectionDataset(object):
        occluders = None
        coco = None
        coco_anns = None

        def __init__(self):
            """
            This function is responsible for setting the dataset-specific
            attributes in both cfg and self.
            """
            # we do it category agnostic, so only foreground and background
            #self.num_category = cfg.DATA.NUM_CATEGORY = 1
            self.num_category = cfg.DATA.NUM_CATEGORY
            cfg.DATA.TRAIN = ["train"]
            cfg.DATA.VAL = ["val"]
            self.num_classes = self.num_category + 1
            self.class_names = cfg.DATA.CLASS_NAMES = ["BG", "FG"]

        def _load_roidb_imagenet_vid(self, subset):
            imageset_postfix = "ImageSets/VID/" + subset + ".txt"
            imagesets_file = os.path.join(cfg.DATA.IMAGENET_VID_ROOT, imageset_postfix)
            vid_names = set()
            with open(imagesets_file) as f:
                for l in f:
                    sp = l.split("/")
                    vid_name = sp[0] + "/" + sp[1]
                    vid_names.add(vid_name)
            vid_names = list(vid_names)
            return vid_names

        def _load_roidb_davis(self, subset):
            imagesets_file = os.path.join(cfg.DATA.DAVIS2017_ROOT, "ImageSets", "2017", subset + ".txt")
            vid_names = []
            with open(imagesets_file) as f:
                for l in f:
                    vid_name = l.strip()
                    vid_names.append(vid_name)
            return vid_names

        def _load_roidb_davis_lucid(self, subset):
            vid_names = sorted(glob.glob(cfg.DATA.DAVIS_LUCID_ROOT + "*/*/"))
            vid_names = ['/'.join(v.split("/")[-3:]) for v in vid_names]

            if cfg.TRACK_VIDEO_ID is not None:
                vid_names = sorted(glob.glob(cfg.DATA.DAVIS_LUCID_ROOT + "test-challenge/*/"))
                vid_names = ['/'.join(v.split("/")[-3:]) for v in vid_names]
                vid_names = [vid_names[cfg.TRACK_VIDEO_ID]]

                print("!!!!!!!!!!!!!!!ONLY DOING: ", vid_names[0], "!!!!!!!!!!!!!!!!!!!!!!!!!")

            # vid_names = ['test-challenge/speed-skating/']

            return vid_names

        def _load_roidb_youtubevos(self, subset):
            meta_file = os.path.join(cfg.DATA.YOUTUBE_VOS_ROOT, subset, "meta.json")
            with open(meta_file) as f:
                metadata = json.load(f)
            vid_names = list(metadata["videos"].keys())
            return vid_names

        def _load_roidb_got10k(self, subset):
            vid_names = []
            with open(os.path.join(cfg.DATA.GOT10K_ROOT, 'train/list.txt')) as f:
                for l in f:
                    vid_names.append(l.strip())
            assert len(vid_names) > 0
            return vid_names

        def _load_roidb_lasot(self, subset):
            vid_names = []
            with open(os.path.join(cfg.DATA.LASOT_ROOT, 'training_set.txt')) as f:
                for l in f:
                    vid_names.append(l.strip())
            assert len(vid_names) > 0
            return vid_names

        def _load_roidb_youtube_bb(self, subset):
            clips_fn = os.path.join(cfg.DATA.YOUTUBE_BB_ROOT, "sets", "clips.txt")
            roidbs = []
            with open(clips_fn) as f:
                for l in f:
                    roidbs.append(l.strip())
            return roidbs

        def _load_roidb_trackingnet(self, subset):
            gt_files = glob.glob(os.path.join(cfg.DATA.TRACKINGNET_ROOT, "TRAIN*", "anno", "*.txt"))
            vid_names = [x.split("/")[-3] + "____" + x.split("/")[-1].replace(".txt", "") for x in gt_files]
            return vid_names

        def _load_roidb(self, subset):
            vid_names = []
            if cfg.DATA.IMAGENET_VID:
                logger.info("using imagenet vid")
                vid_names_imgnet = self._load_roidb_imagenet_vid(subset)
                vid_names_imgnet = ["VID/" + x for x in vid_names_imgnet]
                vid_names += vid_names_imgnet
            if cfg.DATA.DAVIS2017:
                logger.info("using davis2017")
                vid_names_davis = self._load_roidb_davis(subset)
                vid_names_davis = ["DAVIS/" + x for x in vid_names_davis]
                vid_names += vid_names_davis
            if cfg.DATA.YOUTUBE_VOS:
                logger.info("using YouTube-VOS")
                vid_names_youtubevos = self._load_roidb_youtubevos(subset)
                vid_names_youtubevos = ["YouTubeVOS/" + x for x in vid_names_youtubevos]
                vid_names += vid_names_youtubevos
            if cfg.DATA.GOT10K:
                logger.info("using GOT10K")
                vid_names_got = self._load_roidb_got10k(subset)
                vid_names_got = ["GOT10K/" + x for x in vid_names_got]
                vid_names += vid_names_got
            if cfg.DATA.LASOT:
                logger.info("using LaSOT")
                vid_names_lasot = self._load_roidb_lasot(subset)
                vid_names_lasot = ["LaSOT/" + x for x in vid_names_lasot]
                vid_names += vid_names_lasot
            if cfg.DATA.YOUTUBE_BB:
                logger.info("using YouTube-BB")
                vid_names_youtube_bb = self._load_roidb_youtube_bb(subset)
                vid_names_youtube_bb = ["YouTube-BB/" + x for x in vid_names_youtube_bb]
                # duplicate all other vid names in order to sample them more often (YouTube-BB is very large, 300k clips)
                vid_names *= 60
                vid_names += vid_names_youtube_bb
            if cfg.DATA.TRACKINGNET:
                logger.info("using TrackingNet")
                vid_names_trackingnet = self._load_roidb_trackingnet(subset)
                vid_names_trackingnet = ["TrackingNet/" + x for x in vid_names_trackingnet]
                # duplicate all other vid names in order to sample them more often (trackingnet is very large)
                vid_names *= 2
                vid_names += vid_names_trackingnet
            random.shuffle(vid_names)
            return vid_names

        def load_training_roidbs(self, names):
            """
            Args:
                names (list[str]): name of the training datasets, e.g.  ['train2014', 'valminusminival2014']

            Returns:
                roidbs (list[dict]):

            Produce "roidbs" as a list of dict, each dict corresponds to one image with k>=0 instances.
            and the following keys are expected for training:

            height, width: integer
            file_name: str, full path to the image
            boxes: numpy array of kx4 floats, each row is [x1, y1, x2, y2]
            category: numpy array of k integers, in the range of [1, #categories]
            is_crowd: k booleans. Use k False if you don't know what it means.
            segmentation: k lists of numpy arrays (one for each instance).
                Each list of numpy arrays corresponds to the mask for one instance.
                Each numpy array in the list is a polygon of shape Nx2,
                because one mask can be represented by N polygons.

                If your segmentation annotations are originally masks rather than polygons,
                either convert it, or the augmentation will need to be changed or skipped accordingly.

                Include this field only if training Mask R-CNN.
            """
            return self._load_roidb("train")

        def load_inference_roidbs(self, name):
            """
            Args:
                name (str): name of one inference dataset, e.g. 'minival2014'

            Returns:
                roidbs (list[dict]):

                Each dict corresponds to one image to run inference on. The
                following keys in the dict are expected:

                file_name (str): full path to the image
                id (str): an id for the image. The inference results will be stored with this id.
            """
            return self._load_roidb("val")

        def eval_or_save_inference_results(self, results, dataset, output=None):
            ious_at_k = [[] for _ in range(10)]
            ious_per_obj = {}
            # results.sort(key=lambda x: x.gt_file)
            for r in results:
                gt_file, res, target_box = r
                seq, obj_id, timestep = gt_file.split('__')
                obj_name = seq + "__" + obj_id
                res.sort(key=lambda x: x.score, reverse=True)
                max_iou = 0.0
                if obj_name not in ious_per_obj.keys():
                    ious_per_obj[obj_name] = [[] for _ in range(10)]

                for k in range(10):
                    if len(res) > k:
                        det = res[k]
                        det_box = det.box
                        iou = calculate_ious(target_box[np.newaxis], det_box[np.newaxis])[0, 0]
                        max_iou = max(max_iou, iou)
                        if k == 0:
                            best_box = det_box
                    ious_per_obj[obj_name][k].append(max_iou)
                # print(seq,obj_id,timestep,target_box, best_box, ious_per_obj[obj_name][0][-1])

            for obj_name in ious_per_obj.keys():
                for k in range(10):
                    ious_at_k[k].append(np.mean(ious_per_obj[obj_name][k]))
                print(obj_name, np.mean(ious_per_obj[obj_name][0]))

            eval_res = {"miou@" + str(k + 1): np.mean(ious_at_k[k]) for k in range(10)}
            print(eval_res)
            return eval_res

        # code for singleton:
        _instance = None

        def __new__(cls):
            if not isinstance(cls._instance, cls):
                cls._instance = object.__new__(cls)
            return cls._instance
else:
    class DetectionDataset(object):
        """
        A singleton to load datasets, evaluate results, and provide metadata.

        To use your own dataset that's not in COCO format, rewrite all methods of this class.
        """
        def __init__(self):
            """
            This function is responsible for setting the dataset-specific
            attributes in both cfg and self.
            """
            self.num_category = cfg.DATA.NUM_CATEGORY = len(COCODetection.class_names)
            self.num_classes = self.num_category + 1
            self.class_names = cfg.DATA.CLASS_NAMES = ["BG"] + COCODetection.class_names

        def load_training_roidbs(self, names):
            """
            Args:
                names (list[str]): name of the training datasets, e.g.  ['train2014', 'valminusminival2014']

            Returns:
                roidbs (list[dict]):

            Produce "roidbs" as a list of dict, each dict corresponds to one image with k>=0 instances.
            and the following keys are expected for training:

            height, width: integer
            file_name: str, full path to the image
            boxes: numpy array of kx4 floats, each row is [x1, y1, x2, y2]
            category: numpy array of k integers, in the range of [1, #categories]
            is_crowd: k booleans. Use k False if you don't know what it means.
            segmentation: k lists of numpy arrays (one for each instance).
                Each list of numpy arrays corresponds to the mask for one instance.
                Each numpy array in the list is a polygon of shape Nx2,
                because one mask can be represented by N polygons.

                If your segmentation annotations are originally masks rather than polygons,
                either convert it, or the augmentation will need to be changed or skipped accordingly.

                Include this field only if training Mask R-CNN.
            """
            return COCODetection.load_many(
                cfg.DATA.BASEDIR, cfg.DATA.TRAIN, add_gt=True, add_mask=cfg.MODE_MASK)

        def load_inference_roidbs(self, name):
            """
            Args:
                name (str): name of one inference dataset, e.g. 'minival2014'

            Returns:
                roidbs (list[dict]):

                Each dict corresponds to one image to run inference on. The
                following keys in the dict are expected:

                file_name (str): full path to the image
                id (str): an id for the image. The inference results will be stored with this id.
            """
            return COCODetection.load_many(cfg.DATA.BASEDIR, name, add_gt=False)

        def eval_or_save_inference_results(self, results, dataset, output=None):
            """
            Args:
                results (list[dict]): the inference results as dicts.
                    Each dict corresponds to one __instance__. It contains the following keys:

                    image_id (str): the id that matches `load_inference_roidbs`.
                    category_id (int): the category prediction, in range [1, #category]
                    bbox (list[float]): x1, y1, x2, y2
                    score (float):
                    segmentation: the segmentation mask in COCO's rle format.

                dataset (str): the name of the dataset to evaluate.
                output (str): the output file to optionally save the results to.

            Returns:
                dict: the evaluation results.
            """
            continuous_id_to_COCO_id = {v: k for k, v in COCODetection.COCO_id_to_category_id.items()}
            for res in results:
                # convert to COCO's incontinuous category id
                res['category_id'] = continuous_id_to_COCO_id[res['category_id']]
                # COCO expects results in xywh format
                box = res['bbox']
                box[2] -= box[0]
                box[3] -= box[1]
                res['bbox'] = [round(float(x), 3) for x in box]

            assert output is not None, "COCO evaluation requires an output file!"
            with open(output, 'w') as f:
                json.dump(results, f)
            if len(output):
                # sometimes may crash if the results are empty?
                return COCODetection(cfg.DATA.BASEDIR, dataset).print_coco_metrics(output)
            else:
                return {}

        # code for singleton:
        _instance = None

        def __new__(cls):
            if not isinstance(cls._instance, cls):
                cls._instance = object.__new__(cls)
            return cls._instance


if __name__ == '__main__':
    c = COCODetection(cfg.DATA.BASEDIR, 'train2014')
    gt_boxes = c.load(add_gt=True, add_mask=True)
    print("#Images:", len(gt_boxes))
