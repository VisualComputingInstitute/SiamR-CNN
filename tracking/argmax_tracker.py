import cv2
import random
import numpy as np
from got10k.trackers import Tracker
from config import config as cfg, finalize_configs
from tensorpack import PredictConfig, get_model_loader, OfflinePredictor, logger

from train import ResNetFPNModel
from common import CustomResize, box_to_point8, point8_to_box


class PrecomputingReferenceTracker(Tracker):
    def __init__(self, name, need_network=True, need_img=True, model="best"):
        super().__init__(name=name, is_deterministic=True)
        self._resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
        self._prev_box = None
        self._ff_gt_feats = None
        self._need_network = need_network
        self._need_img = need_img
        self._rotated_bbox = None

        if need_network:
            logger.set_logger_dir("/tmp/test_log_/" + str(random.randint(0, 10000)), 'd')
            if model == "best":
                load = "train_log/hard_mining3/model-1360500"
            elif model == "nohardexamples":
                load = "train_log/condrcnn_all_2gpu_lrreduce2/model-1200500"
            elif model == "newrpn":
                load = "train_log/newrpn1/model"
            elif model =="resnet50_nohardexamples":
                load = "train_log/condrcnn_all_resnet50/model-1200500"
                cfg.BACKBONE.RESNET_NUM_BLOCKS = [3, 4, 6, 3]
            elif model =="resnet50":
                load = "train_log/hard_mining3_resnet50/model-1360500"
                cfg.BACKBONE.RESNET_NUM_BLOCKS = [3, 4, 6, 3]
            elif model == "gotonly":
                load = "train_log/hard_mining3_onlygot/model-1361000"
            elif model.startswith("checkpoint:"):
                load = model.replace("checkpoint:", "")
            else:
                assert False, ("unknown model", model)
            from dataset import DetectionDataset
            # init tensorpack model
            # cfg.freeze(False)
            DetectionDataset()  # initialize the config with information from our dataset

            cfg.EXTRACT_GT_FEATURES = True
            cfg.MODE_TRACK = False
            extract_model = ResNetFPNModel()
            extract_ff_feats_cfg = PredictConfig(
                model=extract_model,
                session_init=get_model_loader(load),
                input_names=['image', 'roi_boxes'],
                output_names=['rpn/feature'])
            finalize_configs(is_training=False)
            self._extract_func = OfflinePredictor(extract_ff_feats_cfg)

            cfg.EXTRACT_GT_FEATURES = False
            cfg.MODE_TRACK = True
            cfg.USE_PRECOMPUTED_REF_FEATURES = True
            self._pred_func = self._make_pred_func(load)

    def _resize_image_together_with_boxes(self, img, *list_of_box_or_boxes):
        resized_img, params = self._resizer.augment_return_params(img)
        res_boxes = []
        for box_or_boxes in list_of_box_or_boxes:
            expand = len(box_or_boxes.shape) == 1
            if expand:
                boxes = box_or_boxes[np.newaxis]
            else:
                boxes = box_or_boxes
            points = box_to_point8(boxes)
            points = self._resizer.augment_coords(points, params)
            resized_boxes = point8_to_box(points)
            if expand:
                resized_boxes = np.squeeze(resized_boxes, axis=0)
            res_boxes.append(resized_boxes)
        if len(res_boxes) == 1:
            res_boxes = res_boxes[0]
        return resized_img, res_boxes

    def _make_pred_func(self, load):
        from train import ResNetFPNTrackModel
        pred_model = ResNetFPNTrackModel()
        predcfg = PredictConfig(
            model=pred_model,
            session_init=get_model_loader(load),
            input_names=pred_model.get_inference_tensor_names()[0],
            output_names=pred_model.get_inference_tensor_names()[1])
        return OfflinePredictor(predcfg)

    def init(self, image, box):
        ref_img = np.array(image)[..., ::-1]
        if ref_img is None:
            raise ValueError("failed to load img" + image.filename)
        box[2] += box[0]
        box[3] += box[1]
        ref_bbox = box
        self._prev_box = box
        if self._need_network:
            resized_ref_img, resized_ref_box = self._resize_image_together_with_boxes(ref_img, ref_bbox)
            feats, = self._extract_func(resized_ref_img, resized_ref_box[np.newaxis])
            self._ff_gt_feats = feats[0]

    def update(self, image, use_confidences=False):
        if self._need_img:
            target_img = np.array(image)[..., ::-1]
            if target_img is None:
                raise ValueError("failed to load img" + str(target_img))
        else:
            target_img = None

        new_box, score = self._update(target_img)
        if new_box is not None:
            self._prev_box = new_box

        ret_box = self._prev_box.copy()
        ret_box[2] -= ret_box[0]
        ret_box[3] -= ret_box[1]
        if self._rotated_bbox is not None:
            ret_box = self._rotated_bbox
        if use_confidences:
            return ret_box, score
        else:
            return ret_box


class ArgmaxTracker(PrecomputingReferenceTracker):
    def __init__(self):
        super().__init__("ArgmaxTracker")

    def _update(self, img):
        from eval import predict_image_track_with_precomputed_ref_features
        results = predict_image_track_with_precomputed_ref_features(img, self._ff_gt_feats, self._pred_func)
        det_boxes = np.array([r.box for r in results])
        det_scores = np.array([r.score for r in results])
        if len(det_boxes) > 0:
            return det_boxes[0], det_scores[0]
        else:
            return None, None


# just there to test the precomputing on against
# not intended to be used anymore
class NonPrecomputingArgmaxTracker(Tracker):
    def __init__(self):
        super().__init__(name='ArgmaxTracker', is_deterministic=True)
        self._ref_img = None
        self._ref_bbox = None
        self._prev_box = None
        model = self._init_model()
        load = "train_log/condrcnn_onlygot/model-460000"
        predcfg = PredictConfig(
            model=model,
            session_init=get_model_loader(load),
            input_names=model.get_inference_tensor_names()[0],
            output_names=model.get_inference_tensor_names()[1])
        self._pred_func = OfflinePredictor(predcfg)

    def _init_model(self):
        logger.set_logger_dir("/tmp/test_log/", 'd')
        from dataset import DetectionDataset
        from train import ResNetFPNTrackModel
        # init tensorpack model
        cfg.freeze(False)
        model = ResNetFPNTrackModel()
        DetectionDataset()  # initialize the config with information from our dataset
        finalize_configs(is_training=False)
        return model

    def init(self, image, box):
        self._ref_img = cv2.imread(image.filename, cv2.IMREAD_COLOR)
        if self._ref_img is None:
            raise ValueError("failed to load img" + str(self._ref_img))
        box[2] += box[0]
        box[3] += box[1]
        self._ref_bbox = box
        self._prev_box = box

    def update(self, image):
        target_img = cv2.imread(image.filename, cv2.IMREAD_COLOR)
        # assert target_img is not None
        if target_img is None:
            raise ValueError("failed to load img" + str(target_img))
        from eval import predict_image_track
        results = predict_image_track(target_img, self._ref_img, self._ref_bbox, self._pred_func)
        det_boxes = np.array([r.box for r in results])
        det_scores = np.array([r.score for r in results])
        if len(det_boxes) > 0:
            self._prev_box = det_boxes[0]

        ret_box = self._prev_box.copy()
        ret_box[2] -= ret_box[0]
        ret_box[3] -= ret_box[1]
        return ret_box
