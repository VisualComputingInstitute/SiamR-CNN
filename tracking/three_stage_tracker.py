import platform
import time
import numpy as np
import scipy.sparse
from tensorpack import PredictConfig, get_model_loader, OfflinePredictor
from config import config as cfg

from tracking.argmax_tracker import PrecomputingReferenceTracker
from tracking.util import resize_and_clip_boxes, generate_colors, xyxy_to_cxcywh_np

VIZ_WITH_OPENCV = True


class Tracklet:
    def __init__(self, start_time):
        self.start_time = start_time
        self.end_time = start_time
        self.feats = []
        self.boxes = []
        self.ff_gt_scores = []
        self.ff_gt_tracklet_scores = []

    def add_detection(self, feat, box, ff_gt_score, ff_gt_tracklet_score):
        self.feats = [feat]
        self.boxes.append(box)
        self.ff_gt_scores.append(ff_gt_score)
        self.ff_gt_tracklet_scores.append(ff_gt_tracklet_score)
        self.end_time += 1


class ThreeStageTracker(PrecomputingReferenceTracker):
    def __init__(self, tracklet_distance_threshold=0.08, tracklet_merging_threshold=0.4,
                 tracklet_merging_second_best_relative_threshold=0.1, ff_gt_score_weight=0.4,
                 ff_gt_tracklet_score_weight=0.2, location_score_weight=0.0, do_viz=False,
                 name="ThreeStageTracker", model="best", n_proposals=None, resolution=None):
        """
        :param tracklet_merging_threshold: minimum score required to merge a detection into tracklet
        :param tracklet_merging_second_best_relative_threshold: minimum score gap to second best match allowed to merge the best detection into tracklet
        """
        if n_proposals is not None:
            cfg.RPN.TEST_PER_LEVEL_NMS_TOPK = n_proposals
        if resolution is not None:
            if resolution == "full":
                # nothing do do...
                pass
            elif resolution == "half":
                cfg.PREPROC.TEST_SHORT_EDGE_SIZE = 400
                cfg.PREPROC.MAX_SIZE = 667
            else:
                assert False, ("unknown resolution", resolution)
        super().__init__(name=name, need_network=True, need_img=True, model=model)
        self._n_proposals = n_proposals
        self._resolution = resolution
        self._ff_box = None
        self._ff_gt_tracklet = None
        self._all_tracklets = None
        self._time_idx = None
        self._imgs_for_viz = None
        self._ff_img_noresize = None
        self._ax = None
        self._cv_img = None
        self._do_viz = do_viz
        self._video_idx = -1
        self._video_name = None

        self._dynprog_scores = None
        self._tracklet_merging_threshold = tracklet_merging_threshold
        self._tracklet_merging_second_best_relative_threshold = tracklet_merging_second_best_relative_threshold
        self._tracklet_distance_threshold = tracklet_distance_threshold

        self._ff_gt_score_weight = ff_gt_score_weight
        self._ff_gt_tracklet_score_weight = ff_gt_tracklet_score_weight
        self._location_score_weight = location_score_weight

    def set_video_name(self, vid_name):
        self._video_name = vid_name

    def init(self, image, box):
        self._ff_box = None
        self._ff_gt_tracklet = None
        self._all_tracklets = None
        self._time_idx = 0
        self._ff_img_noresize = np.array(image)[..., ::-1]
        if self._do_viz:
            self._imgs_for_viz = [self._ff_img_noresize]
        self._video_idx += 1
        self._dynprog_scores = None

        super().init(image, box)
        self._ff_box = self._prev_box.copy()
        self._ff_gt_tracklet = Tracklet(start_time=0)
        self._ff_gt_tracklet.add_detection(self._ff_gt_feats, self._ff_box, 1.0, 1.0)
        self._all_tracklets = [self._ff_gt_tracklet]

    def _make_pred_func(self, load):
        cfg.MODE_THIRD_STAGE = True
        from train import ResNetFPNTrackModel
        pred_model = ResNetFPNTrackModel()
        predcfg = PredictConfig(
            model=pred_model,
            session_init=get_model_loader(load),
            input_names=pred_model.get_inference_tensor_names()[0],
            output_names=pred_model.get_inference_tensor_names()[1])
        return OfflinePredictor(predcfg)

    def _update(self, img):
        if self._do_viz:
            # we currently only need the most recent frame for viz
            self._imgs_for_viz = [img]
        self._time_idx += 1
        start = time.time()
        self._update_tracklets(img)
        best_box, score = self._track()
        end = time.time()
        # print("tracking step elapsed (with network)", end - start)
        if self._do_viz:
            self._viz_tracklets()
            self._viz_result(best_box)
            # save out viz
            #import cv2
            #cv2.imwrite("/tmp/viz/%05d.jpg" % self._time_idx, self._cv_img)
        return best_box, score

    def _update_tracklets(self, img):
        active_tracklets = [t for t in self._all_tracklets if t.end_time == self._time_idx]
        if len(active_tracklets) == 0:
            active_tracklets_boxes_noresize = np.zeros((0, 4), dtype=np.float32)
            active_tracklets_feats = np.zeros((0, 256, 7, 7))
        else:
            active_tracklets_boxes_noresize = np.stack([t.boxes[-1] for t in active_tracklets], axis=0)
            active_tracklets_feats = np.stack([t.feats[-1] for t in active_tracklets], axis=0)
        resized_img, active_tracklets_boxes = self._resize_image_together_with_boxes(img,
                                                                                     active_tracklets_boxes_noresize)
        boxes, scores, third_stage_feats_out, ff_gt_tracklet_scores, sparse_tracklet_scores, \
        tracklet_score_indices = self._pred_func(
                resized_img, self._ff_gt_feats, self._ff_gt_tracklet.feats[-1],  active_tracklets_feats,
                active_tracklets_boxes, self._tracklet_distance_threshold)
        boxes = resize_and_clip_boxes(img, resized_img, boxes)
        # for simplicity let's just convert it to a dense matrix. If that gets too large, we can still change it.
        tracklet_scores = scipy.sparse.coo_matrix((sparse_tracklet_scores, (tracklet_score_indices[:, 0],
                                                                            tracklet_score_indices[:, 1])),
                                                  shape=(len(active_tracklets), scores.size)
                                                  ).toarray()
        # free memory
        for t in self._all_tracklets:
            if t.end_time != self._time_idx and t.start_time != 0:
                t.feats = None
        self._update_tracklets_with_network_outputs(active_tracklets, boxes, scores, third_stage_feats_out,
                                                    ff_gt_tracklet_scores, tracklet_scores)

    def _update_tracklets_with_network_outputs(self, active_tracklets, boxes, scores, third_stage_feats_out,
                                               ff_gt_tracklet_scores, tracklet_scores):
        n_dets = scores.size
        for det_idx in range(n_dets):
            merged = False
            det_args = (third_stage_feats_out[det_idx], boxes[det_idx], scores[det_idx],
                        ff_gt_tracklet_scores[det_idx])

            # try to extend tracklets in active_tracklets
            if tracklet_scores.size > 0:
                if tracklet_scores[:, det_idx].max() > self._tracklet_merging_threshold:
                    tracklet_idx = tracklet_scores[:, det_idx].argmax()
                    max_score = tracklet_scores[tracklet_idx, det_idx]
                    # there should be no other det which has a high similarity
                    if (tracklet_scores[tracklet_idx] >= max_score - self._tracklet_merging_second_best_relative_threshold).sum() == 1:
                        # there should be no other tracklet to which this det is similar...
                        if (tracklet_scores[:, det_idx] >= max_score - self._tracklet_merging_second_best_relative_threshold).sum() == 1:
                            active_tracklets[tracklet_idx].add_detection(*det_args)
                            merged = True

            # otherwise start new tracklet
            if not merged:
                tracklet = Tracklet(start_time=self._time_idx)
                tracklet.add_detection(*det_args)
                self._all_tracklets.append(tracklet)

    def _track(self):
        # we know that the tracklets are always sorted by time!
        n_tracklets = len(self._all_tracklets)
        last_dynprog_scores = self._dynprog_scores
        self._dynprog_scores = np.full(n_tracklets, fill_value=-1e20, dtype=np.float32)
        # init gt tracklet score
        self._dynprog_scores[0] = 0.0
        if last_dynprog_scores is not None:
            self._dynprog_scores[:last_dynprog_scores.size] = last_dynprog_scores
        end_times = np.array([t.end_time for t in self._all_tracklets])
        im_h, im_w = self._ff_img_noresize.shape[:2]
        norm = np.array([im_w, im_h, im_w, im_h], np.float32)

        active_indices, = np.where(end_times >= self._time_idx + 1)
        active_tracklets = [self._all_tracklets[idx] for idx in active_indices]

        TRACKLET_KEEP_ALIVE_TIME = 1500
        if len(active_tracklets) > 0:
            if len(active_tracklets) == n_tracklets:
                alive_start_time = 0
            else:
                # select non-active tracklets: end_times < self._time_idx + 1
                alive_start_time = end_times[end_times < self._time_idx + 1].max()

            alive_indices, = np.where(end_times >= alive_start_time + 1 - TRACKLET_KEEP_ALIVE_TIME)
            alive_tracklets = [self._all_tracklets[idx] for idx in alive_indices]
            alive_end_boxes_cxcywh = xyxy_to_cxcywh_np(np.array([t.boxes[-1] for t in alive_tracklets]))
            alive_end_times = end_times[alive_indices]
            alive_dynprog_scores = self._dynprog_scores[alive_indices]
            active_start_boxes_cxcywh = xyxy_to_cxcywh_np(np.array([t.boxes[0] for t in active_tracklets]))
            all_pairwise_diffs = np.abs(active_start_boxes_cxcywh[:, np.newaxis] - alive_end_boxes_cxcywh[np.newaxis]) / norm
            all_pairwise_diffs = -all_pairwise_diffs.mean(axis=2)

        for idx, t_idx in enumerate(active_indices):
            tracklet = self._all_tracklets[t_idx]
            unary = self._ff_gt_score_weight * sum(tracklet.ff_gt_scores) + \
                self._ff_gt_tracklet_score_weight * sum(tracklet.ff_gt_tracklet_scores)

            valid_mask = tracklet.start_time >= alive_end_times
            if valid_mask.any():
                pairwise_scores = all_pairwise_diffs[idx]
                pred_scores = alive_dynprog_scores + self._location_score_weight * pairwise_scores
                pred_scores[np.logical_not(valid_mask)] = -1e20
                best_pred_idx = pred_scores.argmax()
                best_pred_score = pred_scores[best_pred_idx]
                if best_pred_score > -1e20:
                    self._dynprog_scores[t_idx] = best_pred_score + unary

        t_idx = self._dynprog_scores.argmax()
        tracklet = self._all_tracklets[t_idx]
        # add current frame score weighted with epsilon to change relative ranking within tracklet
        EPSILON = 0.00001
        if tracklet.end_time >= self._time_idx + 1:
            score = self._ff_gt_score_weight * max(tracklet.ff_gt_scores) + \
                self._ff_gt_tracklet_score_weight * max(tracklet.ff_gt_tracklet_scores) \
                + EPSILON * tracklet.ff_gt_scores[-1]
        else:
            score = -1.0 + EPSILON * tracklet.ff_gt_scores[-1]
        # or we could select the best tracklet in current frame
        return tracklet.boxes[-1], score

    if VIZ_WITH_OPENCV:
        def _viz_tracklets(self):
            print("viz tracklets frame", self._time_idx)
            import cv2
            self._cv_img = self._imgs_for_viz[-1].copy()
            colors = generate_colors()
            t = self._time_idx
            for idx, tracklet in enumerate(self._all_tracklets):
                # probably filter by confidence and tracklet length
                #if tracklet.end_time - tracklet.start_time < 2:
                #    continue
                if max(tracklet.ff_gt_scores) < 0.2:
                    continue
                if tracklet.start_time <= t < tracklet.end_time:
                    color = colors[idx % len(colors)]
                    box = tracklet.boxes[t - tracklet.start_time]
                    #cv2.rectangle(self._cv_img, (box[0], box[1]), (box[2], box[3]), [255 * x for x in color], 1)

        def _viz_result(self, box):
            import cv2
            #cv2.rectangle(self._cv_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 6)
            cv2.rectangle(self._cv_img, (box[0], box[1]), (box[2], box[3]), (0, 252, 124), 6)
            cv2.imshow('SUPERTRACK', self._cv_img)
            cv2.waitKey(1)
            #cv2.waitKey(0)
    else:
        def _viz_tracklets(self):
            print("viz tracklets frame", self._time_idx)
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            if self._ax is None:
                fig, self._ax = plt.subplots(1)
            colors = generate_colors()
            t = self._time_idx
            img = self._imgs_for_viz[-1]
            self._ax.clear()
            self._ax.imshow(img[..., ::-1])
            for idx, tracklet in enumerate(self._all_tracklets):
                # probably filter by confidence and tracklet length
                if tracklet.end_time - tracklet.start_time < 2:
                    continue
                if max(tracklet.ff_gt_scores) < 0.2:
                    continue
                if tracklet.start_time <= t < tracklet.end_time:
                    color = colors[idx % len(colors)]
                    box = tracklet.boxes[t - tracklet.start_time]
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    rect = Rectangle((box[0], box[1]), width, height, color=color, fill=False)
                    self._ax.add_patch(rect)
            # plt.pause(0.0001)

        def _viz_result(self, box):
            width = box[2] - box[0]
            height = box[3] - box[1]
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            rect = Rectangle((box[0], box[1]), width, height, color="red", fill=False, linewidth=4.0)
            self._ax.add_patch(rect)
            plt.pause(0.00001)
