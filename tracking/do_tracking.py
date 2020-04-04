#!/usr/bin/env python3
import argparse
import os

from got10k.experiments import ExperimentGOT10k, ExperimentVOT, ExperimentOTB, ExperimentUAV123, ExperimentLaSOT, ExperimentDAVIS, ExperimentYouTubeVOS, ExperimentTrackingNet, ExperimentOxuva, ExperimentNfS, ExperimentTColor128
from got10k.experiments.custom import ExperimentCustom

from tracking.argmax_tracker import ArgmaxTracker
from tracking.three_stage_tracker import ThreeStageTracker

# change these data paths to where you have the datasets!
DATASET_PREFIX = "/globalwork/data/"
VOT18_ROOT_DIR = os.path.join(DATASET_PREFIX, 'vot18')
VOT17_ROOT_DIR = os.path.join(DATASET_PREFIX, 'vot17')
VOT16_ROOT_DIR = os.path.join(DATASET_PREFIX, 'vot16')
VOT15_ROOT_DIR = os.path.join(DATASET_PREFIX, 'vot15')
VOT18_LT_ROOT_DIR = os.path.join(DATASET_PREFIX, 'vot18-lt')
OTB_2015_ROOT_DIR = os.path.join(DATASET_PREFIX, 'OTB_new')
OTB_2013_ROOT_DIR = os.path.join(DATASET_PREFIX, 'OTB2013')
DAVIS_2017_ROOT_DIR = os.path.join(DATASET_PREFIX, 'DAVIS2017')
YOUTUBE_VOS_2019_ROOT_DIR = os.path.join(DATASET_PREFIX, "youtube-vos-2019")
GOT10K_ROOT_DIR = os.path.join(DATASET_PREFIX, 'GOT10k')
UAV123_ROOT_DIR = os.path.join(DATASET_PREFIX, 'UAV123')
LASOT_ROOT_DIR = os.path.join(DATASET_PREFIX, 'LaSOTBenchmark')
TRACKINGNET_ROOT_DIR = os.path.join(DATASET_PREFIX, 'TrackingNet')
NFS_ROOT_DIR = os.path.join(DATASET_PREFIX, 'nfs')
TC128_ROOT_DIR = os.path.join(DATASET_PREFIX, 'tc128/Temple-color-128')
OXUVA_ROOT_DIR = os.path.join(DATASET_PREFIX, 'oxuva')

RESULT_DIR = 'tracking_data/results/'
REPORT_DIR = 'tracking_data/reports/'

parser = argparse.ArgumentParser()
parser.add_argument('--start_idx', type=int, help='first video index to process', default=0)
parser.add_argument('--end_idx', type=int, help='last video index to process (exclusive)', default=None)

# TDPA parameters. You can just leave them at the default values which will work well on a wide range of datasets
parser.add_argument('--tracklet_distance_threshold', type=float, default=0.06)
parser.add_argument('--tracklet_merging_threshold', type=float, default=0.3)
parser.add_argument('--tracklet_merging_second_best_relative_threshold', type=float, default=0.3)
parser.add_argument('--ff_gt_score_weight', type=float, default=0.1)
parser.add_argument('--ff_gt_tracklet_score_weight', type=float, default=0.9)
parser.add_argument('--location_score_weight', type=float, default=7.0)

parser.add_argument('--model', type=str, default="best", help='one of "best", "nohardexamples", or "gotonly"')
parser.add_argument('--tracker', type=str, default='ThreeStageTracker')
parser.add_argument('--n_proposals', type=int, default=None)
parser.add_argument('--resolution', type=str, default=None)
parser.add_argument('--visualize_tracker', action='store_true',
                    help='use visualization of tracker (recommended over --visualize_experiment)')
parser.add_argument('--visualize_experiment', action='store_true',
                    help='use visualization of got experiment (not recommended, usually --visualize_tracker is better)')
parser.add_argument('--custom_dataset_name', type=str, default=None)
parser.add_argument('--custom_dataset_root_dir', type=str, default=None)
parser.add_argument('--main', type=str)
args = parser.parse_args()


def build_tracker():
    if args.tracker == "ArgmaxTracker":
        return ArgmaxTracker()
    elif args.tracker == "ThreeStageTracker":
        pass
    else:
        assert False, ("Unknown tracker", args.tracker)

    tracklet_param_str = str(args.tracklet_distance_threshold) + "_" + str(args.tracklet_merging_threshold) + "_" + \
        str(args.tracklet_merging_second_best_relative_threshold)
    if args.n_proposals is not None:
        tracklet_param_str += "_proposals" + str(args.n_proposals)
    if args.resolution is not None:
        tracklet_param_str += "_resolution-" + str(args.resolution)
    if args.model != "best":
        tracklet_param_str = args.model + "_" + tracklet_param_str
    if args.visualize_tracker:
        tracklet_param_str2 = "viz_" + tracklet_param_str
    else:
        tracklet_param_str2 = tracklet_param_str
    param_str = tracklet_param_str2 + "_" + str(args.ff_gt_score_weight) + "_" + \
        str(args.ff_gt_tracklet_score_weight) + "_" + str(args.location_score_weight)

    name = "ThreeStageTracker_" + param_str
    tracker = ThreeStageTracker(tracklet_distance_threshold=args.tracklet_distance_threshold,
                                tracklet_merging_threshold=args.tracklet_merging_threshold,
                                tracklet_merging_second_best_relative_threshold=
                                args.tracklet_merging_second_best_relative_threshold,
                                ff_gt_score_weight=args.ff_gt_score_weight,
                                ff_gt_tracklet_score_weight=args.ff_gt_tracklet_score_weight,
                                location_score_weight=args.location_score_weight,
                                name=name,
                                do_viz=args.visualize_tracker,
                                model=args.model,
                                n_proposals=args.n_proposals,
                                resolution=args.resolution)
    return tracker


def main_vot18(reset=True):
    root_dir = VOT18_ROOT_DIR
    if reset:
        experiments = "supervised"
    else:
        experiments = "unsupervised"
    tracker = build_tracker()
    experiment = ExperimentVOT(
        root_dir=root_dir,
        version=2018,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        experiments=experiments,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_vot18_noreset():
    main_vot18(reset=False)


def main_vot18_threestage():
    tracker = build_tracker()
    root_dir = VOT18_ROOT_DIR
    experiment = ExperimentVOT(
        root_dir=root_dir,
        version=2018,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        experiments="supervised",
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_vot17():
    root_dir = VOT17_ROOT_DIR
    experiments = "supervised"
    tracker = build_tracker()
    experiment = ExperimentVOT(
        root_dir=root_dir,
        version=2017,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        experiments=experiments,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)


def main_vot16():
    root_dir = VOT16_ROOT_DIR
    experiments = "supervised"
    tracker = build_tracker()
    experiment = ExperimentVOT(
        root_dir=root_dir,
        version=2016,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        experiments=experiments,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)


def main_vot15():
    root_dir = VOT15_ROOT_DIR
    experiments = "supervised"
    tracker = build_tracker()
    experiment = ExperimentVOT(
        root_dir=root_dir,
        version=2015,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        experiments=experiments,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)


def main_vot18lt():
    tracker = build_tracker()
    experiment = ExperimentVOT(
        root_dir=VOT18_LT_ROOT_DIR,
        version='LT2018',
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        experiments="unsupervised",
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    # this needs to be eval'ed from matlab, so do not call report()


def main_otb():
    tracker = build_tracker()
    root_dir = OTB_2015_ROOT_DIR
    experiment = ExperimentOTB(
        root_dir=root_dir,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_otb2013():
    tracker = build_tracker()
    root_dir = OTB_2013_ROOT_DIR
    experiment = ExperimentOTB(
        version=2013,
        root_dir=root_dir,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_otb50():
    tracker = build_tracker()
    root_dir = OTB_2015_ROOT_DIR
    experiment = ExperimentOTB(
        version='tb50',
        root_dir=root_dir,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_davis(version="2017_val"):
    tracker = build_tracker()
    root_dir = DAVIS_2017_ROOT_DIR
    experiment = ExperimentDAVIS(
        root_dir=root_dir,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        version=version
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_davis2016():
    main_davis(version="2016_val")


def main_davis2017():
    main_davis(version="2017_val")


def main_davis2017_testdev():
    main_davis(version="2017_testdev")


def main_davis2017_train():
    main_davis(version="2017_train")


def main_davis2017_train_multiobj():
    main_davis(version="2017_train_multiobj")


def main_youtubevos(version="valid"):
    tracker = build_tracker()
    root_dir = YOUTUBE_VOS_2019_ROOT_DIR
    experiment = ExperimentYouTubeVOS(
        root_dir=root_dir,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        version=version
    )
    experiment.run(tracker, visualize=args.visualize_experiment)


def main_got(subset='val'):
    dataset_name = "GOT10k"
    if subset != 'val':
        dataset_name += "_" + subset
    tracker = build_tracker()
    experiment = ExperimentGOT10k(
        root_dir=GOT10K_ROOT_DIR,  # GOT-10k's root directory
        subset=subset,  # 'train' | 'val' | 'test'
        result_dir=RESULT_DIR,  # where to store tracking results
        report_dir=REPORT_DIR,  # where to store evaluation reports
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_got_test():
    main_got(subset='test')


def main_uav123():
    tracker = build_tracker()
    experiment = ExperimentUAV123(
        root_dir=UAV123_ROOT_DIR,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_uav20l():
    tracker = build_tracker()
    experiment = ExperimentUAV123(
        root_dir=UAV123_ROOT_DIR,
        version='UAV20L',
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_lasot():
    tracker = build_tracker()
    experiment = ExperimentLaSOT(
        root_dir=LASOT_ROOT_DIR,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        subset='test',
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_trackingnet():
    tracker = build_tracker()
    experiment = ExperimentTrackingNet(
        root_dir=TRACKINGNET_ROOT_DIR,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        subset='test',
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)


def main_nfs():
    tracker = build_tracker()
    experiment = ExperimentNfS(
        root_dir=NFS_ROOT_DIR,
        fps=30,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_tc128():
    tracker = build_tracker()
    experiment = ExperimentTColor128(
        root_dir=TC128_ROOT_DIR,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


def main_oxuva(testset=True):
    tracker = build_tracker()
    experiment = ExperimentOxuva(
        root_dir=OXUVA_ROOT_DIR,
        result_dir=RESULT_DIR,
        report_dir=REPORT_DIR,
        subset='test' if testset else 'dev',
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)


def main_oxuva_dev():
    main_oxuva(testset=False)


def main_custom():
    custom_dataset_root_dir = args.custom_dataset_root_dir
    assert custom_dataset_root_dir is not None
    custom_dataset_name = args.custom_dataset_name
    assert custom_dataset_name is not None
    tracker = build_tracker()
    experiment = ExperimentCustom(
        root_dir=custom_dataset_root_dir,
        name=custom_dataset_name
    )
    experiment.run(tracker, visualize=args.visualize_experiment)


if __name__ == "__main__":
    assert args.main is not None, "--main not supplied, e.g. --main main_otb"
    eval(args.main + "()")
