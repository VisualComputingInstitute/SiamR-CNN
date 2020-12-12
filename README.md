# Siam R-CNN: Visual Tracking by Re-Detection
### [Paul Voigtlaender](https://www.vision.rwth-aachen.de/person/197/), [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/), [Philip H.S. Torr](https://www.robots.ox.ac.uk/~tvg/), [Bastian Leibe](https://www.vision.rwth-aachen.de/)
The corresponding project page can be found here: https://www.vision.rwth-aachen.de/page/siamrcnn

This software is written in Python3 and powered by TensorFlow 1.

We borrow a lot of code from TensorPack's Faster R-CNN example: https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN

## Installation

### Download necessary libraries
Here we will put all external libraries and this repository into /home/${USERNAME}/vision and use 
pip to install common libraries
```
mkdir /home/${USERNAME}/vision
cd /home/${USERNAME}/vision

git clone https://github.com/VisualComputingInstitute/SiamR-CNN.git
git clone https://github.com/pvoigtlaender/got10k-toolkit.git
git clone https://github.com/tensorpack/tensorpack.git

cd tensorpack
git checkout d24a9230d50b1dea1712a4c2765a11876f1e193c
cd ..

pip3 install cython
pip3 install tensorflow-gpu==1.15
pip3 install wget shapely msgpack msgpack_numpy tabulate xmltodict pycocotools opencv-python tqdm zmq annoy
```
### Add libraries to your PYTHONPATH
```
export PYTHONPATH=${PYTHONPATH}:/home/${USERNAME}/vision/got10k-toolkit/:/home/${USERNAME}/vision/tensorpack/
```

### Make Folder for models and logs and download pre-trained model
```
cd SiamR-CNN/
mkdir train_log
cd train_log
wget --no-check-certificate -r -nH --cut-dirs=2 --no-parent --reject="index.html*" https://omnomnom.vision.rwth-aachen.de/data/siamrcnn/hard_mining3/
cd ..
```
## Evaluation
For evaluation, first set the path to the dataset on which you want to evaluate in tracking/do_tracking.py, e.g.
```
OTB_2015_ROOT_DIR = '/data/otb2015/'
```

Then run tracking/do_tracking.py and specify the dataset you want to evaluate on using the main function for this dataset using e.g. --main main_otb
 
```
python3 tracking/do_tracking.py --main main_otb
```

The result will then be written to tracking_data/results/

## Training
Download the pre-trained Mask R-CNN model from http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz

Now change the paths to the training datasets in config.py, e.g.
```
_C.DATA.IMAGENET_VID_ROOT = "/globalwork/data/ILSVRC_VID/ILSVRC/"
```
there you can also enable and disable different datasets, e.g.
```
_C.DATA.IMAGENET_VID = True
```

To run the main training (without hard example mining):
```
python3 train.py --load /path/to/COCO-R101FPN-MaskRCNN-ScratchGN.npz
```

## Hints about the code
In the code, we sometimes use the terminology "ThreeStageTracker" or three stages. This refers to the Tracklet Dynamic Programming Algorithm (TDPA).

In order to make the code more readable, we removed some parts before publishing. If there's an important feature which you are missing, please write us an email at voigtlaender@vision.rwth-aachen.de

In the current version of the code, the functions to pre-compute the features for hard example mining are not available, but we can share the pre-computed data on request.

## References
If you find this code useful, please cite
```
Siam R-CNN: Visual Tracking by Re-Detection
Paul Voigtlaender, Jonathon Luiten, Philip H.S. Torr, Bastian Leibe.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
```
