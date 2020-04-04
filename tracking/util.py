import numpy as np
import PIL.Image

from examples.FasterRCNN.common import clip_boxes


def xyxy_to_cxcywh_np(boxes_xyxy):
    wh = boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]
    c = boxes_xyxy[:, :2] + wh / 2
    boxes_cwh = np.concatenate((c, wh), axis=1)
    return boxes_cwh


def cxcywh_to_xyxy_np(boxes_cxcywh):
    boxes_xyxy = boxes_cxcywh.copy()
    boxes_xyxy[:, :2] -= 0.5 * boxes_xyxy[:, 2:]
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
    return boxes_xyxy


def resize_and_clip_boxes(img, resized_img, boxes):
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    orig_shape = img.shape[:2]
    boxes = boxes / scale
    boxes = clip_boxes(boxes, orig_shape)
    return boxes


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors():
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    import colorsys
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6, 10]
    colors = [colors[idx] for idx in perm]
    return colors


def postproc_seq_name_otb(seq_name):
    if seq_name == "Human4":
        seq_name_postproc = "Human4-2"
    elif seq_name == "Skating2_1":
        seq_name_postproc = "Skating2-1"
    elif seq_name == "Skating2_2":
        seq_name_postproc = "Skating2-2"
    elif seq_name == "Jogging_1":
        seq_name_postproc = "Jogging-1"
    elif seq_name == "Jogging_2":
        seq_name_postproc = "Jogging-2"
    else:
        seq_name_postproc = seq_name
    return seq_name_postproc


def read_gt_otb(gt_file):
    boxes = []
    with open(gt_file) as f:
        for l in f:
            l = l.strip()
            assert "," in l
            sp = l.split(",")
            x1, y1, w, h = [float(x) for x in sp]
            x2 = x1 + w
            y2 = y1 + h
            box = [x1, y1, x2, y2]
            boxes.append(box)
    boxes = np.array(boxes)
    return boxes


pascal_colormap = [
    0, 0, 0,
    0.5020, 0, 0,
    0, 0.5020, 0,
    0.5020, 0.5020, 0,
    0, 0, 0.5020,
    0.5020, 0, 0.5020,
    0, 0.5020, 0.5020,
    0.5020, 0.5020, 0.5020,
    0.2510, 0, 0,
    0.7529, 0, 0,
    0.2510, 0.5020, 0,
    0.7529, 0.5020, 0,
    0.2510, 0, 0.5020,
    0.7529, 0, 0.5020,
    0.2510, 0.5020, 0.5020,
    0.7529, 0.5020, 0.5020,
    0, 0.2510, 0,
    0.5020, 0.2510, 0,
    0, 0.7529, 0,
    0.5020, 0.7529, 0,
    0, 0.2510, 0.5020,
    0.5020, 0.2510, 0.5020,
    0, 0.7529, 0.5020,
    0.5020, 0.7529, 0.5020,
    0.2510, 0.2510, 0]


def save_segmentation_with_colormap(filename, img):
    """Saves a segmentation with the pascal colormap as expected for DAVIS eval.
    Args:
    filename: Where to store the segmentation.
    img: A numpy array of the segmentation to be saved.
    """
    if img.shape[-1] == 1:
        img = img[..., 0]

    # Save with colormap.
    colormap = (np.array(pascal_colormap) * 255).round().astype('uint8')
    colormap_image = PIL.Image.new('P', (16, 16))
    colormap_image.putpalette(colormap)
    pil_image = PIL.Image.fromarray(img.astype('uint8'))
    pil_image_with_colormap = pil_image.quantize(palette=colormap_image)
    pil_image_with_colormap.save(filename)

