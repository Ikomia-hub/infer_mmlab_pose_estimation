import numpy as np
from mmpose.datasets.dataset_info import DatasetInfo
import warnings
from ikomia import core
from ikomia.core import CPointF, CGraphicsPoint

det_model_zoo = {"hand": {"cfg": "mmdet_configs/hand/cascade_rcnn_x101_64x4d_fpn_1class.py",
                          "ckpt": "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth"},
                 "coco": {"cfg": "mmdet_configs/coco/ssdlite_mobilenetv2_scratch_600e_coco.py",
                          "ckpt": "https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth"}
                 }


def are_params_valid(params):
    for p in [params.body_part, params.task, params.method, params.dataset, params.model_name, params.config_name]:
        if p == "":
            return False
    return True


def process_mmdet_results(mmdet_results, class_names=None, cat_ids=1):
    """Process mmdet results to mmpose input format.
    Args:
        mmdet_results: raw output of mmdet model
        class_names: class names of mmdet model
        cat_ids (int or List[int]): category id list that will be preserved
    Returns:
        List[Dict]: detection results for mmpose input
    """
    if isinstance(mmdet_results, tuple):
        mmdet_results = mmdet_results[0]

    if not isinstance(cat_ids, (list, tuple)):
        cat_ids = [cat_ids]

    # only keep bboxes of interested classes
    bbox_results = [mmdet_results[i - 1] for i in cat_ids]
    bboxes = np.vstack(bbox_results)

    # get textual labels of classes
    labels = np.concatenate([
        np.full(bbox.shape[0], i - 1, dtype=np.int32)
        for i, bbox in zip(cat_ids, bbox_results)
    ])
    if class_names is None:
        labels = [f'class: {i}' for i in labels]
    else:
        labels = [class_names[i] for i in labels]

    det_results = []
    for bbox, label in zip(bboxes, labels):
        det_result = dict(bbox=bbox, label=label)
        det_results.append(det_result)
    return det_results


def vis_pose_result(model,
                    result,
                    graphics_output,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset_info=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
    """

    # get dataset info
    if (dataset_info is None and hasattr(model, 'cfg')
            and 'dataset_info' in model.cfg):
        dataset_info = DatasetInfo(model.cfg.dataset_info)

    if dataset_info is not None:
        skeleton = dataset_info.skeleton
        pose_kpt_color = dataset_info.pose_kpt_color
        pose_link_color = dataset_info.pose_link_color
    else:
        NotImplementedError()

    if hasattr(model, 'module'):
        model = model.module
    for item in result:
        keypoints = item["keypoints"]
        if 'bbox' in item:
            x, y, x2, y2 = item["bbox"][:4]
            if len(item["bbox"]) == 5:
                conf = item["bbox"][-1]
                graphics_output.addText(str(round(conf, 2)), float(x), float(y))
                rect_prop = core.GraphicsRectProperty()
                rect_prop.pen_colr = bbox_color
                graphics_output.addRectangle(float(x), float(y), float(x2 - x), float(y2 - y), rect_prop)
        valid = [False] * len(keypoints)

        for i, (x, y, score) in enumerate(keypoints):
            if score > kpt_score_thr:
                valid[i] = True
                pt_prop = core.GraphicsPointProperty()
                pt_prop.pen_color = [int(c) for c in pose_kpt_color[i]]
                pt_prop.size = 6
                pt = core.CGraphicsPoint(CPointF(float(x), float(y)), pt_prop)
                pt.setCategory(str(i))
                graphics_output.addItem(pt)

        for i, (idx1, idx2) in enumerate(skeleton):
            kp1 = keypoints[idx1]
            kp2 = keypoints[idx2]
            pt1 = CPointF(float(kp1[0]), float(kp1[1]))
            pt2 = CPointF(float(kp2[0]), float(kp2[1]))

            if valid[idx1] and valid[idx2]:
                properties_line = core.GraphicsPolylineProperty()
                properties_line.pen_color = [int(c) for c in pose_link_color[i]]
                graphics_output.addPolyline([pt1, pt2], properties_line)


