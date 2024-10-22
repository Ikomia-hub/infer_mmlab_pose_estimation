import os
from pathlib import Path
import yaml
import numpy as np
from mmengine import Config


det_model_zoo = {
    "Hand": "cascade_rcnn_x101_64x4d_fpn_1class.py",
    "Person": "rtmdet/rtmdet_m_8xb32-300e_coco.py",
    "Face": "yolox-s_8xb8-300e_coco-face.py"
}


def dict_replace(obj, old, new):
    if isinstance(obj, list):
        for elt in obj:
            dict_replace(elt, old, new)
    if isinstance(obj, (dict, Config)):
        for k, elt in obj.items():
            if elt == old:
                obj[k] = new
            else:
                dict_replace(elt, old, new)


def get_detection_config(detector_name: str) -> tuple:
    detector_cfg_path = os.path.join(os.path.dirname(__file__), "mmdetection_cfg", det_model_zoo[detector_name])
    detector_ckpt = None

    # Legacy weights
    ckpt_det = Config.fromfile(detector_cfg_path).load_from
    if ckpt_det is None:
        yaml_file = os.path.join(Path(detector_cfg_path).parent, "metafile.yml")
        if os.path.isfile(yaml_file):
            with open(yaml_file, "r") as f:
                models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']
                for model_dict in models_list:
                    if model_dict["Name"] in det_model_zoo[detector_name]:
                        detector_ckpt = model_dict["Weights"]
                        break

                if detector_ckpt is None:
                    raise Exception(f"Can't find weights for configuration {det_model_zoo[detector_name]}")

    return detector_cfg_path, detector_ckpt


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
