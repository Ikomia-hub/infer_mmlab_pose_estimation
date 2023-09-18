import numpy as np
from mmengine import Config

det_model_zoo = {"Hand": "cascade_rcnn_x101_64x4d_fpn_1class.py",
                 "Person": "ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py",
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

def process_mmdet_results(mmdet_results, class_names=None, cat_ids=1):
    """Process mmdet results to mmpose input format.
    Args:
        mmdet_results: raw output of mmdet model
        class_names: class names of mmdet model
        cat_ids (int or List[int]): category id list that will be preserved
    Returns:
        List[Dict]: detection results for mmpose input
    """
    print(mmdet_results)
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

