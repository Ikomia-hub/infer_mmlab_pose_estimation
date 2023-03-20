import numpy as np

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

