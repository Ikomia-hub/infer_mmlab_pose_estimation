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


def logical_or(arrays: np.ndarray):
    if len(arrays) == 2:
        return np.logical_or(*arrays)
    elif len(arrays) == 1:
        return arrays[0]
    else:
        return np.logical_or(arrays[0], logical_or(arrays[1:]))


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


def get_root_path():
    current_path = Path(os.path.abspath(__file__))
    return current_path.parent.parent


def get_detection_config(detector_name: str) -> tuple:
    detector_cfg_path = os.path.join(get_root_path(), "mmdetection_cfg", det_model_zoo[detector_name])
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


def get_model_zoo() -> list:
    configs_folder = os.path.join(get_root_path(), "configs")
    available_configs = []

    for task in os.listdir(configs_folder):
        if task.startswith('_'):
            continue

        method_folder = os.path.join(configs_folder, task)
        for method in os.listdir(method_folder):
            if not os.path.isdir(os.path.join(method_folder, method)):
                continue

            dataset_folder = os.path.join(configs_folder, task, method)
            for dataset in os.listdir(dataset_folder):
                if not os.path.isdir(os.path.join(dataset_folder, dataset)):
                    continue

                for yaml_file in os.listdir(os.path.join(configs_folder, task, method, dataset)):
                    if not yaml_file.endswith('.yml'):
                        continue

                    yaml_file = os.path.join(configs_folder, task, method, dataset, yaml_file)
                    with open(yaml_file, "r") as f:
                        models_list = yaml.load(f, Loader=yaml.FullLoader)
                        if 'Models' in models_list:
                            models_list = models_list['Models']
                        if not isinstance(models_list, list):
                            continue

                    for model_dict in models_list:
                        available_configs.append({"config_file": model_dict["Config"]})

    return available_configs


def get_full_paths(param):
    config = param.config_file

    if os.path.isfile(config):
        if param.model_weight_file == "":
            print("model_weight_file is not set. Double check your parameters if it isn't intended.")
        return param.config_file, param.model_weight_file

    root_folder = get_root_path()
    files_tree = os.path.normpath(config).split(os.path.sep)
    yaml_folder = os.path.join(root_folder, *files_tree[:-1])

    if not os.path.isdir(yaml_folder):
        raise NotADirectoryError("Make sure the parameter config_file is correct or set both config_file and "
                                 "model_weight_file with absolute paths. "
                                 "See https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_pose_estimation/main/README.md "
                                 "for more information about how to use this algorithm.")

    for maybe_yaml in os.listdir(yaml_folder):
        if maybe_yaml.endswith('.yml'):
            yaml_file = os.path.join(yaml_folder, maybe_yaml)
            with open(yaml_file, "r") as f:
                models_list = yaml.load(f, Loader=yaml.FullLoader)

                if 'Models' in models_list:
                    models_list = models_list['Models']

                if not isinstance(models_list, list):
                    continue

            for model_dict in models_list:
                if os.path.normpath(config) == os.path.normpath(model_dict["Config"]):
                    return os.path.join(root_folder, model_dict['Config']), model_dict['Weights']

    raise NotImplementedError("This config_file has no pretrained weights.")


def postprocess_rtmpose3d(pose_results: list, rebase_keypts: bool = True):
    for idx, pose_est_result in enumerate(pose_results):
        pose_est_result.track_id = pose_results[idx].get('track_id', 1e4)

        pred_instances = pose_est_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores

        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_results[idx].pred_instances.keypoint_scores = keypoint_scores

        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = -keypoints[..., [0, 2, 1]]

        # rebase height (z-axis)
        if rebase_keypts:
            keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)

        pose_results[idx].pred_instances.keypoints = keypoints

    pose_results = sorted(pose_results, key=lambda x: x.get('track_id', 1e4))
    return pose_results
