import os
import sys
import torch
import datetime
from tempfile import NamedTemporaryFile
import numpy as np

from mmengine import Config
from mmengine import DefaultScope
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_model, inference_topdown, inference_bottomup
from mmpose.apis.inference import dataset_meta_from_config
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples

from ikomia.dataprocess import CKeypointLink

from infer_mmlab_pose_estimation.core.utils import (
    logical_or, dict_replace, get_detection_config, get_root_path, get_full_paths, postprocess_rtmpose3d
)


class PoseInference:
    def __init__(self, params):
        self.params = params
        self.device = "cuda" if params.cuda and torch.cuda.is_available() else 'cpu'
        self.det_config = None
        self.det_checkpoint = None
        self.det_model = None
        self.pose_config_path = None
        self.pose_model = None
        self.cat_ids = None
        self.keypoint_links = None
        self.dataset_info = None
        self._setup_rtmpose3d_project()

    @staticmethod
    def _setup_rtmpose3d_project():
        root_path = str(get_root_path())
        rtmpose3d_path = os.path.join(root_path, "rtmpose3d")

        if os.path.isdir(rtmpose3d_path) and root_path not in sys.path:
            sys.path.insert(0, root_path)

    def is_model_loaded(self) -> bool:
        return self.pose_model is not None

    def load_models(self):
        old_torch_hub = torch.hub.get_dir()
        torch.hub.set_dir(os.path.join(get_root_path(), "models"))

        if self.params.detector != "None":
            self.det_config, self.det_checkpoint = get_detection_config(self.params.detector)
            self.det_model = init_detector(self.det_config, self.det_checkpoint, device=self.device.lower())

        if self.params.detector == "Person":
            self.cat_ids = [0]
        elif self.params.detector in ["Hand", "Face"]:
            self.cat_ids = [0]
        else:
            self.cat_ids = []

        self.pose_config_path, ckpt_pose = get_full_paths(self.params)
        cfg_pose = Config.fromfile(self.pose_config_path)
        dict_replace(cfg_pose, "SyncBN", "BN")

        tmp_cfg = NamedTemporaryFile(suffix='.py', delete=False)
        cfg_pose.dump(tmp_cfg.name)
        tmp_cfg.close()
        cfg_pose = tmp_cfg.name

        # build pose models
        self.pose_model = init_model(cfg_pose, ckpt_pose, device=self.device.lower())
        torch.hub.set_dir(old_torch_hub)
        self.dataset_info = dataset_meta_from_config(Config.fromfile(cfg_pose), dataset_mode='val')

        if self.dataset_info is not None:
            skeleton_link_colors = self.dataset_info["skeleton_link_colors"]

            # Compute keypoint links
            self.keypoint_links = []
            for i, (id1, id2) in enumerate(self.dataset_info["skeleton_links"]):
                link = CKeypointLink()
                link.start_point_index = id1
                link.end_point_index = id2

                name1 = self.dataset_info["keypoint_id2name"][id1]
                name2 = self.dataset_info["keypoint_id2name"][id2]
                link.label = f"{name1} - {name2}"
                link.color = [int(c) for c in skeleton_link_colors[i]]
                self.keypoint_links.append(link)
        else:
            raise NotImplementedError()

        # Remove temp file
        os.remove(tmp_cfg.name)

    def _detect_objets(self, src_image, detect_input=None) -> tuple:
        if self.params.detector == "None":
            if detect_input is None:
                raise ValueError("Object detection input can't be empty if you don't set a detector.")

            bboxes = []
            det_scores = []

            for idx, obj in enumerate(detect_input.get_objects()):
                conf = obj.confidence
                x, y, w, h = obj.box
                box = [x, y, x + w, y + h]
                bboxes.append(box)
                det_scores.append(conf)
        else:
            # To avoid registry error when running detection model
            # not using register_mmdet_modules because too many warnings
            DefaultScope.get_instance(f'mmdet-{datetime.datetime.now()}', scope_name='mmdet')
            instances = inference_detector(self.det_model, src_image).pred_instances.detach().cpu().numpy()
            bboxes = np.concatenate(
                (instances.bboxes, instances.labels[:, None], instances.scores[:, None]),
                axis=1
            )
            bboxes = bboxes[np.logical_and(
                logical_or([instances.labels == i for i in self.cat_ids]),
                instances.scores > self.params.conf_thres)
            ]
            bboxes = bboxes[nms(bboxes, self.params.conf_thres)]
            bboxes, det_scores = bboxes[:, :4], bboxes[:, 5]

        return bboxes, det_scores

    def run(self, src_image, detect_input=None):
        cfg_pose = Config.fromfile(self.pose_config_path)
        if cfg_pose.data_mode == "topdown":
            # get detection objects
            bboxes, det_scores = self._detect_objets(src_image, detect_input)
            # inference pose model
            pose_results = inference_topdown(self.pose_model, src_image, bboxes, bbox_format='xyxy')
        elif cfg_pose.data_mode == "bottomup":
            scope = self.pose_model.cfg.get('default_scope', 'mmpose')
            if scope is not None:
                init_default_scope(scope)

            # inference pose model
            pose_results = inference_bottomup(self.pose_model, src_image)
            det_scores = None
        else:
            raise RuntimeError("Unsupported data mode for pose estimation inference.")

        pose_final_results = self.post_process(pose_results)
        return pose_final_results, det_scores

    def post_process(self, pose_results):
        if self.params.method == "rtmpose3d":
            processed_results = postprocess_rtmpose3d(pose_results)
        else:
            processed_results = pose_results

        return merge_data_samples(processed_results)

