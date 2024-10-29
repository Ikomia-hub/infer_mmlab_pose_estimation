# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os.path
import copy
import datetime
import numpy as np
import torch
from tempfile import NamedTemporaryFile

from ikomia import utils, core, dataprocess

from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown, inference_bottomup
from mmpose.apis.inference import dataset_meta_from_config
from mmdet.apis import inference_detector, init_detector
from mmpose.evaluation.functional import nms
from mmengine import Config
from mmengine import DefaultScope
from mmengine.registry import init_default_scope

from infer_mmlab_pose_estimation.utils import logical_or, dict_replace, get_detection_config, get_full_paths


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabPoseEstimationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = True
        self.update = False
        # Parameters only used in widget
        self.conf_thres = 0.5
        self.conf_kp_thres = 0.3
        self.config_file = "configs/body_2d_keypoint/rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py"
        self.model_weight_file = ""
        self.detector = "Person"

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = utils.strtobool(param_map["cuda"])
        self.conf_thres = float(param_map["conf_thres"])
        self.conf_kp_thres = float(param_map["conf_kp_thres"])
        self.model_weight_file = param_map["model_weight_file"]
        self.config_file = param_map["config_file"]
        self.detector = param_map["detector"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "cuda": str(self.cuda),
            "conf_thres": str(self.conf_thres),
            "conf_kp_thres": str(self.conf_kp_thres),
            "model_weight_file": self.model_weight_file,
            "config_file": self.config_file,
            "detector": self.detector
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabPoseEstimation(dataprocess.CKeypointDetectionTask):

    def __init__(self, name, param):
        dataprocess.CKeypointDetectionTask.__init__(self, name)

        self.remove_input(1)
        self.add_input(dataprocess.CObjectDetectionIO())
        self.cat_ids = None
        self.det_model = None
        self.det_config = None
        self.det_checkpoint = None
        self.det_score_thr = 0.5
        self.pose_model = None
        self.pose_config_path = None
        self.kpt_thr = 0.3
        self.device = "cpu"

        # Create parameters class
        if param is None:
            self.set_param_object(InferMmlabPoseEstimationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def load_models(self):
        param = self.get_param_object()
        self.device = "cuda" if param.cuda and torch.cuda.is_available() else 'cpu'
        old_torch_hub = torch.hub.get_dir()
        torch.hub.set_dir(os.path.join(os.path.dirname(__file__), "models"))

        if param.detector != "None":
            self.det_config, self.det_checkpoint = get_detection_config(param.detector)
            self.det_model = init_detector(self.det_config, self.det_checkpoint, device=self.device.lower())

        if param.detector == "Person":
            self.cat_ids = [0]
        elif param.detector in ["Hand", "Face"]:
            self.cat_ids = [0]
        else:
            self.cat_ids = []

        self.det_score_thr = param.conf_thres
        self.kpt_thr = param.conf_kp_thres

        self.pose_config_path, ckpt_pose = get_full_paths(param)
        cfg_pose = Config.fromfile(self.pose_config_path)
        dict_replace(cfg_pose, "SyncBN", "BN")
        
        tmp_cfg = NamedTemporaryFile(suffix='.py', delete=False)
        cfg_pose.dump(tmp_cfg.name)
        tmp_cfg.close()
        cfg_pose = tmp_cfg.name
        
        # build pose models
        self.pose_model = init_pose_estimator(cfg_pose, ckpt_pose, device=self.device.lower())
        torch.hub.set_dir(old_torch_hub)
        dataset_info = dataset_meta_from_config(Config.fromfile(cfg_pose), dataset_mode='val')

        if dataset_info is not None:
            skeleton_link_colors = dataset_info["skeleton_link_colors"]

            # Compute keypoint links
            keypoint_links = []
            for i, (id1, id2) in enumerate(dataset_info["skeleton_links"]):
                link = dataprocess.CKeypointLink()
                link.start_point_index = id1
                link.end_point_index = id2

                name1 = dataset_info["keypoint_id2name"][id1]
                name2 = dataset_info["keypoint_id2name"][id2]
                link.label = f"{name1} - {name2}"

                link.color = [int(c) for c in skeleton_link_colors[i]]
                keypoint_links.append(link)

            self.set_keypoint_links(keypoint_links)

        else:
            raise NotImplementedError()

        self.set_object_names([param.detector])
        self.set_keypoint_names(list(dataset_info["keypoint_id2name"].values()))
        
        # Remove temp file
        os.remove(tmp_cfg.name)
        param.update = False

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Examples :
        # Get input :
        img_input = self.get_input(0)
        detect_input = self.get_input(1)

        if self.pose_model is None or param.update:
            self.load_models()

        # To avoid registry error when running detection model
        # not using register_mmdet_modules because too many warnings
        DefaultScope.get_instance(f'mmdet-{datetime.datetime.now()}', scope_name='mmdet')

        if self.pose_model is None:
            print("Could not create model with chosen parameters")
            # Step progress bar:
            self.emit_step_progress()
            # Call end_task_run to finalize process
            self.end_task_run()
            return

        if img_input.is_data_available():
            # Get image from input/output (numpy array):
            src_image = img_input.get_image()
            cfg_pose = Config.fromfile(self.pose_config_path)

            if cfg_pose.data_mode == "topdown":
                if param.detector == "None":
                    bboxes = []
                    labels = []
                    det_scores = []

                    for idx, obj in enumerate(detect_input.get_objects()):
                        conf = obj.confidence
                        x, y, w, h = obj.box
                        box = [x, y, x + w, y + h]
                        label = obj.label
                        bboxes.append(box)
                        labels.append(label)
                        det_scores.append(conf)
                else:
                    pred_instance = inference_detector(self.det_model, src_image).pred_instances.detach().cpu().numpy()
                    bboxes = np.concatenate(
                        (pred_instance.bboxes, pred_instance.labels[:, None], pred_instance.scores[:, None]),
                        axis=1
                    )
                    bboxes = bboxes[np.logical_and(
                        logical_or([pred_instance.labels == i for i in self.cat_ids]),
                        pred_instance.scores > self.det_score_thr)
                    ]
                    bboxes = bboxes[nms(bboxes, self.det_score_thr)]
                    bboxes, labels, det_scores = bboxes[:, :4], bboxes[:, 4], bboxes[:, 5]

                # inference pose model
                pose_results = inference_topdown(self.pose_model, src_image, bboxes, bbox_format='xyxy')
                self.vis_pose_result(pose_results, *np.shape(src_image)[:2], det_scores,
                                     det_score_thr=self.det_score_thr, kpt_score_thr=self.kpt_thr)
            else:
                scope = self.pose_model.cfg.get('default_scope', 'mmpose')
                if scope is not None:
                    init_default_scope(scope)

                pose_results = inference_bottomup(self.pose_model, src_image)
                self.vis_pose_result(pose_results, *np.shape(src_image)[:2],
                                     det_score_thr=self.det_score_thr, kpt_score_thr=self.kpt_thr)
        else:
            print("Run the workflow with an image")

        # Set image of input/output (numpy array):
        self.forward_input_image(0, 0)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    def vis_pose_result(self, result, w, h, det_scores=None, det_score_thr=0.5, kpt_score_thr=0.3):
        obj_id = 0
        for obj in result:
            item = obj.pred_instances
            for i in range(len(item["bboxes"])):
                if item["bbox_scores"][i] >= det_score_thr:
                    keypoints = item["keypoints"][i]
                    keypoint_scores = item["keypoint_scores"][i]

                    if 'bboxes' in item:
                        x, y, x2, y2 = item["bboxes"][i]
                        w, h = x2 - x, y2 - y
                    else:
                        x, y = 0, 0

                    valid = [False] * len(keypoints)
                    for j, score in enumerate(keypoint_scores):
                        if score > kpt_score_thr:
                            valid[j] = True

                    keypts = []
                    for j, ckpt in enumerate(self.get_keypoint_links()):
                        idx1 = ckpt.start_point_index
                        idx2 = ckpt.end_point_index
                        kp1 = keypoints[idx1]
                        kp2 = keypoints[idx2]
                        pt1 = dataprocess.CPointF(float(kp1[0]), float(kp1[1]))
                        pt2 = dataprocess.CPointF(float(kp2[0]), float(kp2[1]))

                        if valid[idx1] and valid[idx2]:
                            keypts.append((idx1, pt1))
                            keypts.append((idx2, pt2))

                    if det_scores is None:
                        det_score = item["bbox_scores"][i]
                    else:
                        det_score = det_scores[i]

                    self.add_object(obj_id, 0, float(det_score), float(x), float(y), float(w), float(h), keypts)
                    obj_id += 1


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabPoseEstimationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_pose_estimation"
        self.info.short_description = "Inference for pose estimation models from mmpose"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Pose"
        self.info.version = "3.1.2"
        # self.info.min_python_version = "3.8.0"
        # self.info.max_python_version = "3.11.0"
        self.info.min_ikomia_version = "0.11.1"
        # self.info.max_ikomia_version = "0.11.0"
        self.info.icon_path = "icons/mmpose-logo.png"
        self.info.authors = "MMPose contributors"
        self.info.article = "OpenMMLab Pose Estimation Toolbox and Benchmark"
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://mmpose.readthedocs.io/en/v0.24.0/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_mmlab_pose_estimation"
        self.info.original_repository = "https://github.com/open-mmlab/mmpose"
        # Keywords used for search
        self.info.keywords = "infer, mmpose, pose, estimation, human, mmlab, hrnet, vipnas, body, hand, animal, 2D"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "KEYPOINTS_DETECTION"

    def create(self, param=None):
        # Create process object
        return InferMmlabPoseEstimation(self.info.name, param)
