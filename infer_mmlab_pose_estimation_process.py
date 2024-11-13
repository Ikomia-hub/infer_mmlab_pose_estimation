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
import numpy as np

from ikomia import utils, core, dataprocess

from infer_mmlab_pose_estimation.core.inference import PoseInference


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
        self.body_part = "body_2d_keypoint"
        self.method = "rtmo"
        self.dataset = "coco"
        self.model_name = "rtmo_coco"
        self.config_name = "rtmo-m_16xb16-600e_coco-640x640"
        self.config_file = ""
        self.model_weight_file = ""
        self.detector = "None"

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = utils.strtobool(param_map["cuda"])
        self.conf_thres = float(param_map["conf_thres"])
        self.conf_kp_thres = float(param_map["conf_kp_thres"])
        self.body_part = param_map["body_part"]
        self.method = param_map["method"]
        self.dataset = param_map["dataset"]
        self.model_name = param_map["model_name"]
        self.config_name = param_map["config_name"]
        self.config_file = param_map["config_file"]
        self.model_weight_file = param_map["model_weight_file"]
        self.detector = param_map["detector"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "cuda": str(self.cuda),
            "conf_thres": str(self.conf_thres),
            "conf_kp_thres": str(self.conf_kp_thres),
            "body_part": self.body_part,
            "method": self.method,
            "dataset": self.dataset,
            "model_name": self.model_name,
            "config_name": self.config_name,
            "config_file": self.config_file,
            "model_weight_file": self.model_weight_file,
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

        # Create parameters class
        if param is None:
            self.set_param_object(InferMmlabPoseEstimationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.inference = PoseInference(self.get_param_object())

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        img_input = self.get_input(0)
        detect_input = self.get_input(1)

        if not param.config_file:
            param.config_file = os.path.join(
                "configs", param.body_part, param.method, param.dataset, f"{param.config_name}.py"
            )

        if not self.inference.is_model_loaded() or param.update:
            self.inference.load_models()
            self.set_keypoint_links(self.inference.keypoint_links)
            self.set_object_names([param.detector])
            self.set_keypoint_names(list(self.inference.dataset_info["keypoint_id2name"].values()))
            param.update = False

        if not self.inference.is_model_loaded():
            raise RuntimeError("Could not create model with chosen parameters")

        if img_input.is_data_available():
            # Get image from input/output (numpy array):
            src_image = img_input.get_image()
            pose_results, det_scores = self.inference.run(src_image, detect_input)
            self.process_pose_result(pose_results, *np.shape(src_image)[:2], det_scores)
        else:
            raise RuntimeError("Input image could not be empty.")

        # Set image of input/output (numpy array):
        self.forward_input_image(0, 0)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    def process_pose_result(self, result, w, h, det_scores=None):
        param = self.get_param_object()
        obj_id = 0
        item = result.pred_instances

        for i in range(len(item["bboxes"])):
            if item["bbox_scores"][i] >= param.conf_thres:
                if "transformed_keypoints" in item:
                    keypoints = item["transformed_keypoints"][i]
                else:
                    keypoints = item["keypoints"][i]

                keypoint_scores = item["keypoint_scores"][i]

                if 'bboxes' in item:
                    x, y, x2, y2 = item["bboxes"][i]
                    w, h = x2 - x, y2 - y
                else:
                    x, y = 0, 0

                valid = [False] * len(keypoints)
                for j, score in enumerate(keypoint_scores):
                    if score > param.conf_kp_thres:
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
        self.info.version = "3.2.0"
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
