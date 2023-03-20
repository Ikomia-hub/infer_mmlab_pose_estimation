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

from ikomia import core, dataprocess
import copy
# Your imports below
from argparse import Namespace
from distutils.util import strtobool
from mmcv import Config

from mmpose.apis import inference_top_down_pose_model, init_pose_model, inference_bottom_up_pose_model
from mmdet.apis import inference_detector, init_detector
from infer_mmlab_pose_estimation.utils import process_mmdet_results, det_model_zoo, are_params_valid
import numpy as np
from mmpose.datasets.dataset_info import DatasetInfo


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabPoseEstimationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.cuda = True
        self.update = False
        self.body_part = "body"
        self.task = "2d_kpt_sview_rgb_img"
        self.method = "topdown_heatmap"
        self.dataset = "coco"
        self.model_name = "vipnas_coco"
        self.config_name = "topdown_heatmap_vipnas_res50_coco_256x192"
        self.det_thr = 0.5
        self.kp_thr = 0.3
        self.cfg_pose = "body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py"
        self.ckpt_pose = "https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192-cc43b466_20210624.pth"
        self.detector = "coco"

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.cuda = strtobool(param_map["cuda"])
        self.body_part = param_map["body_part"]
        self.task = param_map["task"]
        self.method = param_map["method"]
        self.dataset = param_map["dataset"]
        self.model_name = param_map["model_name"]
        self.config_name = param_map["config_name"]
        self.det_thr = float(param_map["det_thr"])
        self.kp_thr = float(param_map["kp_thr"])
        self.ckpt_pose = param_map["ckpt_pose"]
        self.cfg_pose = param_map["cfg_pose"]
        self.detector = param_map["detector"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["cuda"] = str(self.cuda)
        param_map["model_name"] = self.model_name
        param_map["body_part"] = self.body_part
        param_map["task"] = self.task
        param_map["method"] = self.method
        param_map["dataset"] = self.dataset
        param_map["config_name"] = self.config_name
        param_map["det_thr"] = str(self.det_thr)
        param_map["kp_thr"] = str(self.kp_thr)
        param_map["ckpt_pose"] = self.ckpt_pose
        param_map["cfg_pose"] = self.cfg_pose
        param_map["detector"] = self.detector
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabPoseEstimation(dataprocess.CKeypointDetectionTask):

    def __init__(self, name, param):
        dataprocess.CKeypointDetectionTask.__init__(self, name)

        self.det_model = None
        self.pose_model = None
        self.namespace = None
        # Create parameters class
        if param is None:
            self.set_param_object(InferMmlabPoseEstimationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def vis_pose_result(self,
                        result,
                        w,
                        h,
                        kpt_score_thr=0.3):

        obj_id = 0
        for item in result:
            keypoints = item["keypoints"]
            if 'bbox' in item:
                x, y, x2, y2 = item["bbox"][:4]
                if len(item["bbox"]) == 5:
                    conf = item["bbox"][-1]
                else:
                    conf = 0.
                w, h = x2 - x, y2 - y
            else:
                x, y = 0, 0

            valid = [False] * len(keypoints)

            for i, (x_kp, y_kp, score) in enumerate(keypoints):
                if score > kpt_score_thr:
                    valid[i] = True

            keypts = []
            for i, ckpt in enumerate(self.get_keypoint_links()):
                idx1 = ckpt.start_point_index
                idx2 = ckpt.end_point_index
                kp1 = keypoints[idx1]
                kp2 = keypoints[idx2]
                pt1 = dataprocess.CPointF(float(kp1[0]), float(kp1[1]))
                pt2 = dataprocess.CPointF(float(kp2[0]), float(kp2[1]))

                if valid[idx1] and valid[idx2]:
                    keypts.append((idx1, pt1))
                    keypts.append((idx2, pt2))
            self.add_object(obj_id, 0, float(conf), float(x), float(y), float(w), float(h), keypts)
            obj_id += 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Examples :
        # Get input :
        input = self.get_input(0)
        graphics_input = self.get_input(1)

        if (self.pose_model is None or param.update) and are_params_valid(param):
            cfg_dir = os.path.join(os.path.dirname(__file__), "configs")
            cfg_pose = os.path.join(cfg_dir, "mmpose_configs", param.cfg_pose)
            ckpt_pose = param.ckpt_pose
            self.no_detector = param.detector == "None"
            self.namespace = Namespace()

            if not self.no_detector:
                cfg_det = os.path.join(cfg_dir, det_model_zoo[param.detector]["cfg"])
                ckpt_det = det_model_zoo[param.detector]["ckpt"]
                self.namespace.det_config = cfg_det
                self.namespace.det_checkpoint = ckpt_det

            self.namespace.device = "cuda" if param.cuda else 'cpu'
            self.namespace.enable_animal_pose = False
            self.namespace.enable_human_pose = True

            self.namespace.pose_config = cfg_pose
            self.namespace.pose_checkpoint = ckpt_pose

            if not self.no_detector:
                if param.body_part in ['body', 'face', 'wholebody', 'hand']:
                    self.namespace.cat_ids = [1]
                elif param.body_part in ['animal']:
                    # 15: ‘bird’, 16: ‘cat’, 17: ‘dog’, 18: ‘horse’, 19: ‘sheep’, 20: ‘cow’, 21: ‘elephant’, 22: ‘bear’,
                    # 23: ‘zebra’, 24: ‘giraffe’
                    self.namespace.cat_ids = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

            self.namespace.det_score_thr = param.det_thr
            self.namespace.kpt_thr = param.kp_thr

            cfg_pose = Config.fromfile(self.namespace.pose_config)
            # cfg_pose.data_cfg["image_size"] = [288, 384]
            # build detection model
            if not self.no_detector:
                self.det_model = init_detector(self.namespace.det_config, self.namespace.det_checkpoint,
                                               device=self.namespace.device.lower())

            # build pose models
            if self.namespace.enable_human_pose:
                self.pose_model = init_pose_model(
                    cfg_pose,
                    self.namespace.pose_checkpoint,
                    device=self.namespace.device.lower())

            dataset_info = None
            # get dataset info
            if hasattr(self.pose_model, 'cfg') and ('dataset_info' in self.pose_model.cfg):
                dataset_info = DatasetInfo(self.pose_model.cfg.dataset_info)

            if dataset_info is not None:
                skeleton = dataset_info.skeleton
                pose_kpt_color = dataset_info.pose_kpt_color
                pose_link_color = dataset_info.pose_link_color

                # Compute keypoint links
                keypoint_links = []
                for i ,(id1, id2) in enumerate(dataset_info.skeleton):
                    link = dataprocess.CKeypointLink()
                    link.start_point_index = id1
                    link.end_point_index = id2

                    name1 = dataset_info.keypoint_id2name[id1]
                    name2 = dataset_info.keypoint_id2name[id2]
                    link.label = f"{name1} - {name2}"

                    link.color = [int(c) for c in pose_link_color[i]]
                    keypoint_links.append(link)

                self.set_keypoint_links(keypoint_links)

            else:
                NotImplementedError()

            self.set_object_names(["person"])
            self.set_keypoint_names(list(dataset_info.keypoint_id2name.values()))

            param.update = False

        if self.pose_model is None:
            print("Could not create model with chosen parameters")
            # Step progress bar:
            self.emit_step_progress()
            # Call end_task_run to finalize process
            self.end_task_run()
            return

        if input.is_data_available():
            # Get image from input/output (numpy array):
            srcImage = input.get_image()

            if self.no_detector:
                det_results = []
                if graphics_input.is_data_available():
                    for item in graphics_input.get_items():
                        bbox = None
                        if isinstance(item, (core.CGraphicsRectangle, core.CGraphicsEllipse)):
                            h, w, x, y = item.height, item.width, item.x, item.y
                            bbox = [x, y, x + w, y + h]
                        elif isinstance(item, core.CGraphicsPolygon):
                            pts = item.points
                            pts = np.array([[pt.x, pt.y] for pt in pts])
                            bbox = [min(pts[:, 0]), min(pts[:, 1]), max(pts[:, 0]), max(pts[:, 1])]
                        if bbox is not None:
                            det_results.append({'bbox': bbox, 'label': item.get_category()})
                det_results = det_results if len(det_results) else None
                self.namespace.det_score_thr = None
            else:
                mmdet_results = inference_detector(self.det_model, srcImage)
                det_results = process_mmdet_results(mmdet_results, class_names=self.det_model.CLASSES,
                                                    cat_ids=self.namespace.cat_ids)

            # inference pose model
            if param.method in ['topdown_heatmap', 'deeppose']:
                pose_results, _ = inference_top_down_pose_model(
                    self.pose_model,
                    srcImage,
                    det_results,
                    bbox_thr=self.namespace.det_score_thr,
                    format='xyxy')
            elif param.method == 'associative_embedding':
                pose_results, _ = inference_bottom_up_pose_model(
                    self.pose_model,
                    srcImage,
                    det_results,
                    pose_nms_thr=0.9)
            else:
                raise Exception("Method {} doesn't exist. Choose one of 'topdown_heatmap', 'deeppose' or 'associative_embedding".format(param.method))

            bbox_color = (148, 139, 255)
            self.vis_pose_result(
                pose_results,
                *np.shape(srcImage)[:2],
                kpt_score_thr=self.namespace.kpt_thr)

        else:
            print("Run the workflow with an image")

        # Set image of input/output (numpy array):
        self.forward_input_image(0, 0)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


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
        self.info.description = "Inference for pose estimation models from mmpose"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Pose"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/mmpose-logo.png"
        self.info.authors = "MMPose contributors"
        self.info.article = "OpenMMLab Pose Estimation Toolbox and Benchmark"
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://mmpose.readthedocs.io/en/v0.24.0/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmpose"
        # Keywords used for search
        self.info.keywords = "infer, mmpose, pose, estimation, human, mmlab, hrnet, vipnas, body, hand, animal, 2D"

    def create(self, param=None):
        # Create process object
        return InferMmlabPoseEstimation(self.info.name, param)
