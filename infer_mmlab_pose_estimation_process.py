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

from mmpose.apis import inference_top_down_pose_model, init_pose_model, inference_bottom_up_pose_model, vis_pose_result
from mmdet.apis import inference_detector, init_detector
from infer_mmlab_pose_estimation.utils import process_mmdet_results, det_model_zoo, vis_pose_result, are_params_valid
import numpy as np


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

    def setParamMap(self, param_map):
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

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
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
class InferMmlabPoseEstimation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Example :  self.addInput(dataprocess.CImageIO())
        #           self.addOutput(dataprocess.CImageIO())
        self.addOutput(dataprocess.CGraphicsOutput())
        self.det_model = None
        self.pose_model = None
        self.namespace = None
        # Create parameters class
        if param is None:
            self.setParam(InferMmlabPoseEstimationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        # Examples :
        # Get input :
        input = self.getInput(0)
        # Get output :
        output = self.getOutput(0)
        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("Mmpose-Keypoints")
        graphics_output.setImageIndex(0)
        graphics_input = self.getInput(1)

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

            param.update = False

        if self.pose_model is None:
            print("Could not create model with chosen parameters")
            # Step progress bar:
            self.emitStepProgress()
            # Call endTaskRun to finalize process
            self.endTaskRun()
            return

        if input.isDataAvailable():
            # Get image from input/output (numpy array):
            srcImage = input.getImage()

            if self.no_detector:
                det_results = []
                if graphics_input.isDataAvailable():
                    for item in graphics_input.getItems():
                        bbox = None
                        if isinstance(item, (core.CGraphicsRectangle, core.CGraphicsEllipse)):
                            h, w, x, y = item.height, item.width, item.x, item.y
                            bbox = [x, y, x + w, y + h]
                        elif isinstance(item, core.CGraphicsPolygon):
                            pts = item.points
                            pts = np.array([[pt.x, pt.y] for pt in pts])
                            bbox = [min(pts[:, 0]), min(pts[:, 1]), max(pts[:, 0]), max(pts[:, 1])]
                        if bbox is not None:
                            det_results.append({'bbox': bbox, 'label': item.getCategory()})
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

            bbox_color = (148, 139, 255)
            vis_pose_result(
                self.pose_model,
                pose_results,
                graphics_output,
                kpt_score_thr=self.namespace.kpt_thr,
                bbox_color=bbox_color)

        else:
            print("Run the workflow with an image")

        # Set image of input/output (numpy array):
        self.forwardInputImage(0, 0)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabPoseEstimationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_pose_estimation"
        self.info.shortDescription = "Inference for pose estimation models from mmpose"
        self.info.description = "Inference for pose estimation models from mmpose"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Pose"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/mmpose-logo.png"
        self.info.authors = "MMPose contributors"
        self.info.article = "OpenMMLab Pose Estimation Toolbox and Benchmark"
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentationLink = "https://mmpose.readthedocs.io/en/v0.24.0/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmpose"
        # Keywords used for search
        self.info.keywords = "infer, mmpose, pose, estimation, human, mmlab, hrnet, vipnas, body, hand, animal, 2D"

    def create(self, param=None):
        # Create process object
        return InferMmlabPoseEstimation(self.info.name, param)
