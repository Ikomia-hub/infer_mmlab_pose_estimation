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
from ikomia.utils import pyqtutils, qtconversion
from infer_mmlab_pose_estimation.infer_mmlab_pose_estimation_process import InferMmlabPoseEstimationParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
import torch
import yaml


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferMmlabPoseEstimationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferMmlabPoseEstimationParam()
        else:
            self.parameters = param

        self.available_cfg_ckpt = {}
        self.cfg = ""
        self.ckpt = ""

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        is_cuda_available = torch.cuda.is_available()

        self.check_cuda = pyqtutils.append_check(self.gridLayout, "Use Cuda",
                                                 self.parameters.cuda and is_cuda_available)
        self.check_cuda.setChecked(self.parameters.cuda and is_cuda_available)
        self.check_cuda.setEnabled(is_cuda_available)

        self.configs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "mmpose_configs")

        self.combo_body_part = pyqtutils.append_combo(self.gridLayout, "Body part")
        self.combo_task = pyqtutils.append_combo(self.gridLayout, "Task")
        self.combo_method = pyqtutils.append_combo(self.gridLayout, "Method")
        self.combo_dataset = pyqtutils.append_combo(self.gridLayout, "Dataset")
        self.combo_model_name = pyqtutils.append_combo(self.gridLayout, "Model name")
        self.combo_config_name = pyqtutils.append_combo(self.gridLayout, "Config name")
        self.combo_detector = pyqtutils.append_combo(self.gridLayout, "Integrated detector")

        for dir in os.listdir(self.configs_path):
            dir_path = os.path.join(self.configs_path, self.parameters.body_part)
            if dir != "_base_" and os.path.isdir(dir_path):
                self.combo_body_part.addItem(dir)

        self.combo_body_part.currentTextChanged.connect(self.on_body_part_changed)
        self.combo_task.currentTextChanged.connect(self.on_task_changed)
        self.combo_method.currentTextChanged.connect(self.on_method_changed)
        self.combo_dataset.currentTextChanged.connect(self.on_dataset_changed)
        self.combo_model_name.currentTextChanged.connect(self.on_model_name_changed)
        self.combo_config_name.currentTextChanged.connect(self.on_config_name_changed)
        self.combo_body_part.setCurrentText(self.parameters.body_part)
        self.on_body_part_changed("")
        self.spin_det_thr = pyqtutils.append_double_spin(self.gridLayout, "Detection threshold", self.parameters.conf_thres
                                                         , min=0, max=1, step=0.01, decimals=2)
        self.spin_kp_thr = pyqtutils.append_double_spin(self.gridLayout, "Keypoint threshold",
                                                        self.parameters.conf_kp_thres, min=0, max=1, step=0.01, decimals=2)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_body_part_changed(self, s):
        supported_tasks = ["2d_kpt_sview_rgb_img"]
        if self.combo_body_part.currentText() != "":
            tasks = []
            self.combo_task.clear()
            current_body_part = self.combo_body_part.currentText()
            for dir in os.listdir(os.path.join(self.configs_path, current_body_part)):
                dir_path = os.path.join(self.configs_path, current_body_part, dir)
                if os.path.isdir(dir_path) and dir in supported_tasks:
                    self.combo_task.addItem(dir)
                    tasks.append(dir)
            if self.parameters.task in tasks:
                self.combo_task.setCurrentText(self.parameters.task)
            else:
                self.combo_task.setCurrentText(tasks[0])

    def on_task_changed(self, s):
        if self.combo_task.currentText() != "":
            methods = []
            self.combo_method.clear()
            current_body_part = self.combo_body_part.currentText()
            current_task = self.combo_task.currentText()
            for dir in os.listdir(os.path.join(self.configs_path, current_body_part, current_task)):
                dir_path = os.path.join(self.configs_path, current_body_part, current_task, dir)
                if os.path.isdir(dir_path):
                    self.combo_method.addItem(dir)
                    methods.append(dir)
            if self.parameters.method in methods:
                self.combo_method.setCurrentText(self.parameters.method)
            else:
                self.combo_method.setCurrentText(methods[0])

    def on_method_changed(self, s):
        if self.combo_method.currentText() != "":
            datasets = []
            self.combo_dataset.clear()
            current_body_part = self.combo_body_part.currentText()
            current_task = self.combo_task.currentText()
            current_method = self.combo_method.currentText()
            for dir in os.listdir(os.path.join(self.configs_path, current_body_part, current_task, current_method)):
                dir_path = os.path.join(self.configs_path, current_body_part, current_task, current_method, dir)
                if os.path.isdir(dir_path):
                    self.combo_dataset.addItem(dir)
                    datasets.append(dir)
            if self.parameters.dataset in datasets:
                self.combo_dataset.setCurrentText(self.parameters.dataset)
            else:
                self.combo_dataset.setCurrentText(datasets[0])

    def on_dataset_changed(self, s):
        if self.combo_dataset.currentText() != "":
            model_names = []
            self.combo_model_name.clear()
            current_body_part = self.combo_body_part.currentText()
            current_task = self.combo_task.currentText()
            current_method = self.combo_method.currentText()
            current_dataset = self.combo_dataset.currentText()
            for filename in os.listdir(os.path.join(self.configs_path, current_body_part, current_task, current_method,
                                                    current_dataset)):
                dir_path = os.path.join(self.configs_path, current_body_part, current_task, current_method,
                                        current_dataset, filename)
                if os.path.isfile(dir_path) and filename.endswith(".yml"):
                    self.combo_model_name.addItem(os.path.splitext(filename)[0])
                    model_names.append(os.path.splitext(filename)[0])
            if len(model_names) > 0:
                if self.parameters.model_name in model_names:
                    self.combo_model_name.setCurrentText(self.parameters.model_name)
                else:
                    self.combo_model_name.setCurrentText(model_names[0])

    def on_model_name_changed(self, s):
        self.combo_config_name.clear()
        if self.combo_model_name.currentText() != "":
            config_names = []
            current_body_part = self.combo_body_part.currentText()
            current_task = self.combo_task.currentText()
            current_method = self.combo_method.currentText()
            current_dataset = self.combo_dataset.currentText()
            current_model_name = self.combo_model_name.currentText()
            self.combo_config_name.show()
            yaml_file = os.path.join(self.configs_path, current_body_part, current_task, current_method,
                                     current_dataset, current_model_name + ".yml")
            with open(yaml_file, "r") as f:
                models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']

            self.available_cfg_ckpt = {model_dict["Name"]: {'cfg': model_dict["Config"],
                                                            'ckpt': model_dict["Weights"]}
                                       for
                                       model_dict in models_list}
            for experiment_name in self.available_cfg_ckpt.keys():
                self.combo_config_name.addItem(experiment_name)
                config_names.append(experiment_name)

            if self.parameters.config_name in config_names:
                self.combo_config_name.setCurrentText(self.parameters.config_name)
            else:
                self.combo_config_name.setCurrentText(list(self.available_cfg_ckpt.keys())[0])

    def on_config_name_changed(self, s):
        if self.combo_config_name.currentText() != "":
            current_config_name = self.combo_config_name.currentText()
            self.combo_detector.clear()
            self.combo_detector.addItem("None")
            if current_config_name in self.available_cfg_ckpt:
                current_body_part = self.combo_body_part.currentText()
                current_task = self.combo_task.currentText()
                current_method = self.combo_method.currentText()
                current_dataset = self.combo_dataset.currentText()
                selected_config = os.path.basename(self.available_cfg_ckpt[current_config_name]["cfg"])
                self.cfg = os.path.join(current_body_part, current_task, current_method,
                                        current_dataset, selected_config)
                self.ckpt = self.available_cfg_ckpt[current_config_name]["ckpt"]
                if current_method == 'associative_embedding':
                    pass
                elif current_body_part in ['body', 'animal', 'wholebody']:
                    self.combo_detector.addItem('coco')
                    if self.parameters.detector in ['None', 'coco']:
                        self.combo_detector.setCurrentText(self.parameters.detector)
                    else:
                        self.combo_detector.setCurrentText('coco')
                elif current_body_part == 'hand':
                    self.combo_detector.addItem('hand')
                    if self.parameters.detector in ['None', 'hand']:
                        self.combo_detector.setCurrentText(self.parameters.detector)
                    else:
                        self.combo_detector.setCurrentText('hand')
                else:
                    self.combo_detector.setCurrentText('None')

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.update = True
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.body_part = self.combo_body_part.currentText()
        self.parameters.task = self.combo_task.currentText()
        self.parameters.method = self.combo_method.currentText()
        self.parameters.dataset = self.combo_dataset.currentText()
        self.parameters.model_name = self.combo_model_name.currentText()
        self.parameters.config_name = self.combo_config_name.currentText()
        self.parameters.conf_thres = self.spin_det_thr.value()
        self.parameters.conf_kp_thres = self.spin_kp_thr.value()
        self.parameters.model_url = self.ckpt
        self.parameters.config = self.cfg
        self.parameters.detector = self.combo_detector.currentText()
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferMmlabPoseEstimationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_mmlab_pose_estimation"

    def create(self, param):
        # Create widget object
        return InferMmlabPoseEstimationWidget(param, None)
