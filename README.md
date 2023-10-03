<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_pose_estimation/main/icons/mmpose-logo.png" alt="Algorithm icon">
  <h1 align="center">infer_mmlab_pose_estimation</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_mmlab_pose_estimation">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_mmlab_pose_estimation">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_mmlab_pose_estimation/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_mmlab_pose_estimation.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Inference for pose estimation models from mmpose.

![basket mmpose kp](https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_pose_estimation/main/icons/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

wf = Workflow()

algo = wf.add_task(name = 'infer_mmlab_pose_estimation', auto_connect=True)

wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

display(algo.get_image_with_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **config_file** (str): Path to the .py config file.
- **model_weight_file** (str): Path or URL to model weights file .pth. Optional if config_file come from method get_model_zoo (see below for more information).
- **conf_thres** (float) default '0.5': Threshold of Non Maximum Suppression. It will retain Object Keypoint Similarity overlap when inferior to ‘conf_thres’, [0,1].
- **conf_kp_thres** (float) default '0.3': Threshold of the keypoint visibility. It will calculate Object Keypoint Similarity based on those keypoints whose visibility higher than ‘conf_kp_thres’, [0,1].
- **detector**: object detector, ‘Person’, ‘Hand’, Face’. 


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mmlab_pose_estimation", auto_connect=True)

algo.set_parameters({
    "config_file": "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_vipnas-mbv3_8xb64-210e_coco-256x192.py",
    "conf_thres": "0.5",
    "conf_kp_thres": "0.3",
    "detector": "Person",
})

# Run on your image  
wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

display(algo.get_image_with_graphics())

```

You can get the full list of available **config_file** by running this code snippet:
```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mmlab_pose_estimation", auto_connect=True)

# Get pretrained models
model_zoo = algo.get_model_zoo()

# Print possibilities
for parameters in model_zoo:
    print(parameters)
```
## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mmlab_pose_estimation", auto_connect=True)

# Run on your image  
wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```


