# TASC: Task-Aware Shared Control for Teleoperated Manipulation

This is the official implementation for the paper "TASC: Task-Aware Shared Control for Teleoperated Manipulation".

## Installation
### Install `tasc`
```bash
conda create -n tasc python=3.8
conda activate tasc
pip install -r requirements.txt
pip install -e .
```

### Install Other Dependencies
> **Note:** These installations are optional. We provide offline segmentation and grasp pose results for quick testing. Install this if you want to try the complete pipeline.


- `Grounded-SAM`

    Clone the [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) repository, set some environment variables and install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://arxiv.org/abs/2304.02643).
    ```bash
    git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
    export AM_I_DOCKER=False
    export BUILD_WITH_CUDA=True
    export CUDA_HOME=/path/to/cuda/
    python -m pip install -e segment_anything
    pip install --no-build-isolation -e GroundingDINO
    ```

    Download DINO and SAM checkpoints, in TASC root directory:
    ```bash
    mkdir ckpts && cd ckpts
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    ```

- `GraspNet`

1. Clone the repo.
    ```bash
    cd ~/tasc
    git clone https://github.com/graspnet/graspnet-baseline.git
    cd graspnet-baseline
    ```
2. Install packages, pointnet2 operators, knn operator and graspnetAPI.
    ```bash
    pip install -r requirements.txt
    cd pointnet2
    python setup.py install
    cd ../knn
    python setup.py install
    cd ..
    git clone https://github.com/graspnet/graspnetAPI.git
    cd graspnetAPI
    pip install .
    ```
3. Download GraspNet checkpoints: [`checkpoint-rs.tar`](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing), in TASC root directory:
    ```bash
    mv ./checkpoint-rs.tar ~/tasc/ckpts
    ```
    Please refer to [graspnet-baseline repo](https://github.com/graspnet/graspnet-baseline) for troubleshooting.

## Usage

Run TASC with stored query results:
```bash
python tasc/tasc_simulation.py 
```
Franka Panda Teleoperation with keyboard:

```
Keys                            Command
Ctrl+q                          reset simulation
spacebar                        toggle gripper (open/close)
up-right-down-left              move horizontally in x-y plane
.-;                             move vertically
o-p                             rotate (yaw)
y-h                             rotate (pitch)
e-r                             rotate (roll)
```

Other arguments:
```bash
# Run TASC with debug information
python tasc/tasc_simulation.py --debug

# Run TASC with online query
python tasc/tasc_simulation.py --use_llm

# Run TASC with grasping planner
python tasc/tasc_simulation.py --use_graspnet

# Run in pure teleoperation mode (for comparison)
python tasc/tasc_simulation.py --pure_teleop

# Record operation data
python tasc/tasc_simulation.py --record
```

Analyse operation data:
```bash
python tasc/result_analyzer.py --input tasc --episode_idx 1 --task_idx 1
```

## Running on a Headless Server
If you need to run TASC on a headless server (no monitor attached) and display the GUI on your local machine, you can use X11 forwarding.

### On Windows
1. Install [XLaunch](https://sourceforge.net/projects/vcxsrv/) or [Xming](https://sourceforge.net/projects/xming/).
2. Start XLaunch (keep default settings).
3. In PowerShell, set the display environment variable:
```powershell
$env:DISPLAY="localhost:0.0"
```
4. Connect to the server with X11 forwarding enabled:
```powershell
ssh -Y <user>@<server-ip>
```

### On Linux/macOS
1. Make sure you have an X server installed locally (Linux usually has it by default; macOS can use [XQuartz](https://www.xquartz.org/)).
2. Connect to the server with X11 forwarding:
```bash
ssh -Y <user>@<server-ip>
```

## Real World

See our real-world setup in this [repo](https://github.com/fitz0401/tasc_ros2_ws).

<img src="assets/real_world_setting.png" alt="system architecture" width="50%">

In shrot, we use [moveit_servo](https://moveit.picknik.ai/main/doc/how_to_guides/controller_teleoperation/controller_teleoperation.html) to teleoperate a [Franka Research3 robot in ROS2](https://github.com/frankarobotics/franka_ros2) environment. We also seperate the ROS2 and CUDA environment and use websocket to communicate, to aviod python version conflict.


## Acknowledgements
We gratefully acknowledge the following open-source projects that greatly facilitated our work:

- Segmentation pipeline adapted from [MOKA](https://github.com/moka-manipulation/moka/tree/main).
- Goal prediction adapted from [Improved APF](https://github.com/Shared-control/improved-apf).

We also thank the [robosuite](https://github.com/ARISE-Initiative/robosuite) community for providing an excellent simulation framework that supported our research.