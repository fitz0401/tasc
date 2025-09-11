
'''
Table Environment for Robotic Manipulation

Provides a customized robosuite environment with table-top manipulation setup,
camera integration, and object placement for TASC experiments.
'''

import os
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.models.objects import MujocoXMLObject
from xml.etree import ElementTree as ET
from robosuite.utils.camera_utils import (get_camera_intrinsic_matrix, get_real_depth_map)


def get_camera_data(sim, camera_name: str, width: int, height: int):
    '''
    Extract RGB image, point cloud, segmentation, and depth from simulation camera.
    Args:
        sim: MuJoCo simulation instance
        camera_name: Name of camera to render from
        width: Image width in pixels
        height: Image height in pixels
    Returns:
        tuple: (rgb, points, seg, depth) arrays for vision processing
    '''
    # Get RGB, depth image and segmentation mask -> Never use this because of Robosuite won't freshly render the image
    # rgb = obs[camera_name + "_image"]
    # depth = obs[camera_name + "_depth"].squeeze()
    # seg = obs[camera_name + "_segmentation_class"].squeeze()
    rgb, depth = sim.render(width=width, height=height, camera_name=camera_name, depth=True)
    rgb = rgb[::-1, :, :]  # flipud
    depth = depth[::-1, :]
    seg = sim.render(width=width, height=height, camera_name=camera_name, segmentation=True)[::-1, :]
    seg = seg[..., 1]   # 0th channel is the class id, 1st channel is the instance id

    # convert depth to meters (https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160)
    depth = get_real_depth_map(sim, depth)
    # logging.debug(f"Depth range: {np.min(depth)} to {np.max(depth)}")

    # Intrinsic parameters (https://github.com/haonan16/Stow/blob/b3d3045a64992a190bbad9f25b14e83b45d2ae8b/perception/sample.py#L46-L120)
    intrinsic_matrix = get_camera_intrinsic_matrix(sim, camera_name, height, width)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # Convert depth image to point cloud (https://blog.csdn.net/weixin_53610475/article/details/135610636)
    xmap, ymap = np.arange(width), np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    z = depth
    x = (xmap - cx) * z / fx
    y = (ymap - cy) * z / fy
    points = np.stack((x, -y, -z), axis=-1)

    return rgb, points, seg, depth


class TableEnv(ManipulationEnv):
    '''
    Customized robosuite environment for table-top manipulation tasks.
    Features:
    - Configurable table setup with multiple objects (banana, hammer, plate, etc.)
    - Integrated camera system with point cloud generation
    - Object placement initialization for reproducible experiments
    - Support for both visual and object-based observations
    Inherits from ManipulationEnv with specialized object handling and camera integration.
    '''
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 1.0, 0.05),
        table_friction=(1.5, 0.005, 0.0001),
        table_offset=(0, 0, 0.9),
        use_camera_obs=True,
        use_object_obs=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="birdview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # task settings

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = table_offset

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        '''
        Compute reward for current state (returns 0.0 for teleoperation tasks).
        '''
        return 0.0

    def _load_model(self):
        '''
        Load and configure simulation model with table, objects, and camera setup.
        Creates table arena, adds manipulation objects (banana, hammer, plate, etc.),
        configures object placement samplers, and sets up table_view camera.
        '''
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Add camera to the arena
        arena_body = mujoco_arena.worldbody
        table_cam = ET.Element("camera", 
            name="table_view",
            pos="0.441 -0.017 1.740",
            xyaxes="-0.000 1.000 -0.000 -0.897 -0.000 0.443"
        )
        arena_body.append(table_cam)

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Create default (SequentialCompositeSampler) sampler if it has not already been specified
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        # Load objects
        current_dir = os.path.dirname(os.path.abspath(__file__))
        object_names = ["banana", "hammer", "plate", "marker", "mug", "wood_block"]
        self.objects = {}
        for name in object_names:
            xml_path = os.path.join(current_dir, f"objects/{name}.xml")
            self.objects[name] = MujocoXMLObject(fname=xml_path, name=name)

        # Add objects to the placement initializer
        object_pos_configs = {
            "banana":       {"center": [-0.15, 0.05], "radius": 0.015, "rotation": None},
            "hammer":       {"center": [-0.10, -0.20], "radius": 0.015, "rotation": [np.pi/4, 3*np.pi/4]},
            "plate":        {"center": [0.20, -0.20],"radius": 0.015, "rotation": None},
            "marker":       {"center": [-0.10, 0.20], "radius": 0.005, "rotation": None},
            "mug":          {"center": [0.15, 0.05], "radius": 0.005, "rotation": None},
            "wood_block":   {"center": [0.0, 0.25], "radius": 0.005, "rotation": [0, np.pi/2]},
        }
        for i, (name, obj) in enumerate(self.objects.items()):
            cfg = object_pos_configs[name]
            center_x, center_y = cfg["center"]
            radius = cfg["radius"]
            rotation = cfg["rotation"]
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name=f"{name.capitalize()}Sampler",
                    mujoco_objects=obj,
                    x_range=[center_x - radius, center_x + radius],
                    y_range=[center_y - radius, center_y + radius],
                    rotation=rotation,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.02,
                )
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.objects.values()),
        )

    def _setup_references(self):
        '''
        Initialize references to simulation bodies and geometries for efficient access.
        Sets up object body/geometry ID mappings and table reference for collision detection.
        '''
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        self.table_body_id = self.sim.model.body_name2id("table")


    def _setup_observables(self):
        '''
        Configure observable sensors for environment state monitoring.
        
        Returns:
            OrderedDict: Mapping of observable names to Observable objects
        '''
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            # Reset nut sensor mappings
            self.nut_id_to_sensors = {}
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors = [
                self._get_world_pose_in_gripper_sensor(full_pf, f"world_pose_in_{arm_pf}gripper", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [fn.__name__ for fn in sensors]
            actives = [False] * len(sensors)
            enableds = [True] * len(sensors)

            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )

        return observables

    def _reset_internal(self):
        '''
        Reset simulation state and randomize object/robot positions.
        
        Samples new object placements and resets robot to initial joint configuration.
        '''
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        init_qpos = [0.011, -0.368, -0.013, -2.502, -0.006,  2.369,  0.776]
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
            
            for i, joint in enumerate(self.robots[0].robot_model.joints):
                self.sim.data.set_joint_qpos(joint, init_qpos[i])

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)