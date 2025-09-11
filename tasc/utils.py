import os
import yaml
import logging
import cv2
import numpy as np
from rich.logging import RichHandler
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d as o3d

file_dir = os.path.dirname(os.path.abspath(__file__))


## =========== Global Utils ===========

def get_logger(name: str, verbose: bool = False, clean_root: bool = False) -> logging.Logger:
    """
    Get a logger with the specified name.
    """
    if clean_root:
        logging.getLogger().handlers.clear()
    
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        handler = RichHandler(markup=True)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        logger.setLevel(log_level)
        logger.addHandler(handler)
        logger.propagate = False

    return logger

def get_config(config_path=None):
    """
    Get the configuration from the specified YAML file.
    """
    if config_path is None:
        config_path = os.path.join(file_dir, '../configs/config.yaml')
    assert os.path.exists(config_path), f'config file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    return config


## =========== MLLM Client Utils ===========

def extract_tag_content(content, tag):
    """
    Extract the content of a tag from a string.
    """
    start = content.find(f"<{tag}>")
    end = content.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return content[start + len(tag) + 2 : end]


## =========== Perception Utils ===========

def farthest_point_sampling(points, num_samples, return_idx=False):
    """
    Perform farthest point sampling on a set of points.
    """
    if len(points) == 0:
        raise ValueError("No points to sample from.")
    if len(points) <= num_samples:
        idx = np.arange(len(points))
        return (points, idx) if return_idx else points
    # Select the first point randomly
    idx_sel = np.random.randint(len(points))
    sampled_points = [points[idx_sel]]
    sampled_idx = [idx_sel]
    for _ in range(num_samples - 1):
        # Calculate the distance of each point to the closest selected point
        dist_to_closest = np.min(
            np.linalg.norm(points - np.array(sampled_points)[:, np.newaxis], axis=2),
            axis=0,
        )
        # Select the farthest point
        farthest_idx = np.argmax(dist_to_closest)
        sampled_points.append(points[farthest_idx])
        sampled_idx.append(farthest_idx)
    if return_idx:
        return np.array(sampled_points), np.array(sampled_idx)
    return np.array(sampled_points)

def extract_ordered_masks(seg, sim, mask_name_dict):
    """
    Extract ordered masks from segmentation.
    """
    # geom_id → name
    all_geom_ids = np.unique(seg)
    geomid_to_name = {}
    for gid in all_geom_ids:
        if gid == 0:
            continue
        try:
            name = sim.model.geom_id2name(gid)
            geomid_to_name[gid] = name
        except:
            continue
    # name → mask
    name_to_mask = {}
    for gid, name in geomid_to_name.items():
        name_to_mask[name] = (seg == gid).astype(np.uint8)
    final_masks = []
    for id, name in mask_name_dict.items():
        if name == "wood_block_body":
            mask1 = name_to_mask.get("wood_block_body", np.zeros_like(seg, dtype=np.uint8))
            mask2 = name_to_mask.get("wood_block_nail_visual", np.zeros_like(seg, dtype=np.uint8))
            combined_mask = np.logical_or(mask1, mask2).astype(bool)
            final_masks.append(combined_mask)
        else:
            mask = name_to_mask.get(name, np.zeros_like(seg, dtype=np.uint8))
            final_masks.append(mask.astype(bool))
    return final_masks


## =========== Visualization ===========

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

def project_point_world2img(point_3d_world, intrinsic_matrix, T_world2cam, image_size=[640, 480], 
                            is_mujoco_camera=True):
    """
    Project a 3D point in world coordinates to 2D image coordinates.
    """
    point_3d_homo = np.append(point_3d_world, 1.0)
    point_3d_cam = T_world2cam @ point_3d_homo
    x, y, z = point_3d_cam[:3]
    # mujoco camera
    if is_mujoco_camera:
        y *= -1
        z *= -1
    u = int((x * intrinsic_matrix[0, 0] / z) + intrinsic_matrix[0, 2])
    v = int((y * intrinsic_matrix[1, 1] / z) + intrinsic_matrix[1, 2])
    u = max(0, min(u, image_size[0] - 1))
    v = max(0, min(v, image_size[1] - 1))
    return u, v

def draw_pose_on_image(image, pose, T_cam2world, intrinsic_matrix, is_mujoco_camera=True):
    """
    Draw the 3D pose on the 2D image.
    """
    rotation_matrix = pose[:3, :3]
    translation_vector = pose[:3, 3]
    axis_length = 0.05
    x_axis = translation_vector + axis_length * rotation_matrix[:, 0]
    y_axis = translation_vector + axis_length * rotation_matrix[:, 1]
    z_axis = translation_vector + axis_length * rotation_matrix[:, 2]
    
    T_world2cam = np.linalg.inv(T_cam2world)
    center_2d = project_point_world2img(translation_vector, intrinsic_matrix, T_world2cam, is_mujoco_camera=is_mujoco_camera)
    x_axis_2d = project_point_world2img(x_axis, intrinsic_matrix, T_world2cam, is_mujoco_camera=is_mujoco_camera)
    y_axis_2d = project_point_world2img(y_axis, intrinsic_matrix, T_world2cam, is_mujoco_camera=is_mujoco_camera)
    z_axis_2d = project_point_world2img(z_axis, intrinsic_matrix, T_world2cam, is_mujoco_camera=is_mujoco_camera)

    # Draw 3-axis for grasp
    cv2.line(image, center_2d, x_axis_2d, (0, 0, 255), 2) # Red for x
    cv2.line(image, center_2d, y_axis_2d, (0, 255, 0), 2) # Green for y
    cv2.line(image, center_2d, z_axis_2d, (255, 0, 0), 2) # Blue for z
    cv2.circle(image, center_2d, 6, (0, 255, 255), -1)    # Yellow center point

    # Add labels for x, y, z axes
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    cv2.putText(image, 'x', x_axis_2d, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
    cv2.putText(image, 'y', y_axis_2d, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
    cv2.putText(image, 'z', z_axis_2d, font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)
    return image

def visualize_point_cloud(pcd, title="Point Cloud Visualization", point_size=1, color='b', view=None):
    """
    Visualize a point cloud.
    """
    # Ensure the point cloud is in (N, 3) format
    if len(pcd.shape) == 3:  # If shape is (H, W, 3)
        pcd = pcd.reshape(-1, 3)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=point_size, c=color, alpha=0.8)

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set equal aspect ratio for better visualization
    max_range = np.array([pcd[:, 0].max() - pcd[:, 0].min(),
                          pcd[:, 1].max() - pcd[:, 1].min(),
                          pcd[:, 2].max() - pcd[:, 2].min()]).max() / 2.0
    mid_x = (pcd[:, 0].max() + pcd[:, 0].min()) * 0.5
    mid_y = (pcd[:, 1].max() + pcd[:, 1].min()) * 0.5
    mid_z = (pcd[:, 2].max() + pcd[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if view is not None and len(view) == 2:
        ax.view_init(elev=view[0], azim=view[1])
    # Show the plot
    plt.show()


## =========== Robosuite ===========

def get_camera_trans(sim_data, cam_name):
    """
    Get the camera transformation matrix in Robosuite
    """
    cam_pos = sim_data.get_camera_xpos(cam_name)
    cam_rot = sim_data.get_camera_xmat(cam_name).reshape(3, 3)
    T_cam2world = np.eye(4)
    T_cam2world[:3, :3] = cam_rot
    T_cam2world[:3, 3] = cam_pos
    return T_cam2world

def save_scene(save_dir, rgb, points, mask, camera_name: str, save_counter: int):
    """
    Saves the RGB image, point cloud, and segmentation mask.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save RGB image
    rgb_path = os.path.join(save_dir, f"{camera_name}_rgb_{save_counter}.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    # Save segmentation mask
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, max value: {mask.max()}")
    seg_path = os.path.join(save_dir, f"{camera_name}_mask_{save_counter}.png")
    cv2.imwrite(seg_path, (mask * (255 / mask.max())).astype(np.uint8))
    
    # Save point cloud as PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    points_path = os.path.join(save_dir, f"{camera_name}_points_{save_counter}.ply")
    o3d.io.write_point_cloud(points_path, pcd)
    logging.info(f"Saved RGB image, point cloud, and segmentation mask to {save_dir}")

def get_object_pose(sim_data, sim_name):
    """
    Get the object pose transformation matrix in Robosuite.
    """
    sim_name = sim_name + "_main"
    pos = sim_data.get_body_xpos(sim_name)
    quat = sim_data.get_body_xquat(sim_name)  # [w, x, y, z] format
    T_obj2world = np.eye(4)
    # Convert quaternion to rotation matrix using scipy
    T_obj2world[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()  # scipy expects [x, y, z, w]
    T_obj2world[:3, 3] = pos
    return T_obj2world


## =========== Simulation Utils ===========

def get_sim_info(sim_info_path=None):
    """
    Get the simulation information from the specified YAML file.
    """
    if sim_info_path is None:
        sim_info_path = os.path.join(file_dir, '../configs/sim_info.yaml')
    assert os.path.exists(sim_info_path), f'sim info file does not exist ({sim_info_path})'
    with open(sim_info_path, 'r') as f:
        sim_info = yaml.load(f, Loader=yaml.FullLoader)
    return sim_info

def process_sim_info(sim_info, scene_name='default'):
    """
    Process simulation information.
    """
    objects = sim_info['objects']
    interactions = sim_info['interactions']
    scenes = sim_info['scenes']
    # target scene
    scene = next((s for s in scenes if s['name'] == scene_name), None)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' not found in YAML.")
    scene_ids = set(scene['object_ids'])

    sim_name_dict = {}
    mask_name_dict = {}
    objects_list = []
    # target objects
    for obj in objects:
        if obj['id'] in scene_ids:
            sim_name_dict[obj['id']] = obj['sim_name']
            mask_name_dict[obj['id']] = obj['mask_name']
            objects_list.append((obj['id'], obj['vlm_return_name']))
    # target interactions
    interactions_list = []
    constraints_list = []
    for inter in interactions:
        if len(inter) == 3:
            source_id, target_id, relation = inter
            constraint = []
        elif len(inter) == 4:
            source_id, target_id, relation, constraint = inter
            constraint = [tuple(c) for c in constraint]
        else:
            raise ValueError(f"Invalid interaction format: {inter}")
        if source_id in scene_ids and target_id in scene_ids:
            interactions_list.append((source_id, target_id, relation))
            constraints_list.append((source_id, target_id, constraint))
    return sim_name_dict, mask_name_dict, objects_list, interactions_list, constraints_list

def stabilize_object_pose(sim):
    """
    Stabilize the pose of an object in the simulation.
    """
    # Only for marker now
    joint_name = "marker_joint0"
    qpos = sim.data.get_joint_qpos(joint_name).copy()
    pos = qpos[:3]
    quat = R.from_euler("xyz", [0, 0, 0]).as_quat()  # [x, y, z, w]
    quat_wxyz = np.roll(quat, 1)                    # [w, x, y, z]
    qpos_new = np.concatenate([pos, quat_wxyz])
    sim.data.set_joint_qpos(joint_name, qpos_new)
    sim.data.set_joint_qvel(joint_name, np.zeros(6))


## ========== Transformation Utils ==========

def get_trans_mat(pos, quat):
    """
    Get the transformation matrix from position and quaternion.
    """
    rot = Rotation.from_quat(quat).as_matrix()
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot
    trans_mat[:3, 3] = pos
    return trans_mat

def pcd_cam2world(pcd_cam, T_cam2world, visualize=False):
    """
    Convert a point cloud from camera coordinates to world coordinates.
    """
    if len(pcd_cam.shape) == 3:  # If shape is (H, W, 3)
        H, W, _ = pcd_cam.shape
        pcd_homo = np.concatenate([pcd_cam.reshape(-1, 3), np.ones((H * W, 1))], axis=1)  # (N, 4)
        pcd_world = (T_cam2world @ pcd_homo.T).T[:, :3].reshape(H, W, 3)
    elif len(pcd_cam.shape) == 2:  # If shape is (N, 3)
        N, _ = pcd_cam.shape
        pcd_homo = np.concatenate([pcd_cam, np.ones((N, 1))], axis=1)  # (N, 4)
        pcd_world = (T_cam2world @ pcd_homo.T).T[:, :3]  # (N, 3)
    else:
        raise ValueError(f"Unsupported point cloud shape: {pcd_cam.shape}. Expected (H, W, 3) or (N, 3).")

    if visualize:
        visualize_point_cloud(pcd_world)
    return pcd_world

def get_orientation_delta(current_rot: np.ndarray, target_rot: np.ndarray):
    """
    Get the rotation vector that represents the difference between the current and target orientations.
    """
    delta_R = target_rot @ current_rot.T  # R_delta = R_target * R_current.T
    delta_rotvec = R.from_matrix(delta_R).as_rotvec()  # shape (3,)
    return delta_rotvec
