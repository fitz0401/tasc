import os
import cv2
import numpy as np
np.float = float
import matplotlib.pyplot as plt
import open3d as o3d
from tasc.utils import get_logger, pcd_cam2world, project_point_world2img, farthest_point_sampling, Arrow3D
from tasc.goal_predictor.scene_graph import SceneGraph


logger = get_logger("tasc")
file_path = os.path.dirname(os.path.abspath(__file__))


class ObjectDetector:
    '''
    Object detection and grasp planning.
    
    Processes 3D point clouds and RGB images to detect objects, compute grasp poses,
    and generate oriented bounding boxes for objects.
    All coordinates are maintained in the global robot base frame.
    '''
    def __init__(self, config, pcd, T_cam2world, intrinsic_matrix, image):
        '''
        Initialize object detector with camera configuration and image input
        '''
        self.config = config
        self.image = image  # (H, W, 3) uint8
        self.T_cam2world = T_cam2world  # (H, W, 3)
        self.T_world2cam = np.linalg.inv(T_cam2world)
        self.intrinsic_matrix = intrinsic_matrix
        apply_transform = self.config['vision_module'].get('apply_pcd_transform', True)
        if apply_transform:
            self.pcd_global = pcd_cam2world(pcd, T_cam2world)
        else:
            self.pcd_global = pcd
        x_min, x_max, y_min, y_max, z_min, z_max = self.config['vision_module']['table_region']
        table_mask = (
            (self.pcd_global[:, :, 0] >= x_min) & (self.pcd_global[:, :, 0] <= x_max) &
            (self.pcd_global[:, :, 1] >= y_min) & (self.pcd_global[:, :, 1] <= y_max) &
            (self.pcd_global[:, :, 2] >= z_min) & (self.pcd_global[:, :, 2] <= z_max)
        )
        self.valid_mask = table_mask & np.isfinite(self.pcd_global).all(axis=-1)
        self.task_pcd_global = self.pcd_global[self.valid_mask].reshape(-1, 3)
        self.valid_color = self.image[self.valid_mask].reshape(-1, 3).astype(np.float32) / 255.0     

    def update_visual_info(self, scene_graph: SceneGraph, masks, refine_radius=25):
        '''
        Extract and update visual information for detected objects in the scene graph.
        Processes segmentation masks to compute object centers, extract point cloud regions,
        and update scene graph nodes with visual properties.
        Args:
            scene_graph (SceneGraph): Scene graph containing object nodes to update
            masks (list): List of boolean masks for each object in the scene
            refine_radius (int, optional): Radius for refining center point calculation. Defaults to 25.
        Updates:
            - Object pixel centers in image coordinates
            - Point cloud points belonging to each object
            - 3D center points in global coordinates
            - Image masks for each object node
        '''
        for obj_id, obj_node in scene_graph.nodes_dict.items():
            mask = masks[obj_id]
            mask &= self.valid_mask  # apply valid mask to the object mask
            # get center point from global pcd around the mask center
            y_idx, x_idx = np.where(mask)
            cy = int(np.mean(y_idx))
            cx = int(np.mean(x_idx))
            y_start, y_end = max(0, cy - refine_radius), min(mask.shape[0], cy + refine_radius)
            x_start, x_end = max(0, cx - refine_radius), min(mask.shape[1], cx + refine_radius)
            center_mask = mask[y_start:y_end, x_start:x_end]
            center_pts = self.pcd_global[y_start:y_end, x_start:x_end][center_mask]
            center_pt = np.mean(center_pts, axis=0)

            # get region mask (bounding box) for the object's points to make it contain the table
            y_idx, x_idx = np.where(mask)
            ymin, ymax = max(0, y_idx.min() - refine_radius), min(mask.shape[0], y_idx.max() + refine_radius)
            xmin, xmax = max(0, x_idx.min() - refine_radius), min(mask.shape[1], x_idx.max() + refine_radius)
            region_mask = np.zeros_like(mask, dtype=bool)
            region_mask[ymin:ymax+1, xmin:xmax+1] = True

            # update the object node
            obj_node.update_visual_info(
                image_mask=mask,
                pixel_center=[cx, cy],
                pcd_points=self.pcd_global[mask].astype(np.float32),
                center_point=center_pt[:3],
            )
            logger.info(f"Object {obj_id} center point: {center_pt[:3]}")
    
    def update_grasp_poses(self, scene_graph: SceneGraph, visualize=False, save_path=None):
        '''
        Generate and update grasp poses for objects using AnyGrasp neural network.
        Runs the AnyGrasp model on the scene point cloud to generate grasp poses, 
        then assigns the best grasps to each object based on spatial proximity.
        Args:
            scene_graph (SceneGraph): Scene graph containing objects to generate grasps for
            visualize (bool, optional): Whether to display 3D grasp visualization. Defaults to False.
            save_path (str, optional): Path to save grasp visualization image. Defaults to None.
        Updates:
            - Grasp poses for each object node in the scene graph
            - Top-N grasp candidates selected using farthest point sampling
        '''
        # Load the AnyGrasp model
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from tasc.vision_module.gsnet import AnyGrasp
        import argparse
        import multiprocessing as mp
        self.anygrasp_config = self.config["grasp_planner"]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(base_dir, self.anygrasp_config["checkpoint_path"])
        cfgs = argparse.Namespace(
            checkpoint_path=ckpt_path,
            max_gripper_width=self.anygrasp_config["max_gripper_width"],
            gripper_height=self.anygrasp_config["gripper_height"],
            top_down_grasp=self.anygrasp_config["top_down_grasp"],
            debug=self.anygrasp_config["debug"]
        )
        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()

        # Align with anygrasp coordinate system
        R_flip_z = np.diag([1, -1, -1])
        pcd_filp_z = self.task_pcd_global @ R_flip_z.T
        gg, cloud = self.anygrasp.get_grasp(
            pcd_filp_z.astype(np.float32),
            self.valid_color,
            lims=self.anygrasp_config['bounds'], 
            apply_object_mask=True, 
            dense_grasp=False, 
            collision_detection=True
        )
        gg = gg.nms().sort_by_score()
        # show grasp poses globally
        trans_mat = np.diag([1, -1, -1, 1])
        cloud.transform(trans_mat)
        if visualize:
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            def visualize_grasps(grippers, cloud, save_path=None):
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                for gripper in grippers:
                    vis.add_geometry(gripper)
                vis.add_geometry(cloud)
                vis.run()
                if save_path is not None:
                    vis.capture_screen_image(save_path)
                    print(f"Grasp visualization saved to: {save_path}")
                vis.destroy_window()
            p = mp.Process(target=visualize_grasps, args=(grippers, cloud, save_path))
            p.start()
            p.join()
        
        R_anygrasp2robosuite = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        all_grasp_poses = {obj_id: [] for obj_id in scene_graph.nodes_dict.keys()}
        for grasp in gg:
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] =  R_flip_z @ grasp.rotation_matrix.copy() @ R_anygrasp2robosuite
            pose_matrix[:3, 3] = R_flip_z @ grasp.translation.copy()
            if pose_matrix[2, 3] < self.anygrasp_config['table_height'] + self.anygrasp_config['z_limit_above_table']:
                continue
            pose_matrix[:3, 3] += self.anygrasp_config['z_offset'] * pose_matrix[:3, 2]  # Move along the z-axis
            grasp_pos = pose_matrix[:3, 3]
            u, v = project_point_world2img(grasp_pos, self.intrinsic_matrix, self.T_world2cam, 
                                           is_mujoco_camera=self.config['vision_module'].get('is_mujoco_camera', True))
            for obj_id, obj_node in scene_graph.nodes_dict.items():
                dilated_mask = cv2.dilate(obj_node.image_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
                if dilated_mask[v, u]:
                    all_grasp_poses[obj_id].append(pose_matrix)
                    break
        top_n = 5
        for obj_id, obj_node in scene_graph.nodes_dict.items():
            obj_grasp_poses = all_grasp_poses[obj_id]
            selected_grasps = obj_grasp_poses
            if len(obj_grasp_poses) > top_n:
                grasp_points = np.array([pose[:3, 3] for pose in obj_grasp_poses])
                _, selected_idx = farthest_point_sampling(grasp_points, top_n, return_idx=True)
                selected_grasps = [obj_grasp_poses[i] for i in selected_idx]
            obj_node.update_grasp_poses(selected_grasps)

    def update_grasp_poses_offline(self, scene_graph: SceneGraph, sim_name_dict: dict, sim):
        '''
        Update grasp poses using predefined grasp sites from simulation environment.
        '''
        all_site_names = sim.model.site_names
        for obj_id, obj_name in sim_name_dict.items():
            grasp_poses = []
            matched_site_names = [s for s in all_site_names if s.startswith(f"{obj_name}_grasp_")]
            for site_name in matched_site_names:
                site_id = sim.model.site_name2id(site_name)
                pos = sim.data.site_xpos[site_id]
                rot = sim.data.site_xmat[site_id].reshape(3, 3)
                pose = np.eye(4)
                pose[:3, :3] = rot
                pose[:3, 3] = pos
                grasp_poses.append(pose)
            obj_node = scene_graph.get_node_by_id(obj_id)
            obj_node.update_grasp_poses(grasp_poses)
    
    def update_pcd_obb(self, scene_graph, visualize=False, save_path=None, return_images=False):
        '''
        Compute and update oriented bounding boxes (OBB) for active objects in the scene.
        Args:
            scene_graph (SceneGraph): Scene graph containing objects to process
            visualize (bool, optional): Whether to display OBB visualizations. Defaults to False.
            save_path (str, optional): Directory path to save OBB visualization images. Defaults to None.
            return_images (bool, optional): Whether to return visualization images as numpy arrays. Defaults to False.
        Returns:
            dict or None: If return_images=True, returns dictionary mapping object_id to image arrays.
                         Otherwise returns None.
        Updates:
            - Object pose matrices (pcd_pose) in the scene graph nodes
            - OBB parameters (center, axes, extent) for spatial reasoning
        '''
        obb_images = {}  # Store images by object_id if return_images is True
        for obj_id, obj_node in scene_graph.nodes_dict.items():
            if obj_node.is_active or obj_node.is_grasped:
                # Step 1: calculate OBB
                center, axes, extent = self.compute_obb_with_fixed_z(obj_node.pcd_points)
                T = np.eye(4)
                T[:3, :3] = axes
                T[:3, 3] = center
                # Step 2: visualize and/or save with matplotlib
                fig_name = f"obb_a_{obj_id}.png" if obj_node.is_grasped else f"obb_b_{obj_id}.png"
                if visualize or save_path is not None or return_images:
                    image_patch=(self.image[obj_node.image_mask].astype(np.float32)) / 255.0
                    save_file_path = os.path.join(save_path, fig_name) if save_path is not None else None
                    image_data = self.plot_pcd_and_obb(obj_node.pcd_points, image_patch, center, axes, extent, 
                                          save_path=save_file_path, visualize=visualize, return_image=return_images)
                    if return_images and image_data is not None:
                        obb_images[obj_id] = image_data
                obj_node.update_pcd_pose(T)
        return obb_images if return_images else None
    
    @staticmethod
    def compute_obb_with_fixed_z(pcd_points, z_axis=np.array([0, 0, 1])):
        '''
        Compute oriented bounding box with fixed vertical Z-axis for table-top objects.
        Performs 2D PCA on the XY projection of point cloud to determine principal
        axes while keeping Z-axis fixed as the vertical direction.
        Args:
            pcd_points (np.ndarray): Point cloud points in shape (N, 3)
            z_axis (np.ndarray, optional): Fixed Z-axis direction. Defaults to [0, 0, 1].
        Returns:
            tuple: (center_world, axes, extent) where:
                - center_world (np.ndarray): OBB center in world coordinates (3,)
                - axes (np.ndarray): OBB orientation matrix (3, 3) with columns as axes
                - extent (np.ndarray): Half-extents along each axis (3,)
        '''
        # Center the point cloud
        center = np.mean(pcd_points, axis=0)
        pcd_centered = pcd_points - center
        # Project onto XY plane
        pcd_proj = pcd_centered[:, :2]  # (N, 2)
        # 2D PCA on XY
        cov = np.cov(pcd_proj.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        xy_axes = eigvecs[:, ::-1]  # sort by descending eigenvalue
        # Reconstruct 3D axes: [X, Y, Z]
        x_axis = np.array([xy_axes[0, 0], xy_axes[1, 0], 0])
        y_axis = np.array([xy_axes[0, 1], xy_axes[1, 1], 0])
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.array(z_axis)
        axes = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3, 3)
        # Transform all points to this frame to compute extent
        pcd_local = pcd_centered @ axes
        min_p = np.min(pcd_local, axis=0)
        max_p = np.max(pcd_local, axis=0)
        extent = (max_p - min_p) / 2.0
        center_local = (min_p + max_p) / 2.0
        center_world = axes @ center_local + center
        return center_world, axes, extent

    def plot_pcd_and_obb(self, points, color, center, axes, extent, save_path=None, visualize=True, return_image=False):
        '''
        Generate 3D visualization of point cloud with oriented bounding box coordinate frame.
        '''
        # Convert points to object frame: x_obj = R^T * (x - center)
        points_obj = (points - center) @ axes  # shape (N, 3)

        # Convert OBB vertices to object frame
        fig = plt.figure(figsize=(10, 5))
        axis_colors = ['r', 'g', 'b']
        axis_labels = ['X', 'Y', 'Z']

        # Normalize axis scaling
        max_range = (points_obj.max(axis=0) - points_obj.min(axis=0)).max() * 1.3
        mid = (points_obj.max(axis=0) + points_obj.min(axis=0)) / 2.0
        arrow_length = np.linalg.norm(points_obj - np.zeros(3), axis=1).max() * 2.4
        
        # Draw three views, each focusing on one axis
        for i in range(3):
            axis_dir = np.eye(3)[:, i]  # x, y, z unit vector
            arrow_start = -axis_dir * arrow_length / 2.0
            arrow_end = axis_dir * arrow_length / 2.0
            label_pos = arrow_end + axis_dir * 0.03
            if i == 0:
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                ax.view_init(elev=90, azim=-90)
            elif i == 2:
                ax = fig.add_subplot(1, 2, 2, projection='3d')
                ax.view_init(elev=0, azim=-90)
            # Draw axis
            arrow = Arrow3D([arrow_start[0], arrow_end[0]], 
                            [arrow_start[1], arrow_end[1]], 
                            [arrow_start[2], arrow_end[2]], 
                            mutation_scale=40, lw=4, arrowstyle="->", color=axis_colors[i], zorder=1)
            ax.add_artist(arrow)
            ax.text(label_pos[0], label_pos[1], label_pos[2],
                    axis_labels[i] + "_axis", color=axis_colors[i], 
                    fontsize=16, weight="bold", zorder=3,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            if i == 1 or i == 2:
                ax.scatter(points_obj[:, 0], points_obj[:, 1], points_obj[:, 2],
                           c=color, s=3, alpha=0.25, zorder=2)

            # Set limits and labels
            ax.computed_zorder = False
            ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
            ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
            ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)
            ax.set_xlabel("X_obj")
            ax.set_ylabel("Y_obj")
            ax.set_zlabel("Z_obj")
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        plt.tight_layout()

        # Get image data if requested
        image_data = None
        if return_image:
            # Simple method: save to memory buffer and read back
            from io import BytesIO
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            from PIL import Image as PILImage
            pil_img = PILImage.open(buffer)
            image_data = np.array(pil_img)
            if image_data.shape[-1] == 4:  # RGBA to RGB
                image_data = image_data[:, :, :3]
            buffer.close()
        # Save if needed
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        # Show if needed
        if visualize:
            plt.show()
        else:
            plt.close(fig)
        return image_data