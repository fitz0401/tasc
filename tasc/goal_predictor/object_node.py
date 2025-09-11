import numpy as np
from tasc.goal_predictor.prediction_utils import HuberCost, RotationMatrixDistance
from tasc.utils import get_logger, normalize


logger = get_logger("tasc")


class ObjectNode:
    def __init__(self, name):
        self.name = name
        self.has_visual_info = False
        self.image_mask = None
        self.pixel_center = None
        self.center_point = None
        self.pcd_points=None
        self.pcd_pose = np.eye(4)
        self.tracking_pose = None
        self.grasp_poses = []
        self.equivalent_grasp_poses = []
        self.keypoints = []
        self.relations = {}     # {other_id: [relation_type, target_pose]}
        self.is_active = True
        self.is_grasped = False
    
    ## ======= Perception Info =======
    def update_visual_info(self, image_mask, pixel_center, pcd_points, center_point):
        self.image_mask = image_mask
        self.pixel_center = pixel_center
        self.pcd_points = pcd_points
        self.center_point = center_point
        self.has_visual_info = True
    
    def update_pcd_pose(self, pcd_pose):
        self.pcd_pose = pcd_pose

    def update_keypoints(self, keypoints):
        self.keypoints = keypoints
    
    def add_relation(self, other_id, relation_type):
        if other_id not in self.relations:
            self.relations[other_id] = [relation_type, None]

    def get_obj_pose(self):
        return self.pcd_pose

    ## ======= Prediction =======
    def init_prediction_policy(self):
        self.prediction_policy = []
        self.min_val_idx = 0
        self.prediction_policy.append(HuberCost(self.center_point))
    
    def update_prediction_policy(self, eef_pose, eef_pose_after_action):
        for policy in self.prediction_policy:
            policy.update(eef_pose, eef_pose_after_action)
        values = [one_target_policy.get_value() for one_target_policy in self.prediction_policy]
        self.min_val_idx = np.argmin(values)

    def get_value(self):
        return self.prediction_policy[self.min_val_idx].get_value()

    def get_qvalue(self):
        return self.prediction_policy[self.min_val_idx].get_qvalue()

    ## ====== Grasping ======
    def get_grasp_poses(self):
        return self.grasp_poses
    
    def get_equivalent_grasp_poses(self):
        return self.equivalent_grasp_poses
    
    def _cal_equivalent_grasp_poses(self):
        R_z_180 = np.array([
            [-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0,  1]
        ])
        equivalent_grasp_poses = [pose.copy() for pose in self.grasp_poses]
        for pose in equivalent_grasp_poses:
            pose[:3, :3] = pose[:3, :3] @ R_z_180
        return equivalent_grasp_poses
    
    def update_grasp_poses(self, grasp_poses):
        self.grasp_poses = grasp_poses
        self.equivalent_grasp_poses = self._cal_equivalent_grasp_poses()
        # Refine direction of grasp poses
        self.grasp_poses = self._refine_grasp_orientations()
        self.equivalent_grasp_poses = self._cal_equivalent_grasp_poses()

    def _refine_grasp_orientations(self):
        best_poses = []
        candidate_pairs = list(zip(self.grasp_poses, self.equivalent_grasp_poses))
        # greedily select the best grasp pose based on the current mean direction
        current_mean = np.mean([pose[:3, 0] for pose in self.grasp_poses], axis=0)
        current_mean = normalize(current_mean)
        for original, equivalent in candidate_pairs:
            x1 = normalize(original[:3, 0])
            x2 = normalize(equivalent[:3, 0])
            score1 = np.dot(x1, current_mean)
            score2 = np.dot(x2, current_mean)
            best_poses.append(original if score1 >= score2 else equivalent)
        return best_poses

    def get_best_grasp_pose(self, eef_pose, calculate_equivalent=False):
        if len(self.grasp_poses) == 0 or len(self.equivalent_grasp_poses) == 0:
            return None
        dists = [np.linalg.norm(pose[:3, 3] - eef_pose[:3, 3]) for pose in self.grasp_poses]
        idx_min = int(np.argmin(dists))
        if not calculate_equivalent:
            return self.grasp_poses[idx_min]
        
        candidate_poses = [self.grasp_poses[idx_min], self.equivalent_grasp_poses[idx_min]]
        best_pose = None
        min_angle_diff = float('inf')
        for candidate_pose in candidate_poses:
            angle_diff = RotationMatrixDistance(candidate_pose, eef_pose)
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_pose = candidate_pose
        return best_pose

    def __repr__(self):
        return f"Node(name={self.name}, relations={self.relations}), has_visual_info={self.has_visual_info}, is_active={self.is_active})"