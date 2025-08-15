import numpy as np
from tasc.utils import get_logger, get_orientation_delta
from tasc.goal_predictor.scene_graph import SceneGraph

logger = get_logger("tasc")
OPEN = -1
CLOSE = 1


class SharedController:
    def __init__(self):
        self.reset()
        self._pos_tol = 0.008  # Position tolerance in meters
        self._rot_tol = 0.05   # Rotation tolerance in radians
        self._max_pos_step = 0.5  # Maximum position step in meters
        self._max_rot_step_rad = 0.5 # Maximum rotation step in radians
        self._K_pos = 80  # Proportional gain for position control
        self._K_rot = 20  # Proportional gain for rotation control
        self._alpha = 3.0  # Proportional gain (max_gain = (1 + alpha) * K)
        self._scale_pos = 40.0  # Scale factor for pos gain
        self._scale_rot = 2.5   # Scale factor for rot gain
        self._blending_factor = 0.7  # Blending factor for gripper action
        self._pre_grasping_offset = 0.07  # Pre-grasping offset in meters
        self.is_state_change = False  # Flag to indicate if the state has changed

    def reset(self):
        self.T_tgt = None
        self.id_tgt = None
        self.assistance_mode = 'none'
        self.prev_assistance_mode = 'none'
        self.curr_gripper_state = OPEN     # real gripper state
        self.prev_gripper_action = OPEN    # last gripper action (hold state)
    
    def set_parameters(self, pos_tol=None, rot_tol=None, max_pos_step=None, max_rot_step_rad=None, K_pos=None, K_rot=None):
        if pos_tol is not None:
            self._pos_tol = pos_tol
        if rot_tol is not None:
            self._rot_tol = rot_tol
        if max_pos_step is not None:
            self._max_pos_step = max_pos_step
        if max_rot_step_rad is not None:
            self._max_rot_step_rad = max_rot_step_rad
        if K_pos is not None:
            self._K_pos = K_pos
        if K_rot is not None:
            self._K_rot = K_rot

    def _set_target(self, matrix_tgt, id_tgt):
        if matrix_tgt is None:
            self.T_tgt = None
            self.id_tgt = None
            return
        assert matrix_tgt.shape in [(4, 4), (3, 3)], "Target matrix must be a 4x4 or 3x3 transformation matrix."
        assert isinstance(id_tgt, int)
        self.T_tgt = np.eye(4)
        self.id_tgt = id_tgt
        if matrix_tgt.shape == (4, 4):
            self.T_tgt = matrix_tgt
        elif matrix_tgt.shape == (3, 3):
            self.T_tgt[:3, :3] = matrix_tgt

    def manipulation_goal(self):
        return self.T_tgt, self.id_tgt
    
    def _calculate_gain(self, error_norm, type):
        """
        Calculate the gain based on the error norm.
        The gain is inversely proportional to the error norm.
        """
        if type == 'pos':
            return self._K_pos / (1 + self._alpha * np.exp(-self._scale_pos * error_norm))
        elif type == 'rot':
            return self._K_rot / (1 + self._alpha * np.exp(-self._scale_rot * error_norm))
        else:
            raise ValueError("Type must be 'pos' or 'rot'.")
        
    def _assist_rotation(self, eef_pose, goal_pose=None):
        """
        Assist the end-effector rotation towards the goal pose.
        """
        if goal_pose is None:
            goal_pose = self.T_tgt
        assert goal_pose is not None, "Goal pose must be set before assisting rotation."
        delta_rotvec = get_orientation_delta(current_rot=eef_pose[:3, :3], target_rot=goal_pose[:3, :3])
        rot_mag = np.linalg.norm(delta_rotvec)
        if rot_mag < self._rot_tol:
            return np.zeros(3), True
        gain = self._calculate_gain(rot_mag, type='rot')
        delta_rotvec = gain * delta_rotvec
        rot_mag = np.linalg.norm(delta_rotvec)
        if rot_mag > self._max_rot_step_rad:
            delta_rotvec = delta_rotvec / rot_mag * self._max_rot_step_rad
        return delta_rotvec, False

    def _assist_movement(self, eef_pose, goal_pose=None):
        """
        Assist the end-effector movement towards the goal pose.
        """
        if goal_pose is None:
            goal_pose = self.T_tgt
        assert goal_pose is not None, "Goal pose must be set before assisting movement."
        delta_pos = goal_pose[:3, 3] - eef_pose[:3, 3]
        pos_mag = np.linalg.norm(delta_pos)
        if np.linalg.norm(delta_pos) < self._pos_tol:
            return np.zeros(3), True
        gain = self._calculate_gain(pos_mag, type='pos')
        delta_pos = gain * delta_pos
        pos_mag = np.linalg.norm(delta_pos)
        if pos_mag > self._max_pos_step:
            delta_pos = delta_pos / pos_mag * self._max_pos_step
        return delta_pos, False
    
    def _assist_pose(self, eef_pose, goal_pose=None):
        """
        Assist the end-effector pose towards the goal pose.
        """
        if goal_pose is None:
            goal_pose = self.T_tgt
        assert goal_pose is not None, "Goal pose must be set before assisting pose."
        delta_rotvec, rot_done = self._assist_rotation(eef_pose, goal_pose)
        delta_pos, pos_done = self._assist_movement(eef_pose, goal_pose)
        if rot_done and pos_done:
            return np.zeros(6), True
        return np.concatenate([delta_pos, delta_rotvec]), False
    
    def _assist_grasping(self, eef_pose, is_first_entry: bool):
        """
        Assist the gripper to grasp an object.
        The gripper will move to a pre-grasping pose first, then assist the gripper to grasp the object.
        """
        if is_first_entry:
            self.pre_grasping_log = False
            self.pre_grasping_pose = self.T_tgt.copy()
            self.pre_grasping_pose[:3, 3] -= self._pre_grasping_offset * self.pre_grasping_pose[:3, 2]  # Move along the z-axis
        if not self.pre_grasping_log:
            delta_pose, self.pre_grasping_log = self._assist_pose(eef_pose, self.pre_grasping_pose)
            return delta_pose, False
        else:
            return self._assist_pose(eef_pose)
    
    def state_machine(self, gripper_acion):
        """
        Change the assistance mode based on the gripper action.
        gripper_acion is a hold state, not a instant action. Once the user push close/open button, the state will be changed.
        """
        assistance_mode = 'none'
        self.prev_gripper_action = gripper_acion
        ## 1. Grasping assistance
        if self.curr_gripper_state == OPEN and gripper_acion == OPEN:
            assistance_mode = 'grasping_assistance'
        ## 2. Auto grasping
        elif self.curr_gripper_state == OPEN and gripper_acion == CLOSE:
            assistance_mode = 'auto_grasping'
        ## 3. Reset
        elif self.curr_gripper_state == CLOSE and gripper_acion == OPEN:
            assistance_mode = 'reset'
        ## 4. Manipulation assistance
        elif self.curr_gripper_state == CLOSE and gripper_acion == CLOSE:
            assistance_mode = 'manipulation_assistance'
        is_state_changed = (self.prev_gripper_action != gripper_acion) or (self.prev_assistance_mode != assistance_mode)
        return assistance_mode, is_state_changed
    
    def _select_manipulation_goal(self, eef_pose, scene_graph: SceneGraph):
        """
        Select the manipulation goal based on the current assistance mode.
        """
        if self.is_state_change and self.assistance_mode == 'auto_grasping':
            nearest_obj_node_id, nearest_obj_node = scene_graph.get_nearest_node(eef_pose)
            manipulation_goal = nearest_obj_node.get_best_grasp_pose(eef_pose, calculate_equivalent=True)
            self._set_target(manipulation_goal, nearest_obj_node_id)
        elif self.assistance_mode == 'grasping_assistance':
            goal_obj_node_id, goal_obj_node = scene_graph.get_goal_node()
            manipulation_goal = goal_obj_node.get_best_grasp_pose(eef_pose, calculate_equivalent=True)
            self._set_target(manipulation_goal, goal_obj_node_id)
        elif self.assistance_mode == 'manipulation_assistance':
            goal_node_id, _ = scene_graph.get_goal_node()
            grasped_node_id, _ = scene_graph.get_grasped_node()
            affordance, manipulation_goal = scene_graph.get_relations(grasped_node_id, goal_node_id)
            self._set_target(manipulation_goal, goal_node_id)
    
    def assist_action(self, eef_pose, user_command, gripper_acion, scene_graph: SceneGraph):
        """
        Compute assistance actions for shared control policy. 
        Args:
            eef_pose (np.ndarray): Current end-effector pose as 4x4 transformation matrix
            user_command (np.ndarray): User's 6D command vector [x,y,z,rx,ry,rz]
            gripper_acion (int): Gripper action state (-1 for close, 1 for open)
            scene_graph (SceneGraph): Current scene understanding with objects and relationships
        Returns:
            tuple: (assistance_action, gripper_command) where:
                - assistance_action (np.ndarray): 6D assistance vector to add to user command
                - gripper_command (int): Modified gripper command incorporating autonomous actions
        """
        assert eef_pose.shape == (4, 4), "EEF pose must be a 4x4 transformation matrix."
        assert user_command.shape == (6,), "User command must be a 6D vector."
        assert gripper_acion in [-1, 1], "Gripper action must be -1 (close), or 1 (open)."
        
        ## Change assistance mode based on gripper action
        self.is_state_change = False
        self.prev_assistance_mode = self.assistance_mode
        self.assistance_mode, self.is_state_change = self.state_machine(gripper_acion)

        ## Update scene graph state
        if self.is_state_change and self.assistance_mode in ('reset', 'manipulation_assistance'):
            scene_graph.update_stage(self.assistance_mode, eef_pose)

        ## Get manipulation goal
        self._select_manipulation_goal(eef_pose, scene_graph)

        ## Assist action
        # 0. Reset
        if self.assistance_mode == 'reset':
            self.reset()
            return np.zeros(6), gripper_acion
        if self.T_tgt is None:
            return np.zeros(6), gripper_acion
        # 1. Auto grasping: first entry or conducting auto grasping process
        # The manipulation goal is selected during the last grapsing assistance stage.
        elif self.assistance_mode == 'auto_grasping':
            delta_pose, done = self._assist_grasping(eef_pose, self.is_state_change)
            if done:
                _, closest_obj_node = scene_graph.get_nearest_node(eef_pose)
                closest_obj_node.is_grasped = True
                self.curr_gripper_state = CLOSE
                logger.info(f"Grasped object: {closest_obj_node.name}")
                return np.zeros(6), CLOSE
            else:
                self.curr_gripper_state = OPEN
                return delta_pose, OPEN
        # 2. Assistance for grasping or manipulation
        elif self.assistance_mode in ('grasping_assistance', 'manipulation_assistance'):
            self.curr_gripper_state = gripper_acion
            if not np.allclose(user_command, 0.0, atol=1e-4):
                delta_rotvec, _ = self._assist_rotation(eef_pose)
                return np.concatenate([np.zeros(3), delta_rotvec]), gripper_acion
        return np.zeros(6), gripper_acion