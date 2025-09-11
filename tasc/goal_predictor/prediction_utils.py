import numpy as np
import transformations as transmethods


def RotationMatrixDistance(pose1, pose2):
    """
    Compute the distance between two rotation matrices.
    """
    quat1 = transmethods.quaternion_from_matrix(pose1)
    quat2 = transmethods.quaternion_from_matrix(pose2)
    q1 = quat1 / np.linalg.norm(quat1)
    q2 = quat2 / np.linalg.norm(quat2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(abs(dot))

def ApplyTwistToTransform(twist, transform, time=1.):
    """
    Apply a twist (linear and angular velocity) to a transform over a given time interval.
    """
    twist = np.asarray(twist).reshape(-1)
    assert transform.shape == (4, 4), "Transform must be a 4x4 matrix"
    assert twist.shape == (6,), "Twist must be a 6D vector"
    
    new_transform = transform.copy()
    new_transform[0:3,3] += time * twist[0:3]
    angular_velocity = twist[3:]
    angular_velocity_norm = np.linalg.norm(angular_velocity)
    if angular_velocity_norm > 1e-3:
        angle = time*angular_velocity_norm
        axis = angular_velocity/angular_velocity_norm
        new_transform[0:3,0:3] = np.dot(transmethods.rotation_matrix(angle, axis), new_transform)[0:3,0:3]
    return new_transform

def ApplyAngularVelocityToQuaternion(angular_velocity, quat, time=1.):
    """
    Apply angular velocity to a quaternion over a given time interval.
    """
    angular_velocity_norm = np.linalg.norm(angular_velocity)
    angle = time*angular_velocity_norm
    axis = angular_velocity/angular_velocity_norm
    #angle axis to quaternion formula
    quat_from_velocity = np.append(np.sin(angle/2.)*axis, np.cos(angle/2.))
    return transmethods.quaternion_multiply(quat_from_velocity, quat)

def pose_to_mat(pose):
    """
    Convert a Pose msg to a 4x4 numpy matrix
    """
    mat = np.zeros((4, 4))
    mat[3, 3] = 1
    mat[0:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    mat[0:3, 0:3] = np.array(transmethods.quaternion_matrix(q))[0:3, 0:3] 
    return mat


class HuberCost():
  """
  Huber Assistance Policy \n
  Args: 
    pose: matrix of goal
  """
  def __init__(self, goal_pos):
    self.goal_pos = goal_pos
    self._set_parameters()

  def update(self, eef_pose, robot_state_after_action):
    self.dist_translation = np.linalg.norm(eef_pose[0:3,3] - self.goal_pos)
    self.dist_translation_aftertrans = np.linalg.norm(robot_state_after_action[0:3,3] - self.goal_pos) 

  # C(x, u)
  def get_cost(self, dist_translation=None):
    if dist_translation is None:
      dist_translation = self.dist_translation

    if dist_translation > self.TRANSLATION_DELTA_SWITCH:
      return self.ACTION_APPLY_TIME * (self.TRANSLATION_LINEAR_COST_MULT_TOTAL)
    else:
      return self.ACTION_APPLY_TIME * (self.TRANSLATION_QUADRATIC_COST_MULTPLIER * dist_translation + self.TRANSLATION_CONSTANT_ADD)

  # V(x; g)
  def get_value(self, dist_translation=None):
    if dist_translation is None:
      dist_translation = self.dist_translation

    if dist_translation <= self.TRANSLATION_DELTA_SWITCH:
      return self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF * dist_translation*dist_translation + self.TRANSLATION_CONSTANT_ADD*dist_translation
    else:
      return 0.6 *(self.TRANSLATION_LINEAR_COST_MULT_TOTAL * dist_translation - self.TRANSLATION_LINEAR_COST_SUBTRACT)
  
  # Q(x, u; g) = C(x, u) + V(x'; g)
  def get_qvalue(self):
    return self.get_cost() + self.get_value(self.dist_translation_aftertrans)
  
  def _set_parameters(self):
    self.ACTION_APPLY_TIME = 0.05
    self.TRANSLATION_LINEAR_MULTIPLIER = 1.5
    self.TRANSLATION_DELTA_SWITCH = 0.10
    self.TRANSLATION_CONSTANT_ADD = 0.1
    self.TRANSLATION_LINEAR_COST_MULT_TOTAL = self.TRANSLATION_LINEAR_MULTIPLIER + self.TRANSLATION_CONSTANT_ADD
    self.TRANSLATION_LINEAR_COST_SUBTRACT = self.TRANSLATION_LINEAR_MULTIPLIER * self.TRANSLATION_DELTA_SWITCH * 0.5
    self.TRANSLATION_QUADRATIC_COST_MULTPLIER = self.TRANSLATION_LINEAR_MULTIPLIER / self.TRANSLATION_DELTA_SWITCH
    self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF = 0.5 * self.TRANSLATION_QUADRATIC_COST_MULTPLIER