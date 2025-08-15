import numpy as np
from scipy.spatial.transform import Rotation as R
from tasc.utils import get_logger

logger = get_logger("tasc")

class PlaceOptimizer:
    def __init__(self, R_tgt: np.ndarray, R_src_init: np.ndarray, constraints: list):
        self.R_tgt = R_tgt
        self.R_src_init = R_src_init
        self.constraints = constraints

    def solve(self):
        if len(self.constraints) == 0:
            return np.eye(3)
        src_axes = []
        tgt_axes = []
        for i, j, sign in self.constraints:
            tgt_axis = sign * self.R_tgt[:, i]
            src_axis = self.R_src_init[:, j]
            tgt_axes.append(tgt_axis)
            src_axes.append(src_axis)
        # Wahba problem: find R s.t. R @ b ≈ a
        rot, rmsd = R.align_vectors(tgt_axes, src_axes)
        rot = rot.as_matrix()
        logger.info(f"Rotation matrix:\n{rot}, RMSD: {rmsd}")
        return rot


if __name__ == "__main__":
    # Example usage
    R1 = np.eye(3)  # Identity rotation for object 1
    R2 = R.from_euler('z', 45, degrees=True).as_matrix()  # Initial rotation for object 2
    constraints = [(0, 0, 1), (2, 2, -1)]  # R1.x aligns with R2.x, R1.z opposes R2.z

    optimizer = PlaceOptimizer(R1, R2, constraints)
    R_opt = optimizer.solve()
    print("Optimized Rotation Matrix:\n", R_opt)