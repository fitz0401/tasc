import time
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


class ResultAnalyzer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.frame_records = []

    def start_timer(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.frame_records = []

    def record_frame(self, 
                     count: int, 
                     prob_distribution_with_ids: dict, 
                     assistance_mode: str, 
                     assistance_goal: int,
                     is_assisted: bool,
                     user_action: np.ndarray = None,
                     assist_action: np.ndarray = None,
                     eef_pose: np.ndarray = None,
                    ):
        if self.start_time is None:
            return
        end_time = time.time()
        row = {
            "frame_id": count,
            "time_elapsed": end_time - self.start_time if self.start_time else None,
            "assistance_mode": assistance_mode,
            "assistance_goal": assistance_goal,
            "is_assisted": is_assisted,
            "user_action": json.dumps(user_action.tolist() if user_action is not None else None),
            "assist_action": json.dumps(assist_action.tolist() if assist_action is not None else None),
            "eef_pose": json.dumps(eef_pose.tolist() if eef_pose is not None else None),
            "prob_distribution": json.dumps(prob_distribution_with_ids)
        }
        
        self.frame_records.append(row)

    def summarize(self):
        """Summarize the recorded frames"""
        if not self.frame_records:
            print("No records to summarize.")
            return
        total_frames = len(self.frame_records)
        last_time = self.frame_records[-1].get("time_elapsed", 0.0) - self.start_time if self.start_time else 0.0
        
        print("=== Summary ===")
        print(f"Total time: {last_time:.2f} seconds")
        print(f"Total frames: {total_frames}")
        user_input_frames = 0
        rotation_input_frames = 0
        assisted_frames = 0
        for r in self.frame_records:
            user_action = np.array(r.get("user_action"))
            if np.any(user_action != 0):
                user_input_frames += 1
                if r.get("is_assisted") == True:
                    assisted_frames += 1
            if np.any(user_action[3:] != 0):  # rotation components
                rotation_input_frames += 1
        user_input_ratio = user_input_frames / total_frames if total_frames > 0 else 0
        rotation_input_ratio = rotation_input_frames / total_frames if total_frames > 0 else 0
        assist_ratio = assisted_frames / user_input_frames if user_input_frames > 0 else 0
        print(f"Assisted frames: {assisted_frames} ({assist_ratio:.2%})")
        print(f"User input frames: {user_input_frames} ({user_input_ratio:.2%})")
        print(f"Rotation input frames: {rotation_input_frames} ({rotation_input_ratio:.2%})")

    def export_history(self, filepath):
        """Export probability history to CSV for analysis"""
        df = pd.DataFrame(self.frame_records)
        df.to_csv(filepath, index=False)
    
    def import_history(self, filepath):
        """Import probability history from CSV for analysis"""
        df = pd.read_csv(filepath)
        json_columns = ['user_action', 'assist_action', 'eef_pose', 'prob_distribution']
        for col in json_columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if pd.notnull(x) else None)
        self.frame_records = df.to_dict(orient='records')

    def analyze_goal_focus(self, grasping_goal: int, manipulation_goal: int):
        mode_stats = {
            'grasping_assistance': {
                'target': grasping_goal,
                'total': 0,
                'match': 0
            },
            'manipulation_assistance': {
                'target': manipulation_goal,
                'total': 0,
                'match': 0
            }
        }
        for row in self.frame_records:
            mode = row.get("assistance_mode")
            goal = row.get("assistance_goal")
            if mode in mode_stats:
                mode_stats[mode]['total'] += 1
                if goal == mode_stats[mode]['target']:
                    mode_stats[mode]['match'] += 1
        weighted_sum = 0
        total_frames = 0
        for mode, stats in mode_stats.items():
            total = stats['total']
            match = stats['match']
            target = stats['target']
            ratio = match / total if total > 0 else 0
            print(f"[{mode}]")
            print(f"  Total frames: {total}")
            print(f"  Goal == {target}: {match} frames")
            print(f"  Match ratio: {ratio:.2%}\n")
            weighted_sum += match
            total_frames += total
        weighted_avg = weighted_sum / total_frames if total_frames > 0 else 0
        print(f"Weighted average match ratio: {weighted_avg:.2%}")

    def plot_belief_over_time(self, use_time=True, grasping_goal=0, manipulation_goal=1):
        """Plot the belief over time for the specified goals"""
        time_or_frame = []
        prob_distributions = []
        mask_correct = []
        for row in self.frame_records:
            prob = row.get("prob_distribution")
            if prob is None:
                continue
            time_or_frame.append(row["time_elapsed"] if use_time else row["frame_id"])
            prob_distributions.append(prob)
            mode = row.get("assistance_mode")
            goal = row.get("assistance_goal")
            if (mode == "grasping_assistance" and goal == grasping_goal) or \
            (mode == "manipulation_assistance" and goal == manipulation_goal):
                mask_correct.append(True)
            else:
                mask_correct.append(False)
        if not prob_distributions:
            print("No prob_distribution data available.")
            return
        all_keys = sorted({k for dist in prob_distributions for k in dist.keys()}, key=int)
        data = {k: [] for k in all_keys}
        for dist in prob_distributions:
            for k in all_keys:
                data[k].append(dist.get(k, 0.0))
        fixed_colors = {
            '0': '#1f77b4',
            '1': '#ff7f0e',
            '2': '#2ca02c',
            '3': '#d62728',
            '4': '#9467bd',
            '5': '#8c564b',
        }
        color_list = [fixed_colors.get(k, '#aaaaaa') for k in all_keys]
        fig, ax = plt.subplots(figsize=(10, 2.5))
        stack_data = [data[k] for k in all_keys]
        ax.stackplot(time_or_frame, *stack_data, colors=color_list)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(0, 80)
        ax.set_xticks([0, 20.0, 40.0, 60.0, 80.0])
        # ax.set_ylabel("Belief", fontsize=14)
        ax.tick_params(labelsize=14)
        def find_segments(mask):
            segments = []
            start = None
            for i, val in enumerate(mask):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    segments.append((start, i - 1))
                    start = None
            if start is not None:
                segments.append((start, len(mask) - 1))
            return segments
        for start_idx, end_idx in find_segments(mask_correct):
            start_t = time_or_frame[start_idx]
            end_t = time_or_frame[end_idx]
            ax.axvspan(start_t, end_t, color='#444444', alpha=0.35)
        plt.tight_layout()
        plt.show()

    def plot_3d_trajectory(self, draw_gripper=True, num_gripper_samples=5):
        '''Plot the 3D trajectory of the end-effector'''
        xyz_points = []
        colors = []
        poses = []
        for row in self.frame_records:
            eef = row.get("eef_pose")
            if not eef:
                continue
            try:
                pose = np.array(eef)
                pos = pose[:3, 3]
                xyz_points.append(pos)
                colors.append('red' if row.get("is_assisted") else 'grey')
                poses.append(pose)
            except Exception as e:
                print(f"Error parsing eef_pose: {e}")
                continue
        if not xyz_points:
            print("No EEF poses to plot.")
            return
        xs, ys, zs = np.array(xyz_points).T
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, c=colors, s=8)
        # Set equal axis limits for better visualization
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        # Draw gripper orientations
        if draw_gripper and len(poses) >= num_gripper_samples:
            indices = np.linspace(0, len(poses)-1, num_gripper_samples, dtype=int)
            for idx in indices:
                pose = poses[idx]
                origin = pose[:3, 3]
                R = pose[:3, :3]
                axis_len = 0.05
                ax.quiver(*origin, *R[:, 0]*axis_len, color='r')    # red for x-axis
                ax.quiver(*origin, *R[:, 1]*axis_len, color='g')    # green for y-axis
                ax.quiver(*origin, *R[:, 2]*axis_len, color='b')    # blue for z-axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D End-Effector Trajectory with Gripper Orientation')
        plt.tight_layout()
        plt.show()

    def calculate_trajectory_length(self):
        """Calculate the total trajectory length based on end-effector poses"""
        total_length = 0.0
        prev_pos = None
        for row in self.frame_records:
            eef = row.get("eef_pose")
            if not eef:
                continue
            try:
                pose = np.array(eef)
                current_pos = pose[:3, 3]  # Extract position (x, y, z)
                
                if prev_pos is not None:
                    # Calculate Euclidean distance between consecutive positions
                    distance = np.linalg.norm(current_pos - prev_pos)
                    total_length += distance
                prev_pos = current_pos
            except Exception as e:
                print(f"Error parsing eef_pose: {e}")
                continue
        return total_length


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='tasc', help='Input type: pure_teleop, tasc')
    parser.add_argument('--episode_idx', type=int, default='1', help='Index of the record to analyze')
    parser.add_argument('--task_idx', type=int, default='1', help='Index of the task to analyze. ' \
                                                            '0-banana&plate, 1-marker&cup, 2-hammer&peg')
    args = parser.parse_args()
    idx = args.episode_idx
    if args.input.lower() == "pure_teleop":
        csv_path = os.path.join(os.path.dirname(__file__), f"../data/pure_teleop_record_{idx}.csv")
    else:
        csv_path = os.path.join(os.path.dirname(__file__), f"../data/tasc_record_{idx}.csv")
    analyzer = ResultAnalyzer()
    analyzer.import_history(csv_path)
    analyzer.summarize()
    obj_pairs = ((2, 0), (5, 4), (3, 1))
    obj_pair = obj_pairs[args.task_idx]
    analyzer.analyze_goal_focus(grasping_goal=obj_pair[0], manipulation_goal=obj_pair[1])
    analyzer.plot_belief_over_time(use_time=True, grasping_goal=obj_pair[0], manipulation_goal=obj_pair[1])
    analyzer.plot_3d_trajectory()
