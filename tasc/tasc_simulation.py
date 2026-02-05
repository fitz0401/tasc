import os
import cv2
import time
import numpy as np
import argparse

from PIL import Image
from copy import deepcopy
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
from tasc.utils import *
from tasc.vision_module.object_detector import ObjectDetector
from tasc.shared_controller.shared_controller import SharedController
from tasc.shared_controller.place_optimizer import PlaceOptimizer
from tasc.goal_predictor.scene_graph import SceneGraph
from robosuite_env.table_env import TableEnv, get_camera_data
from tasc.result_analyzer import ResultAnalyzer
from tasc.mllm_client import GPTClient

script_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--camera", type=str, default="table_view", help="Name of camera to render")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max_fr", default=50, type=int, help="Sleep when simulation runs faster than specified frame rate; 20 fps is real time.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--record", action="store_true", help="Record the simulation data")
    parser.add_argument("--pure_teleop", action="store_true", help="Disable the assist mode")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for relationship and pose constraints")
    parser.add_argument("--use_graspnet", action="store_true", help="Use GraspNet-baseline for grasp planning")
    args = parser.parse_args()

    # Set debug mode
    logging = get_logger("tasc", verbose=args.verbose, clean_root=True)
    enable_debug = args.debug
    enable_record = args.record
    is_assistance_disabled = args.pure_teleop
    use_llm = args.use_llm
    use_graspnet = args.use_graspnet

    # Load global config and simulation info
    config_path = os.path.join(script_dir, f"../configs/config.yaml")
    sim_info_path = os.path.join(script_dir, f"../configs/sim_info.yaml")
    save_dir = os.path.join(script_dir, "../data")
    os.makedirs(save_dir, exist_ok=True)
    global_config = get_config(config_path)
    sim_info = get_sim_info(sim_info_path)

    # Create environment
    env = TableEnv(
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        hard_reset=False,
        use_camera_obs=True,
        camera_names=[args.camera],
        camera_heights=args.height,
        camera_widths=args.width,
        camera_depths=True,
        camera_segmentations="class",
        use_object_obs=True,
        render_gpu_device_id=0,
        render_camera="table_view"  # or free
    )

    # Setup printing options for numbers
    np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    result_analyzer = ResultAnalyzer()
    exp_idx = -1
    while True:
        # Reset the environment
        obs = env.reset()

        # Initialize device control
        device.start_control()

        # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        loop_count = 0
        init_loop_offset = 20   # number of frames to skip at the beginning for object stability
        object_detector = None
        shared_controller = SharedController()
        exp_idx += 1
        if enable_record and exp_idx > 0:
            if is_assistance_disabled:
                result_analyzer.export_history(os.path.join(save_dir, f"pure_teleop_record_{exp_idx}.csv"))
            else:
                result_analyzer.export_history(os.path.join(save_dir, f"tasc_record_{exp_idx}.csv"))
        result_analyzer.reset()
        exp_start_flag = False
        manipulation_goal = None
        id_goal = None
        place_optimizer = None
        scene_graph = None
        sg_img = None
        assist_action = np.zeros(6)  # assist twist for shared control
        user_action = np.zeros(6)  # user twist for shared control
        intrinsic_matrix = get_camera_intrinsic_matrix(env.sim, args.camera, args.height, args.width)
        T_cam2world = get_camera_trans(env.sim.data, args.camera)
        sim_name_dict, mask_name_dict, objects_list, interactions, constraints = process_sim_info(sim_info)
        gpt_client = GPTClient()
        while True:
            start = time.time()
            
            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action()

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                break

            action_dict = deepcopy(input_ac_dict)  # {}
            # set arm actions
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type
                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            ## Get state information        
            eef_pose = get_trans_mat(obs['robot0_eef_pos'], obs['robot0_eef_quat'])
            '''
                Scene Analysis and Object Detection
                Process camera data, detect objects, and build scene graph.
            '''
            if loop_count <= init_loop_offset:
                ## stabilize the objects in the scene
                stabilize_object_pose(env.sim)
                if loop_count == init_loop_offset:
                    ## Get observation
                    rgb, points, seg, depth = get_camera_data(env.sim, args.camera, args.width, args.height)

                    # debug info
                    logging.debug("Observation keys: %s", obs.keys())
                    logging.debug("Objects: %s", env.sim.model.body_names)
                    logging.debug("Available cameras: %s", env.sim.model.camera_names)
                    
                    '''
                    LLM-based Object Relationship Detection
                    Use GPT to identify objects and their relationships automatically.
                    '''
                    if use_llm:
                        llm_objects, llm_interactions, _ = gpt_client.obtain_object_relationship_auto(rgb[:,:,::-1])
                        # Create mapping from LLM indices to 0-based indices
                        llm_to_zero_based = {}
                        objects_list = []
                        for i, (llm_idx, obj_name) in enumerate(llm_objects):
                            llm_to_zero_based[llm_idx] = i
                            objects_list.append((i, obj_name))
                        interactions = []
                        for actor_idx, target_idx, action in llm_interactions:
                            if actor_idx in llm_to_zero_based and target_idx in llm_to_zero_based:
                                interactions.append((llm_to_zero_based[actor_idx], llm_to_zero_based[target_idx], action))
                        logging.info("Objects List from LLM: %s", objects_list)
                        logging.info("Interactions from LLM: %s", interactions)
                        
                        '''
                        Object Segmentation using Vision Models
                        Generate bounding boxes and segmentation masks for detected objects.
                        '''
                        ## Segmentation
                        obs_image = Image.fromarray(rgb)
                        from tasc.vision_module.mask_generator import MaskGenerator
                        mask_generator = MaskGenerator(global_config)
                        # get bounding boxes
                        boxes, logits, phrases = mask_generator.get_scene_object_bboxes(obs_image, objects_list, visualize=False, logdir=None)
                        # get segmentation masks
                        segmasks = mask_generator.get_segmentation_masks(obs_image, boxes, logits, phrases, visualize=False, save_path=None)
                    else:
                        segmasks = extract_ordered_masks(seg, env.sim, mask_name_dict)
                    
                    '''
                    Scene Graph Construction and Keypoint Detection
                    Build scene graph and initialize keypoint detection system.
                    '''
                    logging.info("Objects List: %s", objects_list)
                    combined_mask = np.any(np.array(segmasks) > 0, axis=0).astype(np.uint8)
                    logging.info(f"Number of objects detected: {len(segmasks)}")
                    if len(segmasks) < len(objects_list):
                        raise RuntimeError("Not all objects detected!")
                    
                    ## Build Scene Graph
                    scene_graph = SceneGraph(objects_list, interactions) 
                    object_detector = ObjectDetector(global_config, points, T_cam2world, intrinsic_matrix, image=rgb)
                    object_detector.update_visual_info(scene_graph, segmasks)

                    '''
                    Grasp Planning
                    '''
                    ## Grasp Planner
                    if use_graspnet:
                        object_detector.update_grasp_poses(scene_graph, visualize=True)
                    else:
                        object_detector.update_grasp_poses_offline(scene_graph, sim_name_dict, env.sim)

                    ## Point Predictor
                    scene_graph.update_stage(eef_pose=eef_pose)
                    logging.info("Scene graph: %s", scene_graph)
            else:
                '''
                Active Control Phase
                Handle user input, prediction updates, and shared control assistance.
                '''
                # save_scene(save_dir, rgb, points, combined_mask, args.camera, loop_count)
                ## Point Predictor
                user_action[0:3] = np.array([action_dict["right"][0], action_dict["right"][1], action_dict["right"][2]])
                user_action[3:6] = np.array([action_dict["right"][3], action_dict["right"][4], action_dict["right"][5]])
                # Update policy
                if not np.allclose(user_action, 0.0, atol=1e-4):
                    scene_graph.update_prediction_policies(eef_pose, user_action)
                    ## Result Analyzer
                    if not exp_start_flag:
                        exp_start_flag = True
                        result_analyzer.start_timer()

                '''
                Shared Control System
                Compute assistance actions and handle pose constraints during manipulation.
                '''
                ## Shared Control
                assist_action, gripper_action = shared_controller.assist_action(eef_pose, 
                                                                                user_action, 
                                                                                action_dict["right_gripper"][0], 
                                                                                scene_graph)
                '''
                Pose Constraint Optimization
                Update pose relationships and constraints during manipulation assistance.
                '''
                # Update poseture alignment during the manipulation assistance mode
                if shared_controller.is_state_change and shared_controller.assistance_mode == "manipulation_assistance":
                    obb_images = object_detector.update_pcd_obb(scene_graph, visualize=False, save_path=None, return_images=use_llm)
                    grasped_node_id, grasped_node = scene_graph.get_grasped_node()
                    relations = scene_graph.get_relations(grasped_node_id)
                    for other_id, relation in relations.items():
                        print(f"Relation between {grasped_node_id} and {other_id}: {relation[0]}")
                        other_node = scene_graph.get_node_by_id(other_id)
                        constraint = []
                        if use_llm:
                            constraint, _, _ = gpt_client.obtain_pose_constraints_auto(obb_images[grasped_node_id][:,:,::-1], 
                                                                                        obb_images[other_id][:,:,::-1], 
                                                                                        grasped_node.name, 
                                                                                        other_node.name, 
                                                                                        relation[0])
                            logging.info(f"Constraint from LLM: {constraint}")
                        else:
                            for ct in constraints:
                                if ct[0] == grasped_node_id and ct[1] == other_id:
                                    logging.info(f"Applying constraint: {ct[2]}")
                                    constraint = ct[2]
                                    break
                        if len(constraint) == 0:
                            continue
                        place_optimizer = PlaceOptimizer(grasped_node.pcd_pose[:3, :3],
                                                         other_node.pcd_pose[:3, :3], 
                                                         constraint)
                        rot_opt = place_optimizer.solve()
                        relation[1] = rot_opt @ eef_pose[:3, :3]
                        logging.info(f"Relation between {grasped_node_id} and {other_id}: {relation[0]}, rotation: {relation[1]}")
                        grasped_node_pcd = grasped_node.pcd_points
                        grasped_node_center = grasped_node.center_point
                        other_node_pcd = other_node.pcd_points
                        grasped_node_pcd_new = (grasped_node_pcd - grasped_node_center) @ rot_opt.T + grasped_node_center
                        original_points = np.vstack((other_node_pcd, grasped_node_pcd))
                        combined_points = np.vstack((other_node_pcd, grasped_node_pcd_new))
                        # visualize_point_cloud(original_points, title="Before", view=(15, -45))
                        # visualize_point_cloud(combined_points, title="After", color='r', view=(15, -45))

                manipulation_goal, id_goal = shared_controller.manipulation_goal()
                prob_distribution = scene_graph.get_goal_distribution()
                if not is_assistance_disabled:
                    action_dict["right"][0:6] += assist_action
                    action_dict['right_gripper'][0] = gripper_action

                # Record the frame
                if enable_record:
                    result_analyzer.record_frame(
                        count=loop_count,
                        prob_distribution_with_ids=prob_distribution,
                        assistance_mode=shared_controller.assistance_mode,
                        assistance_goal=id_goal,
                        is_assisted=np.any(assist_action),
                        user_action=user_action,
                        assist_action=assist_action,
                        eef_pose=eef_pose
                    )

            # Debugging info
            if loop_count % 100 == 0:
                logging.debug("EE Pose: %s", eef_pose)
                logging.debug("GP Pose: %s", manipulation_goal)
                logging.debug("Action: %s", action_dict)
                logging.info(f"Stage: {shared_controller.assistance_mode}, Goal ID: {id_goal}")
                logging.debug("Joint pos: %s", active_robot._joint_positions)

            '''
            Robot Action Execution
            Maintain gripper states and execute robot actions in simulation.
            '''
            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]
            
            # Step through the simulation and render
            obs, reward, done, info = env.step(env_action)
            img = np.flip(obs[args.camera + "_image"][..., ::-1], 0).astype(np.uint8)        
            env.render()

            '''
            Debug Visualization
            Display debug information including poses, keypoints, and scene graph.
            '''
            if enable_debug:
                # Visualize the offscreen render and the keypoints with distribution
                draw_pose_on_image(img, eef_pose, T_cam2world, intrinsic_matrix)
                if manipulation_goal is not None and manipulation_goal.shape == (4, 4):
                    draw_pose_on_image(img, manipulation_goal, T_cam2world, intrinsic_matrix)
                if loop_count > init_loop_offset:
                    for obj_id, obj_node in scene_graph.nodes_dict.items():
                        if not obj_node.is_active:
                            continue
                        # center point
                        if obj_node.pixel_center is None or len(obj_node.pixel_center) < 2:
                            logging.warning(f"Object {obj_id} has invalid pixel_center: {obj_node.pixel_center}")
                            continue
                        obj_pixel_center = obj_node.pixel_center
                        u = int(obj_pixel_center[0])
                        v = int(obj_pixel_center[1])
                        cv2.circle(img, (u, v), 6, (0, 255, 0), -1)
                        cv2.putText(img, str(obj_id), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(img, f"{prob_distribution[obj_id]:.3f}", (u + 15, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                        # grasp pose
                        if obj_node.grasp_poses is not None:
                            for grasp_pose in obj_node.grasp_poses:
                                draw_pose_on_image(img, grasp_pose, T_cam2world, intrinsic_matrix)
                    if loop_count % 10 == 0:
                        sg_img = scene_graph.get_visualization(args.width, args.height, use_pixel_center=True)
                        sg_img = sg_img[..., ::-1].astype(np.uint8)
                    if sg_img is not None:
                        img = np.hstack((img, sg_img))
                cv2.imshow("offscreen render", img)
                cv2.waitKey(1)

            # limit frame rate if necessary
            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)

            loop_count += 1