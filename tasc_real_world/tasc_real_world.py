import os
import cv2
import time
import numpy as np
import argparse
import asyncio

from PIL import Image
from tasc.utils import *
from tasc.vision_module.mask_generator import MaskGenerator
from tasc.vision_module.object_detector import ObjectDetector
from tasc.shared_controller.shared_controller import SharedController
from tasc.shared_controller.place_optimizer import PlaceOptimizer
from tasc.goal_predictor.scene_graph import SceneGraph
from websocket_client import WebSocketClient
from tasc.result_analyzer import ResultAnalyzer

script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "../data/real_world")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--websocket-uri", type=str, default="ws://localhost:8765", help="WebSocket URI for ROS2 communication")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--record", action="store_true", help="Record the simulation data")
    parser.add_argument("--pure_teleop", action="store_true", help="Disable the assist mode")
    args = parser.parse_args()

    enable_debug = args.debug
    enable_record = args.record

    # Set up logging and configuration
    logging = get_logger("tasc_real_world", verbose=args.verbose, clean_root=True)
    is_assistance_disabled = args.pure_teleop

    # Load configuration
    config_path = os.path.join(script_dir, "../configs/real_world_config.yaml")
    real_world_info_path = os.path.join(script_dir, "../configs/real_world_info.yaml")
    global_config = get_config(config_path)
    real_world_info = get_sim_info(real_world_info_path)

    # Create WebSocket client
    ws_client = WebSocketClient(args.websocket_uri)
    ws_client.start()
    time.sleep(2)
    
    if not ws_client.is_connected():
        logging.error(f"Failed to connect to WebSocket server at {args.websocket_uri}")
        logging.error("Please ensure the ROS2 tasc_node is running")
        exit(1)
    logging.info(f"Connected to WebSocket server at {args.websocket_uri}")

    loop_count = 0
    init_loop_offset = 20
    keypoint_detector = None
    shared_controller = SharedController()
    result_analyzer = ResultAnalyzer()
    manipulation_goal = None
    id_goal = None
    place_optimizer = None
    scene_graph = None
    rgb = None
    points = None
    prob_distribution = None
    sg_img = None
    assist_action = np.zeros(6)
    user_action = np.zeros(6)
    intrinsic_matrix = np.array(global_config['main']['intrinsic_matrix']).reshape(3, 3)
    T_cam2world = np.array(global_config['main']['extrinsic_matrix']).reshape(4, 4)
    _, _, objects_list, interactions, constraints = process_sim_info(real_world_info, scene_name='banana_cup')
    
    try:
        while True:
            start = time.time()
            result_analyzer.start_timer()

            # Check connection status and reconnect if necessary
            if not ws_client.is_connected():
                logging.warning("WebSocket connection lost, reconnecting...")
                ws_client.ensure_connection()
                if not ws_client.is_connected():
                    logging.error("Failed to reconnect to WebSocket server")
                    exit(1)

            # Get state information from ROS2 via WebSocket
            eef_pose = ws_client.get_eef_pose()
            action_dict = {
                "eef": ws_client.get_twist(),
                "gripper": ws_client.get_gripper_action()
            }
        
            if loop_count <= init_loop_offset:        
                if loop_count == init_loop_offset:
                    logging.info("Starting initialization phase...")
                    logging.info("object_name_list: %s", objects_list)

                    # Get RGB and point cloud data
                    rgb, points = asyncio.run(ws_client.request_data_once())
                    if rgb is None or points is None:
                        logging.error("Failed to get RGB and point cloud data from ROS2")
                        exit(1)
                    logging.info(f"RGB: {rgb.shape}, Points: {points.shape}")

                    # Stop WebSocket during long processing
                    ws_client.stop()
                    
                    try:
                        ## Segmentation
                        logging.info("Step 1/4: Object detection and segmentation...")
                        obs_image = Image.fromarray(rgb[:, :, ::-1])  # Convert RGB to BGR for PIL
                        mask_generator = MaskGenerator(global_config)
                        boxes, logits, phrases = mask_generator.get_scene_object_bboxes(obs_image, objects_list, visualize=False, logdir=None)
                        segmasks = mask_generator.get_segmentation_masks(obs_image, boxes, logits, phrases, visualize=enable_record, 
                                                                         save_path=os.path.join(log_dir, f"seg_{int(time.time())}.png"))
                        logging.info(f"Detected {len(segmasks)} objects")
                        if len(segmasks) < len(objects_list):
                            raise RuntimeError("Not all objects detected!")
                        
                        ## Build Scene Graph
                        logging.info("Step 2/4: Building scene graph...")
                        scene_graph = SceneGraph(objects_list, interactions) 
                        keypoint_detector = ObjectDetector(global_config, points, T_cam2world, intrinsic_matrix, image=rgb[:, :, ::-1])
                        keypoint_detector.update_visual_info(scene_graph, segmasks, refine_radius=35)

                        ## Grasp Planner
                        logging.info("Step 3/4: Computing grasp poses...")
                        keypoint_detector.update_grasp_poses(scene_graph, visualize=enable_record, 
                                                             save_path=os.path.join(log_dir, f"grasps_{int(time.time())}.png"))
                        keypoint_detector.update_pcd_obb(scene_graph, visualize=False)
                        
                        ## Shared Controller
                        shared_controller.assistance_mode = 'grasping_assistance'
                        scene_graph.update_stage('reset', eef_pose)

                        ## Point Predictor
                        logging.info("Step 4/4: Initializing point predictor...")
                        # Restart WebSocket to get EEF pose
                        ws_client.start()
                        time.sleep(1)
                        if ws_client.is_connected():
                            eef_pose = ws_client.get_eef_pose()
                            scene_graph.update_stage(eef_pose=eef_pose)
                        else:
                            logging.warning("Using identity pose for initialization")
                            scene_graph.update_stage(eef_pose=np.eye(4))
                        logging.info("✅ Initialization completed!")
                    except Exception as e:
                        logging.error(f"Initialization failed: {e}")
                        exit(1)
            else:
                ## Point Predictor
                user_action[0:3] = np.array([action_dict["eef"][0], action_dict["eef"][1], action_dict["eef"][2]])
                user_action[3:6] = np.array([action_dict["eef"][3], action_dict["eef"][4], action_dict["eef"][5]])
                # Update policy
                if not np.allclose(user_action, 0.0, atol=0.1):
                    scene_graph.update_prediction_policies(eef_pose, user_action)
                
                # Shared Control
                assist_action, gripper_action = shared_controller.assist_action_real_world(eef_pose, 
                                                                                        user_action, 
                                                                                        action_dict["gripper"], 
                                                                                        scene_graph)
                # Update posture alignment during the manipulation assistance mode
                if shared_controller.is_state_change and shared_controller.assistance_mode == "manipulation_assistance":
                    keypoint_detector.update_pcd_obb(scene_graph, visualize=enable_record, 
                                                     save_path=os.path.join(log_dir, f"pcd_obb_{int(time.time())}.png"))
                    grasped_node_id, grasped_node = scene_graph.get_grasped_node()
                    relations = scene_graph.get_relations(grasped_node_id)
                    if len(relations) > 0:
                        for other_id, relation in relations.items():
                            print(f"Relation between {grasped_node_id} and {other_id}: {relation[0]}")
                            constraint = []
                            for ct in constraints:
                                if ct[0] == grasped_node_id and ct[1] == other_id:
                                    logging.info(f"Applying constraint: {ct[2]}")
                                    constraint = ct[2]
                                    break
                            if len(constraint) == 0:
                                continue
                            other_node = scene_graph.get_node_by_id(other_id)
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
                
                # Send assistance action to websocket server
                if not is_assistance_disabled and not np.allclose(assist_action, 0.0, atol=0.1):
                    asyncio.run(ws_client.send_assist_action(assist_action, gripper_action))

                # Record the history of actions and observations
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
            if loop_count % 10 == 0:
                logging.info(f"Stage: {shared_controller.assistance_mode}, Goal ID: {id_goal}")
                logging.info(f"Assistance Action: {assist_action}")
            loop_count += 1

            if enable_debug and loop_count > init_loop_offset:
                is_mujoco_camera = global_config['vision_module'].get('is_mujoco_camera', False)
                img = rgb.astype(np.uint8)   
                # Visualize the offscreen render and the keypoints with distribution
                draw_pose_on_image(img, eef_pose, T_cam2world, intrinsic_matrix, is_mujoco_camera=is_mujoco_camera)
                if manipulation_goal is not None and manipulation_goal.shape == (4, 4):
                    draw_pose_on_image(img, manipulation_goal, T_cam2world, intrinsic_matrix, is_mujoco_camera=is_mujoco_camera)
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
                    if prob_distribution is not None:
                        cv2.putText(img, f"{prob_distribution[obj_id]:.3f}", (u + 15, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                    # grasp pose
                    if obj_node.grasp_poses is not None:
                        for grasp_pose in obj_node.grasp_poses:
                            draw_pose_on_image(img, grasp_pose, T_cam2world, intrinsic_matrix, is_mujoco_camera=is_mujoco_camera)
                    # # pcd pose
                    # draw_pose_on_image(img, obj_node.pcd_pose, T_cam2world, intrinsic_matrix, is_mujoco_camera=is_mujoco_camera)
                if loop_count % 500 == 0:
                    h, w = img.shape[:2]
                    sg_img = scene_graph.get_visualization(w, h, use_pixel_center=True)
                    # sg_img = sg_img[..., ::-1].astype(np.uint8)
                if sg_img is not None:
                    # Ensure both images have the same height before horizontal concatenation
                    if img.shape[0] != sg_img.shape[0]:
                        sg_img = cv2.resize(sg_img, (sg_img.shape[1], img.shape[0]))
                    img = np.hstack((img, sg_img))
                cv2.imshow("offscreen render", img)
                cv2.waitKey(1)
                if loop_count % 500 == 0 and enable_record:
                    # Save the offscreen render
                    save_path = os.path.join(log_dir, f"offscreen_render_{int(time.time())}.png")
                    cv2.imwrite(save_path, img)
                    logging.info(f"Offscreen render saved to: {save_path}")

            # Control loop frequency
            elapsed = time.time() - start
            if elapsed < 0.1:  # 10 Hz
                time.sleep(0.1 - elapsed)

    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if enable_record:
            logging.info("Exporting record...")
            result_analyzer.export_history(os.path.join(log_dir, f"real_world_record_{int(time.time())}.csv"))
        ws_client.stop()
        cv2.destroyAllWindows()
        logging.info("Finished")