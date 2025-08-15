import time
import numpy as np
import asyncio
import websockets
import json
import threading
from queue import Queue
import base64
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import open3d as o3d
import logging


class WebSocketClient:
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.data_queue = Queue()
        self.running = False
        self.thread = None
        self.rgb_data = None
        self.points_data = None
        self.twist_data = np.zeros(6)
        self.eef_pose_data = np.eye(4)
        self.gripper_action_data = 0  # 0 or 1 or -1
        self.data_received = False
        
    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.uri, max_size=10 * 1024 * 1024)
            print(f"Connected to WebSocket at {self.uri}")
            return True
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def request_pointcloud_and_image(self):
        """Request point cloud and RGB image from ROS2"""
        if self.websocket:
            request = {"type": "get_pointcloud_and_image"}
            await self.websocket.send(json.dumps(request))
    
    async def listen(self):
        """Listen for incoming messages"""
        try:
            while self.running and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "pointcloud_and_image":
                    # Decode base64 encoded data
                    rgb_bytes = base64.b64decode(data["rgb"])
                    points_bytes = base64.b64decode(data["points"])
                    
                    # Convert to numpy arrays
                    self.rgb_data = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(data["rgb_shape"])
                    self.points_data = np.frombuffer(points_bytes, dtype=np.float32).reshape(data["points_shape"])
                    self.data_received = True
                    print(f"Received RGB shape: {self.rgb_data.shape}, Points shape: {self.points_data.shape}")
                    
                elif data["type"] == "twist":
                    self.twist_data = np.array(data["twist"])
                    
                elif data["type"] == "eef_pose":
                    self.eef_pose_data = np.array(data["pose"]).reshape(4, 4)
                    
                elif data["type"] == "gripper_action":
                    self.gripper_action_data = int(data["gripper_action"])
                    
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"Error in WebSocket listener: {e}")
        finally:
            self.running = False
    
    def start(self):
        """Start WebSocket client in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_async)
        self.thread.start()
    
    def _run_async(self):
        """Run async event loop in thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_main())
    
    async def send_heartbeat(self):
        """Send periodic heartbeat to keep connection alive"""
        while self.running and self.websocket:
            try:
                if self.websocket.open:
                    heartbeat_msg = {"type": "heartbeat", "timestamp": time.time()}
                    await self.websocket.send(json.dumps(heartbeat_msg))
                    await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                else:
                    break
            except Exception as e:
                logging.debug(f"Heartbeat failed: {e}")
                break

    async def _async_main(self):
        """Main async function"""
        if await self.connect():
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self.send_heartbeat())
            listen_task = asyncio.create_task(self.listen())
            
            # Wait for either task to complete
            try:
                await asyncio.gather(heartbeat_task, listen_task)
            except Exception as e:
                logging.debug(f"WebSocket tasks error: {e}")
            finally:
                # Cancel remaining tasks
                heartbeat_task.cancel()
                listen_task.cancel()
    
    def stop(self):
        """Stop WebSocket client"""
        self.running = False
        if self.websocket:
            # Use asyncio to close the websocket properly
            try:
                # Check if websocket is still open before trying to close
                should_close = True
                try:
                    should_close = self.websocket.open
                except AttributeError:
                    # For different websockets versions, assume we should try to close
                    should_close = True
                
                if should_close:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.websocket.close())
                    else:
                        loop.run_until_complete(self.websocket.close())
            except Exception:
                pass
        if self.thread:
            self.thread.join()
    
    def get_twist(self):
        """Get latest twist data"""
        return self.twist_data.copy()
    
    def get_eef_pose(self):
        """Get latest EEF pose"""
        return self.eef_pose_data.copy()
    
    def get_gripper_action(self):
        """Get latest gripper action (0 or 1 or -1)"""
        return self.gripper_action_data
    
    def ensure_connection(self):
        """Ensure WebSocket connection is active, reconnect if necessary"""
        if not self.is_connected():
            logging.warning("WebSocket connection lost, attempting to reconnect...")
            self.stop()
            time.sleep(1)
            self.start()
            time.sleep(2)
            return self.is_connected()
        return True
    
    def get_connection_health(self):
        """Get detailed connection health information"""
        try:
            websocket_open = self.websocket.open if self.websocket else False
        except AttributeError:
            websocket_open = self.websocket is not None
            
        thread_alive = self.thread.is_alive() if self.thread else False
        
        return {
            "running": self.running,
            "websocket_exists": self.websocket is not None,
            "websocket_open": websocket_open,
            "thread_alive": thread_alive,
            "data_received": self.data_received,
            "is_healthy": self.running and websocket_open and thread_alive
        }

    async def send_assist_action(self, assist_action, gripper_action=None):
        """Send assist action back to ROS2"""
        if self.websocket:
            try:
                # Check if websocket is open
                is_open = True
                try:
                    is_open = self.websocket.open
                except AttributeError:
                    # Assume it's open if we can't check
                    is_open = True
                
                if is_open:
                    if isinstance(assist_action, np.ndarray):
                        assist_action_list = [round(float(x), 3) for x in assist_action.flatten()]
                    elif isinstance(assist_action, (list, tuple)):
                        assist_action_list = [round(float(x), 3) for x in assist_action]
                    else:
                        assist_action_list = [round(float(assist_action), 3)] if np.isscalar(assist_action) else [0.0] * 6
                    request = {
                        "type": "assist_action",
                        "assist_action": assist_action_list
                    }
                    await self.websocket.send(json.dumps(request))
            except Exception as e:
                logging.debug(f"Failed to send assist action: {e}")

    async def request_data_once(self):
        """Request point cloud and image once"""
        if await self.connect():
            await self.request_pointcloud_and_image()
            # Wait for response
            timeout = 10  # seconds
            start_time = time.time()
            while not self.data_received and time.time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    if data["type"] == "pointcloud_and_image":
                        rgb_bytes = base64.b64decode(data["rgb"])
                        points_bytes = base64.b64decode(data["points"])
                        
                        self.rgb_data = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(data["rgb_shape"])
                        self.points_data = np.frombuffer(points_bytes, dtype=np.float32).reshape(data["points_shape"])
                        self.data_received = True
                        print(f"Received data once - RGB: {self.rgb_data.shape}, Points: {self.points_data.shape}")
                        break
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error receiving data: {e}")
                    break
            
            if not self.data_received:
                print("Timeout waiting for pointcloud and image data")
            
            await self.websocket.close()
        
        return self.rgb_data, self.points_data
    
    def is_connected(self):
        """Check if WebSocket is connected and running"""
        try:
            return self.running and self.websocket and self.websocket.open
        except AttributeError:
            # Fallback for different websockets versions
            return self.running and self.websocket is not None
    
    def get_connection_status(self):
        """Get detailed connection status"""
        websocket_open = False
        try:
            websocket_open = self.websocket.open if self.websocket else False
        except AttributeError:
            websocket_open = self.websocket is not None
            
        return {
            "running": self.running,
            "websocket_exists": self.websocket is not None,
            "websocket_open": websocket_open,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "data_received": self.data_received
        }

# Test function to verify the WebSocket client
async def test_websocket_connection(uri="ws://localhost:8765"):
    """Test function to verify WebSocket connection"""
    print(f"Testing WebSocket connection to {uri}")
    
    try:
        async with websockets.connect(uri, max_size=10 * 1024 * 1024) as websocket:
            print("✓ Successfully connected to WebSocket server")
            
            # Test sending a request
            test_request = {"type": "test", "message": "hello"}
            await websocket.send(json.dumps(test_request))
            print("✓ Successfully sent test message")
            
            # Try to receive a response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"✓ Received response: {data}")
            except asyncio.TimeoutError:
                print("⚠ No response received (timeout)")
            
            return True
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def visualize_rgb_and_points(rgb, points, roi_bounds=None):
    """
    Visualize RGB image and point cloud using Open3D
    
    Args:
        rgb: RGB image (H, W, 3)
        points: Point cloud (H, W, 3) or (N, 3)
        roi_bounds: [xmin, xmax, ymin, ymax, zmin, zmax] for filtering display region
    """
    try:
        # Default ROI bounds if not specified
        if roi_bounds is None:
            roi_bounds = [-0.2, 0.8, -0.2, 0.6, 0.0, 0.8]  # xmin, xmax, ymin, ymax, zmin, zmax
        
        # Display RGB image
        if rgb is not None:
            print(f"RGB Image shape: {rgb.shape}")
            cv2.imshow("RGB Image", rgb)
            cv2.waitKey(1000)  # Show for 1 second
            cv2.destroyAllWindows()
        else:
            print("No RGB image data received")
        
        # Display point cloud using Open3D
        if points is not None and len(points) > 0:
            print(f"Original point cloud shape: {points.shape}")
            print(f"ROI bounds: {roi_bounds}")
            
            rgb_colors = None
            
            # Handle structured point cloud (H×W×3)
            if len(points.shape) == 3 and points.shape[2] == 3:
                H, W, _ = points.shape
                print(f"Structured point cloud: {H}×{W}×3")
                
                # Convert to (N, 3) format
                points_flat = points.reshape(-1, 3)
                
                # Get corresponding RGB colors if available
                if rgb is not None and rgb.shape[:2] == (H, W):
                    rgb_colors = rgb.reshape(-1, 3) / 255.0
                
            elif len(points.shape) == 2 and points.shape[1] == 3:
                # Already in (N, 3) format
                print(f"Unstructured point cloud: {points.shape}")
                points_flat = points
                
            else:
                print(f"Unsupported point cloud shape: {points.shape}")
                return
            
            # Filter out invalid points (zeros, NaN, inf)
            valid_mask = ~(np.isnan(points_flat).any(axis=1) | 
                          np.isinf(points_flat).any(axis=1) |
                          np.all(points_flat == 0, axis=1))
            
            # Apply ROI bounds filtering
            xmin, xmax, ymin, ymax, zmin, zmax = roi_bounds
            roi_mask = ((points_flat[:, 0] >= xmin) & (points_flat[:, 0] <= xmax) &
                       (points_flat[:, 1] >= ymin) & (points_flat[:, 1] <= ymax) &
                       (points_flat[:, 2] >= zmin) & (points_flat[:, 2] <= zmax))
            
            # Combine valid and ROI masks
            final_mask = valid_mask & roi_mask
            points_filtered = points_flat[final_mask]
            
            print(f"Valid points: {np.sum(valid_mask)}/{len(points_flat)}")
            print(f"Points in ROI: {np.sum(roi_mask)}/{len(points_flat)}")
            print(f"Final points to display: {len(points_filtered)}/{len(points_flat)}")
            
            if len(points_filtered) == 0:
                print("No valid points found in the specified ROI")
                return
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_filtered)
            
            # Add colors if RGB data is available
            if rgb_colors is not None:
                rgb_filtered = rgb_colors[final_mask]
                pcd.colors = o3d.utility.Vector3dVector(rgb_filtered)
                print("Added RGB colors to point cloud")
            
            # Estimate normals for better visualization
            pcd.estimate_normals()
            
            # Print point cloud statistics
            points_array = np.asarray(pcd.points)
            print(f"Point cloud bounds (after filtering):")
            print(f"  X: [{np.min(points_array[:, 0]):.3f}, {np.max(points_array[:, 0]):.3f}]")
            print(f"  Y: [{np.min(points_array[:, 1]):.3f}, {np.max(points_array[:, 1]):.3f}]")
            print(f"  Z: [{np.min(points_array[:, 2]):.3f}, {np.max(points_array[:, 2]):.3f}]")
            
            # Create ROI bounding box for visualization
            roi_corners = np.array([
                [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
                [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
            ])
            roi_lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            roi_bbox = o3d.geometry.LineSet()
            roi_bbox.points = o3d.utility.Vector3dVector(roi_corners)
            roi_bbox.lines = o3d.utility.Vector2iVector(roi_lines)
            roi_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(roi_lines))])  # Red color
            
            # Create coordinate frame at origin
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            
            # Create grid on XY plane (table plane)
            grid_lines = []
            grid_points = []
            grid_size = 1.0  # 1 meter grid
            grid_range = 1.0  # +/- 1 meter
            grid_step = 0.1  # 10cm intervals
            
            # Create grid lines parallel to X-axis
            for y in np.arange(-grid_range, grid_range + grid_step, grid_step):
                grid_points.extend([[xmin, y, 0], [xmax, y, 0]])
                grid_lines.append([len(grid_points)-2, len(grid_points)-1])
            
            # Create grid lines parallel to Y-axis  
            for x in np.arange(-grid_range, grid_range + grid_step, grid_step):
                grid_points.extend([[x, ymin, 0], [x, ymax, 0]])
                grid_lines.append([len(grid_points)-2, len(grid_points)-1])
            
            # Create grid line set
            grid = o3d.geometry.LineSet()
            grid.points = o3d.utility.Vector3dVector(grid_points)
            grid.lines = o3d.utility.Vector2iVector(grid_lines)
            grid.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for _ in range(len(grid_lines))])  # Light gray
            
            # Create scale rulers
            rulers = []
            
            # X-axis ruler (red)
            x_ruler_points = []
            x_ruler_lines = []
            for i, x in enumerate(np.arange(0, 1.0 + 0.1, 0.1)):
                x_ruler_points.extend([[x, 0, 0], [x, 0, 0.02]])  # Small vertical marks
                if i > 0:
                    x_ruler_lines.append([len(x_ruler_points)-2, len(x_ruler_points)-1])
                    x_ruler_lines.append([len(x_ruler_points)-4, len(x_ruler_points)-2])  # Connect to previous
            
            x_ruler = o3d.geometry.LineSet()
            x_ruler.points = o3d.utility.Vector3dVector(x_ruler_points)
            x_ruler.lines = o3d.utility.Vector2iVector(x_ruler_lines)
            x_ruler.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(x_ruler_lines))])  # Red
            rulers.append(x_ruler)
            
            # Y-axis ruler (green)
            y_ruler_points = []
            y_ruler_lines = []
            for i, y in enumerate(np.arange(0, 1.0 + 0.1, 0.1)):
                y_ruler_points.extend([[0, y, 0], [0.02, y, 0]])  # Small horizontal marks
                if i > 0:
                    y_ruler_lines.append([len(y_ruler_points)-2, len(y_ruler_points)-1])
                    y_ruler_lines.append([len(y_ruler_points)-4, len(y_ruler_points)-2])  # Connect to previous
            
            y_ruler = o3d.geometry.LineSet()
            y_ruler.points = o3d.utility.Vector3dVector(y_ruler_points)
            y_ruler.lines = o3d.utility.Vector2iVector(y_ruler_lines)
            y_ruler.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(y_ruler_lines))])  # Green
            rulers.append(y_ruler)
            
            # Z-axis ruler (blue)
            z_ruler_points = []
            z_ruler_lines = []
            for i, z in enumerate(np.arange(0, 1.0 + 0.1, 0.1)):
                z_ruler_points.extend([[0, 0, z], [0.02, 0, z]])  # Small marks along Z
                if i > 0:
                    z_ruler_lines.append([len(z_ruler_points)-2, len(z_ruler_points)-1])
                    z_ruler_lines.append([len(z_ruler_points)-4, len(z_ruler_points)-2])  # Connect to previous
            
            z_ruler = o3d.geometry.LineSet()
            z_ruler.points = o3d.utility.Vector3dVector(z_ruler_points)
            z_ruler.lines = o3d.utility.Vector2iVector(z_ruler_lines)
            z_ruler.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(z_ruler_lines))])  # Blue
            rulers.append(z_ruler)
            
            # Combine all geometries for visualization
            geometries = [pcd, roi_bbox, coord_frame, grid] + rulers
            
            # Visualize with Open3D
            print("Opening Open3D visualizer...")
            print("Controls: Mouse to rotate, Scroll to zoom, Ctrl+Mouse to pan")
            print("Red wireframe: ROI bounds")
            print("RGB axes: Coordinate frame (X=Red, Y=Green, Z=Blue)")
            print("Gray grid: 10cm intervals on table plane")
            print("Colored rulers: 10cm scale markers")
            o3d.visualization.draw_geometries(geometries, 
                                            window_name="Point Cloud with Coordinate System",
                                            width=1024, height=768)
        else:
            print("No point cloud data received")
            
    except ImportError as e:
        print(f"Visualization requires opencv-python and matplotlib: {e}")
    except Exception as e:
        print(f"Error in visualization: {e}")

async def test_rgb_and_points(uri="ws://localhost:8765"):
    """Test RGB and point cloud data reception and visualization"""
    print("\n=== Testing RGB and Point Cloud Data ===")
    
    client = WebSocketClient(uri)
    try:
        # Get data once
        rgb, points = await client.request_data_once()
        
        if rgb is not None and points is not None:
            print("✓ Successfully received RGB and point cloud data")
            print(f"  RGB shape: {rgb.shape}")
            print(f"  Points shape: {points.shape}")
            
            # Visualize the data with ROI bounds from config
            roi_bounds = [-0.2, 0.8, -0.2, 0.6, 0.0, 0.8]  # xmin, xmax, ymin, ymax, zmin, zmax
            visualize_rgb_and_points(rgb, points, roi_bounds)
            return True
        else:
            print("✗ Failed to receive RGB and point cloud data")
            return False
            
    except Exception as e:
        print(f"✗ Error testing RGB and points: {e}")
        return False

async def test_continuous_data(uri="ws://localhost:8765", duration=10):
    """Test continuous EEF pose and gripper action reception"""
    print(f"\n=== Testing Continuous Data Reception for {duration} seconds ===")
    
    client = WebSocketClient(uri)
    client.start()
    
    # Wait for connection
    await asyncio.sleep(1)
    
    if not client.is_connected():
        print("✗ Failed to establish continuous connection")
        return False
    
    print("✓ Continuous connection established")
    print("Monitoring EEF pose and gripper action (press Ctrl+C to stop)...")
    
    start_time = time.time()
    last_print_time = 0
    
    try:
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Print data every 0.5 seconds
            if current_time - last_print_time >= 0.5:
                eef_pose = client.get_eef_pose()
                gripper_action = client.get_gripper_action()
                
                # Print EEF position (first 3 elements of pose matrix)
                eef_pos = eef_pose[:3, 3]
                print(f"Time: {current_time - start_time:.1f}s | "
                      f"EEF Pos: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] | "
                      f"Gripper: {gripper_action}")
                
                last_print_time = current_time
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error during continuous monitoring: {e}")
    finally:
        client.stop()
        print("✓ Continuous monitoring stopped")
    
    return True

async def run_all_tests(server_uri="ws://localhost:8765"):
    """Run all tests sequentially"""
    print("🧪 WebSocket Client Test Suite")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    print("Test 1: Basic Connectivity")
    connectivity_result = await test_websocket_connection(server_uri)
    
    if not connectivity_result:
        print("\n❌ Basic connectivity failed. Skipping other tests.")
        return
    
    # Test 2: RGB and point cloud visualization
    print("\nTest 2: RGB and Point Cloud Data")
    rgb_points_result = await test_rgb_and_points(server_uri)
    
    # Test 3: Continuous data monitoring
    print("\nTest 3: Continuous Data Monitoring")
    continuous_result = await test_continuous_data(server_uri, duration=5)  # 5 seconds for demo
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Connectivity: {'✓ PASS' if connectivity_result else '✗ FAIL'}")
    print(f"  RGB & Points: {'✓ PASS' if rgb_points_result else '✗ FAIL'}")
    print(f"  Continuous:   {'✓ PASS' if continuous_result else '✗ FAIL'}")
    
    if all([connectivity_result, rgb_points_result, continuous_result]):
        print("\n🎉 All tests passed! WebSocket client is ready for use!")
    else:
        print("\n⚠️  Some tests failed. Please check your ROS2 WebSocket server.")

def main():
    parser = argparse.ArgumentParser(description="WebSocket Client")
    parser.add_argument("--server_host", default="localhost", 
                        help="WebSocket server host (default: localhost)")
    parser.add_argument("--server_port", type=int, default=8765,
                        help="WebSocket server port (default: 8765)")
    args = parser.parse_args()
    server_uri = f"ws://{args.server_host}:{args.server_port}"
    print(f"🌐 Server: {server_uri}")
    asyncio.run(run_all_tests(server_uri))

if __name__ == "__main__":
    main()