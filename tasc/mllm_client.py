import os
import base64
import cv2
import numpy as np
import json
import time
import re

from openai import OpenAI
from utils import (get_config, extract_tag_content)
from typing import Tuple, List

global_config = get_config()
model_name = global_config['model_name']

class GPTClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=global_config['gpt_api_key'],
            base_url=global_config['gpt_base_url'],
        )
        file_path = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(file_path, "../temp")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def obtain_object_relationship(self, rgb: np.ndarray):
        """
        Prompt GPT to obtain the relationship between objects in the image.
        Args:
            rgb: rgb image
        """
        instruction = (
            "You will infer potential teleoperation tasks with a robotic arm from a single RGB image of a desktop scene.\n"
        )
        img_prompt = "# Task 1: Scene Description\n"
        img_prompt += "Describe each object in the scene: include name, color, and material (max 4 words).\n"
        img_prompt += "Let's think step by step following the pattern:\n"
        img_prompt += "image = '...'  # Describe the image\n"
        img_prompt += "output: <output1>\nIndex-Name\n...\n</output1>\n"
        img_prompt += "## Example:\n<output1>\n1-Silver mobile phone\n2-Cup with water\n</output1>\n\n"

        img_prompt += "# Task 2: Object Interaction Pairs\n"
        img_prompt += "List all plausible interaction pairs based on common affordances.\n"
        img_prompt += "Rules:\n"
        img_prompt += "1. Assume the teleoperator first grasps one object (the 'actor').\n"
        img_prompt += "2. The actor interacts with a second object (the 'target').\n"
        img_prompt += "3. Summarize the action using 1–2 verb words.\n"
        img_prompt += "4. There may be multiple plausible interaction pairs, list all reasonable options.\n"
        img_prompt += "5. Use realistic affordances based on daily tasks, ignore unsafe or impossible actions.\n"
        img_prompt += "6. Do not limit to tool use; also consider placing, inserting, stacking, etc.\n"
        img_prompt += "output: <output2>\nActorIndex-action-TargetIndex\n...\n</output2>\n"
        img_prompt += "## Example:\n<output2>\n4-pour-2  (Teapot pour into cup)\n</output2>"

        save_path = os.path.join(self.cache_dir, f"scene_{int(time.time())}.png")
        cv2.imwrite(save_path, rgb)

        b64_image = base64.b64encode(cv2.imencode(".jpg", rgb)[1]).decode("utf-8")
        messages = [{
            "role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                {"type": "text", "text": instruction + img_prompt},
            ],
        }]
        gpt_return_result = self.client.chat.completions.create(model=model_name, messages=messages)
        return messages, gpt_return_result
    
    def obtain_object_relationship_auto(
        self, rgb: np.ndarray, max_retries: int = 3
    ) -> Tuple[list, list]:
        """
        Automatically obtain object relationships from the image using GPT.
        Args:
            rgb: rgb image
            max_retries: maximum number of retries for GPT call

        Returns:
            Tuple containing the messages and the result from GPT.
        """
        for i in range(max_retries):
            try:
                messages, result = self.obtain_object_relationship(rgb)
                o1_firstframe = extract_tag_content(result.choices[0].message.content, "output1")
                o2_firstframe = extract_tag_content(result.choices[0].message.content, "output2")
                # Return both the messages and the result contacted. This is useful for future prompting.
                messages = messages.copy()
                messages.append({"role": "user", "content": result.choices[0].message.content})
                # Save the results
                save_data = {
                    "output1": o1_firstframe,
                    "output2": o2_firstframe,
                    "messages": messages
                }
                save_path = os.path.join(self.cache_dir, f"gpt_result_{int(time.time())}.json")
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                
                obj_info = self.parse_obj_info(o1_firstframe)
                obj_relation = self.parse_obj_relation(o2_firstframe)
                
                return obj_info, obj_relation, messages
            except Exception as e:
                print(f"Error in GPT call. Retrying. Error: {e}")
        raise RuntimeError("GPT call failed.")
    
    @staticmethod
    def parse_obj_info(obj_info_text: str) -> List[Tuple[int, str]]:
        """
        Parse <output1> section into list of (index, object description).
        """
        lines = obj_info_text.strip().splitlines()
        result = []
        for line in lines:
            if '-' in line:
                try:
                    index, desc = line.strip().split('-', 1)
                    result.append((int(index.strip()), desc.strip()))
                except ValueError:
                    continue  # Skip malformed lines
        return result

    @staticmethod
    def parse_obj_relation(parse_obj_relation: str) -> List[Tuple[int, str, int]]:
        """
        Parse <output2> section into list of (actor_index, action, target_index).
        """
        lines = parse_obj_relation.strip().splitlines()
        result = []
        for line in lines:
            match = re.match(r"(\d+)-([a-zA-Z_]+)-(\d+)", line.strip())
            if match:
                actor, action, target = match.groups()
                result.append((int(actor), int(target), action))
        return result

    @staticmethod
    def build_pose_constraint_prompt(obj_a_name, obj_b_name, action, image_name_a, image_name_b):
        task_description = f'"{obj_a_name}" {action}s "{obj_b_name}"'
        prompt = f"""Please analyze the spatial constraint relationship between the two objects involved in the robot arm manipulation task.
                    ## Input
                    1. Two RGB images [{image_name_a} (objcet a is grasped by robotic arm), {image_name_b} (object b is the target object)], each showing an object from two viewpoints:
                    - Left: top view (showing the X-Y axes of the object)
                    - Right: front view (showing the X-Z axes of the object)
                    The three principal axes (X, Y, Z) of each object are clearly marked in red, green, and blue, respectively.
                    (The axes follow right-handed convention. The local Z-axis of each object is approximately upright when placed stably.)
                    2. Task description indicating how the two objects interact.

                    ## Goal
                    1. Understand the distribution of coordinate axes, that is, the direction in which the X, Y, and Z axes of the object pass through the object respectively;
                    2. Provide the constraints between the main axes of the two objects in order to complete the operation task.

                    ## Output Format
                    [constraint_1, ..., constraint_N]
                    Where constraint_n = (obj1_axis_idx, obj2_axis_idx, alignment_sign), obj_axis_idx ∈ {{0, 1, 2}} represent the xyz axis respectively, and alignment_sign ∈ {{1, -1}} represent alignment or reverse alignment respectively.

                    ## Notes
                    1. Output only the minimal necessary constraints to perform the task.
                    2. Empty list is acceptable if no specific constraint is needed.
                    2. Each object has a local coordinate frame, with X, Y, Z axes labeled on the RGB image. The Z-axis of all objects is approximately upward when placed stably on the table, aligned with the world Z-axis.
                    The orientation of the axes is consistent with standard right-handed conventions.

                    ## Example Input
                    Two RGB images: obb_a_3.png, obb_b_1.png
                    Task description: "hammer with wooden handle" hits "wooden block with peg"

                    ## Example Output
                    "<output>[[1, 2, -1],]</output>"

                    ## Task Input
                    Two RGB images: {image_name_a}, {image_name_b}
                    Task description: {task_description}
                    """
        return prompt
    
    def obtain_pose_constraints(self, rgb_a: np.ndarray, rgb_b: np.ndarray, obj_a_name: str, obj_b_name: str, action: str):
        """
        Use GPT to infer pose axis constraints between two objects.
        Args:
            rgb_a: RGB image for object A (grasped)
            rgb_b: RGB image for object B (target)
            obj_a_name: object A name (e.g. "hammer with wooden handle")
            obj_b_name: object B name (e.g. "wooden block with peg")
            action: action verb (e.g. "hit")
        Returns:
            messages: List of messages sent to GPT
            result: GPT response result
        """
        # Process images - resize if too large
        def process_image(img):
            h, w = img.shape[:2]
            max_size = 1024  # Max dimension
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return img
        rgb_a = process_image(rgb_a)
        rgb_b = process_image(rgb_b)
        ts = int(time.time())
        name_a, name_b = f"obb_a_{ts}.png", f"obb_b_{ts}.png"
        path_a, path_b = os.path.join(self.cache_dir, name_a), os.path.join(self.cache_dir, name_b)
        cv2.imwrite(path_a, rgb_a)
        cv2.imwrite(path_b, rgb_b)
        
        prompt = self.build_pose_constraint_prompt(obj_a_name, obj_b_name, action, name_a, name_b)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', rgb_a)[1]).decode()}" }},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', rgb_b)[1]).decode()}" }},
                {"type": "text", "text": prompt}
            ]
        }]
        result = self.client.chat.completions.create(model=model_name, messages=messages)
        return messages, result

    def obtain_pose_constraints_auto(
        self, rgb_a: np.ndarray, rgb_b: np.ndarray, obj_a_name: str, obj_b_name: str, action: str, max_retries: int = 3
    ) -> Tuple[list, list, list]:
        """
        Automatically obtain pose constraints between two objects using GPT.
        Args:
            rgb_a: RGB image for object A (grasped)
            rgb_b: RGB image for object B (target)
            obj_a_name: object A name (e.g. "hammer with wooden handle")
            obj_b_name: object B name (e.g. "wooden block with peg")
            action: action verb (e.g. "hit")
            max_retries: maximum number of retries for GPT call
        Returns:
            Tuple containing the constraints, messages and content.
        """
        for i in range(max_retries):
            try:
                messages, result = self.obtain_pose_constraints(rgb_a, rgb_b, obj_a_name, obj_b_name, action)
                content = result.choices[0].message.content
                constraint_output = extract_tag_content(content, "output")
                # Parse constraints
                constraints = self.parse_pose_constraints(constraint_output, content)
                # Update messages with result
                messages = messages.copy()
                messages.append({"role": "assistant", "content": content})
                # Save the results
                save_data = {
                    "obj_a_name": obj_a_name,
                    "obj_b_name": obj_b_name,
                    "action": action,
                    "constraints": constraints,
                    "raw_output": constraint_output,
                    "full_response": content,
                    "messages": messages
                }
                save_path = os.path.join(self.cache_dir, f"pose_constraints_{int(time.time())}.json")
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                return constraints, messages, content
            except Exception as e:
                print(f"Error in GPT call for pose constraints. Retrying. Error: {e}")
        raise RuntimeError("GPT call for pose constraints failed.")

    @staticmethod
    def parse_pose_constraints(constraint_str: str, full_content: str = None) -> list:
        """
        Parse constraint output into list of constraints.
        Args:
            constraint_str: String containing constraints from <output> tag, e.g. "[[1, 2, -1]]"
            full_content: Full GPT response content for fallback parsing
        Returns:
            List of constraints, e.g. [[1, 2, -1]]
        """
        # If constraint_str is None, try to extract from full_content
        if constraint_str is None and full_content is not None:
            import re
            # Pattern 1: Look for boxed format like $\boxed{[[1, 2, -1]]}$
            pattern = r'\$\\boxed\{(\[\[.*?\]\])\}\$'
            matches = re.findall(pattern, full_content)
            if matches:
                constraint_str = matches[0]
            else:
                # Pattern 2: Look for Python code block
                pattern = r'```python\s*(\[\[.*?\]\])\s*```'
                matches = re.findall(pattern, full_content, re.DOTALL)
                if matches:
                    constraint_str = matches[0].strip()
                else:
                    # Pattern 3: Look for nested list pattern
                    pattern = r'\[\s*\[.*?\]\s*\]'
                    matches = re.findall(pattern, full_content, re.DOTALL)
                    if matches:
                        constraint_str = matches[0]
                    else:
                        # Pattern 4: More lenient list pattern
                        pattern = r'\[.*?\]'
                        matches = re.findall(pattern, full_content)
                        if matches:
                            # Look for the most list-like match
                            for match in matches:
                                if ',' in match or match.count('[') > 1:
                                    constraint_str = match
                                    break               
                        if constraint_str is None:
                            constraint_str = "[]"
        if constraint_str is None or not constraint_str.strip():
            return []
        # Clean up the input string
        constraint_str = constraint_str.strip()
        # Try to use json.loads first (safer than eval)
        try:
            import json
            constraints = json.loads(constraint_str)
        except:
            # Fallback to eval if json fails
            try:
                constraints = eval(constraint_str)
            except Exception as e:
                print(f"Failed to parse constraint string: {constraint_str}")
                print(f"Error: {e}")
                return []
        # Validate format
        if not isinstance(constraints, list):
            print(f"Constraints should be a list, got: {type(constraints)}")
            return [] 
        validated_constraints = []
        for constraint in constraints:
            if (isinstance(constraint, list) and len(constraint) == 3 and
                all(isinstance(x, int) for x in constraint) and
                constraint[0] in [0, 1, 2] and constraint[1] in [0, 1, 2] and
                constraint[2] in [-1, 1]):
                validated_constraints.append(constraint)
            else:
                print(f"Skipping invalid constraint format: {constraint}")
        return validated_constraints
