import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from tasc.utils import get_logger
from tasc.goal_predictor.object_node import ObjectNode
from tasc.goal_predictor.prediction_utils import ApplyTwistToTransform
from scipy.special import logsumexp
from collections import deque
from typing import Optional, Tuple


logger = get_logger("tasc")
weight_sc = 0.95
max_prob_any_goal = 0.99
log_max_prob_any_goal = np.log(max_prob_any_goal)
window_size = 10
ACTION_APPLY_TIME = 0.05  # used when calculating future state


class SceneGraph:
    def __init__(self, objects_list=None, interaction_list=None, eef_pose=None):
        self.nodes_dict = {}
        if objects_list is not None:
            for obj_id, obj_name in objects_list:
                self.add_node(obj_id, ObjectNode(obj_name))
        if interaction_list is not None:
            for id_src, id_tgt, relation in interaction_list:
                self.add_relation(id_src, id_tgt, relation)
        self._init_prediction_policies(eef_pose)

    def add_node(self, node_id: int, object_node: ObjectNode):
        if node_id in self.nodes_dict:
            raise ValueError(f"Object {node_id} already exists.")
        self.nodes_dict[node_id] = object_node

    def add_relation(self, id_src, id_tgt, relation):
        if id_src not in self.nodes_dict or id_tgt not in self.nodes_dict:
            raise ValueError("Both objects must exist in the scene graph.")
        self.nodes_dict[id_src].add_relation(id_tgt, relation)

    def get_node_by_id(self, id) -> Optional[ObjectNode]:
        return self.nodes_dict.get(id)
    
    def get_node_by_name(self, name):
        for node_id, obj_node in self.nodes_dict.items():
            if obj_node.name == name:
                return obj_node
        logger.warning(f"Node with name {name} not found.")
        return None

    def get_relations(self, id, other_id=None):
        if other_id is None:
            if id not in self.nodes_dict:
                logger.warning(f"Object {id} does not exist.")
                return None
            return self.nodes_dict[id].relations
        else:
            if id not in self.nodes_dict or other_id not in self.nodes_dict:
                logger.warning(f"Both objects {id} and {other_id} must exist.")
                return None, None
            return self.nodes_dict[id].relations.get(other_id, (None, None))
    
    def get_visualization(self, width, height, use_pixel_center=False):
        G = nx.DiGraph()
        # Add nodes
        def wrap_label(text, max_line_length=15):
            words = text.split()
            lines = []
            current = ""
            for w in words:
                if len(current) + len(w) + 1 > max_line_length:
                    lines.append(current.rstrip())
                    current = ""
                current += w + " "
            if current:
                lines.append(current.rstrip())
            return "\n".join(lines)
        for node_id, obj_node in self.nodes_dict.items():
            G.add_node(node_id, label=wrap_label(obj_node.name))
        # Add edges
        for node_id, obj_node in self.nodes_dict.items():
            for other_id, relation in obj_node.relations.items():
                G.add_edge(node_id, other_id, label=relation[0])

        if use_pixel_center:
            pos = {
                node_id: (obj_node.pixel_center[0], height - obj_node.pixel_center[1])  # Flip y for correct orientation
                for node_id, obj_node in self.nodes_dict.items() if obj_node.pixel_center is not None
            }
        else:
            pos = nx.spring_layout(G, k=2.0, seed=42)  # Other layout: shell_layout, kamada_kawai_layout
        labels = nx.get_node_attributes(G, 'label')
        edge_labels = nx.get_edge_attributes(G, 'label')
        node_colors = []
        node_sizes = []
        for node_id, obj_node in self.nodes_dict.items():
            if obj_node.is_active:
                node_colors.append((0.3, 0.6, 1.0, 1.0))  # vivid blue, opaque
            elif obj_node.is_grasped:
                node_colors.append((1.0, 0.0, 0.0, 1.0))  # red, opaque
            else:
                node_colors.append((0.6, 0.6, 0.6, 0.4))  # greyed out, semi-transparent
            node_sizes.append(1200)

        fig = plt.figure(figsize=(width / 100, height / 100))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=30)
        nx.draw_networkx_labels(G, pos, labels, font_size=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=16)
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)
        plt.close(fig)
        return img_array
    
    ## Goal prediction policies
    def _init_prediction_policies(self, eef_pose=None):
        self.ind_max = 0
        self.active_node_ids = []
        self.log_goal_distribution = np.array([])
        self.vq_history = deque(maxlen=window_size)
        distances = []
        for obj_id, obj_node in self.nodes_dict.items():
            if not obj_node.is_active or obj_node.is_grasped:
                continue
            self.active_node_ids.append(obj_id)
            obj_node.init_prediction_policy()
            if eef_pose is not None:
                obj_pos = obj_node.center_point
                dist = np.linalg.norm(eef_pose[:3, 3] - obj_pos)
                distances.append(dist)
        if len(self.active_node_ids) > 0:
            if eef_pose is None:
                self.log_goal_distribution = np.log((1./len(self.active_node_ids)) * 
                                                    np.ones(len(self.active_node_ids)))  # log( p(g|xi) )'
            else:
                distances = np.array(distances)
                weights = np.exp(-distances)
                probs = weights / np.sum(weights)
                self.log_goal_distribution = np.log(probs)

    def update_prediction_policies(self, eef_pose, user_action):
        eef_pose_after_action = ApplyTwistToTransform(user_action, eef_pose, ACTION_APPLY_TIME)
        for obj_id in self.active_node_ids:
            obj_node = self.nodes_dict[obj_id]
            obj_node.update_prediction_policy(eef_pose, eef_pose_after_action)
        # update values and q_values
        v_values = np.ndarray(len(self.active_node_ids))
        q_values = np.ndarray(len(self.active_node_ids))
        for idx, obj_id in enumerate(self.active_node_ids):
            obj_node = self.nodes_dict[obj_id]
            v_values[idx] = obj_node.get_value()
            q_values[idx] = obj_node.get_qvalue()
        # update log distribution: V(x; g) - Q(x, u; g) = V(x; g) - C(x, u) - V(x'; g)
        # log( p(xi|g) ) = log( p(xi|g) )@t-1 + log( exp(V - Q) )@t
        self.log_goal_distribution *= weight_sc
        self.log_goal_distribution += v_values - q_values
        # self.vq_history.append(v_values - q_values)
        # self.log_goal_distribution = np.mean(self.vq_history, axis=0)
        # normalize log distribution
        # log( sum( p(xi|g) ) )
        log_normalization_val = logsumexp(self.log_goal_distribution)
        # log( p(g|xi) )_i = log( p(xi|g) )_i - log( sum( p(xi|g) ) )
        self.log_goal_distribution -= log_normalization_val
        self._clip_prob()
        first_max, second_max = self._get_index_max()
        self.ind_max = first_max
    
    def _clip_prob(self):
        if len(self.log_goal_distribution) <= 1:
            return
        #check if any too high
        max_prob_ind = np.argmax(self.log_goal_distribution)
        if self.log_goal_distribution[max_prob_ind] > log_max_prob_any_goal:
            #see how much we will remove from probability
            diff = np.exp(self.log_goal_distribution[max_prob_ind]) - max_prob_any_goal
            #want to distribute this evenly among other goals
            diff_per = diff / (len(self.log_goal_distribution) - 1.)
            #distribute this evenly in the probability space...this corresponds to doing so in log space
            # e^x_new = e^x_old + diff_per, and this is formulate for log addition
            self.log_goal_distribution += np.log( 1. + diff_per/np.exp(self.log_goal_distribution))
            #set old one
            self.log_goal_distribution[max_prob_ind] = log_max_prob_any_goal
    
    def _get_index_max(self):
        amax = np.argmax(self.log_goal_distribution)
        mask = np.zeros_like(self.log_goal_distribution, dtype=bool)
        mask[amax] = True
        second_max = np.argmax(np.ma.masked_array(self.log_goal_distribution, mask=mask))
        max_id = self.active_node_ids[amax]
        second_max_id = self.active_node_ids[second_max]
        return max_id, second_max_id

    def get_goal_distribution(self):
        goal_distribution = np.exp(self.log_goal_distribution)
        distribution_with_ids = {node_id: goal_distribution[i] for i, node_id in enumerate(self.active_node_ids)}
        return distribution_with_ids
    
    def get_goal_node(self) -> Tuple[int, ObjectNode]:
        if len(self.active_node_ids) == 0 or self.ind_max not in self.nodes_dict:
            logger.warning("No active goal node found in the scene graph.")
            return None, None
        goal_obj_node = self.nodes_dict[self.ind_max]
        return self.ind_max, goal_obj_node
    
    def get_grasped_node(self) -> Tuple[int, ObjectNode]:
        for obj_id, obj_node in self.nodes_dict.items():
            if obj_node.is_grasped:
                return obj_id, obj_node
        logger.warning("No object is currently grasped.")
        return None, None
    
    def get_nearest_node(self, eef_pose) -> Tuple[int, ObjectNode]:
        nearest_node_id = None
        if len(self.nodes_dict) == 0:
            logger.warning("No nodes in the scene graph to find the nearest node.")
            return None, None
        nearest_distance = np.inf
        for obj_id, obj_node in self.nodes_dict.items():
            distance = np.linalg.norm(obj_node.center_point - eef_pose[:3, 3])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node_id = obj_id
        return nearest_node_id, self.nodes_dict[nearest_node_id]
    
    def print_probability(self):
        for i, obj_id in enumerate(self.active_node_ids):
            obj_node = self.nodes_dict[obj_id]
            prob = np.exp(self.log_goal_distribution[i])
            logger.info(f"Object {obj_node.name} (ID: {obj_id}) has probability: {prob:.4f}")

    def get_leaf_nodes(self):
        in_degree = {node_id: 0 for node_id in self.nodes_dict}
        out_degree = {node_id: 0 for node_id in self.nodes_dict}
        for src_id, node in self.nodes_dict.items():
            for tgt_id in node.relations:
                out_degree[src_id] += 1
                in_degree[tgt_id] += 1
        non_active_nodes_ids = []
        for node_id in self.nodes_dict:
            if out_degree[node_id] == 0 and in_degree[node_id] > 0:
                non_active_nodes_ids.append(node_id)
        return non_active_nodes_ids

    ## Stage change
    def update_stage(self, assistance_mode='init', eef_pose=None):
        if assistance_mode in ('reset', 'init'):
            non_active_nodes_ids = self.get_leaf_nodes()
            for obj_id, obj_node in self.nodes_dict.items():
                if obj_id in non_active_nodes_ids:
                    obj_node.is_active = False
                else:
                    obj_node.is_active = True
                obj_node.is_grasped = False
        elif assistance_mode == 'manipulation_assistance':
            grasped_node_id, _ = self.get_grasped_node()
            if grasped_node_id is not None:
                relations = self.get_relations(grasped_node_id)
                logger.info(f"Relations: {relations}")
                for obj_id, obj_node in self.nodes_dict.items():
                    if obj_id in relations:
                        obj_node.is_active = True
                    else:
                        obj_node.is_active = False
        else:
            return
        self._init_prediction_policies(eef_pose)
        
    def __repr__(self):
        return "\n".join(f"ID={node_id}, {node}" for node_id, node in self.nodes_dict.items())