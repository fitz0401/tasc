"""
Adapted from: https://github.com/moka-manipulation/moka/blob/main/moka/vision/segmentation.py
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import groundingdino.datasets.transforms as T
from segment_anything import build_sam, SamPredictor
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, predict, annotate
from torchvision.ops import nms
from PIL import Image

file_path = os.path.dirname(os.path.abspath(__file__))


class MaskGenerator:
    def __init__(self, config):
        self.config = config
        self.ckpt_path = os.path.join(file_path, "../../ckpts")
        self.config_path = os.path.join(file_path, "../../configs")
        self.device = self.config['device']
        self.dino, self.sam = self._prepare_dino_and_sam()

    def _prepare_dino_and_sam(self):
        DINO_WEIGHTS_PATH = os.path.join(self.ckpt_path, "groundingdino_swint_ogc.pth")
        DINO_CONFIG_PATH = os.path.join(self.config_path, "grounding_dino.py")
        dino = load_model(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH)
        SAM_WEIGHTS_PATH = os.path.join(self.ckpt_path, "sam_vit_h_4b8939.pth")
        sam = build_sam(checkpoint=SAM_WEIGHTS_PATH)
        sam.to(self.device)
        return dino, sam
    
    def _load_pil_image(self, image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_np = np.asarray(image_pil)
        image_transformed, _ = transform(image_pil, None)
        self.H, self.W, _ = image_np.shape
        return image_np, image_transformed

    @staticmethod
    def _show_mask(masks, frame, random_color=True):
        annotated_frame_pil = Image.fromarray(frame).convert("RGBA")
        for i in range(masks.shape[0]):
            mask = masks[i][0]
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
            annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)
        return np.array(annotated_frame_pil)
    
    def get_scene_object_bboxes(self, image, texts, gripper_region=None, visualize=False, logdir=None):
        # mask out gripper region
        image_np, image_transformed = self._load_pil_image(image)
        if gripper_region is None:
            gripper_region = self.config['vision_module']['gripper_region']
            y1, x1, y2, x2 = gripper_region
            image_transformed[:, y1:y2, x1:x2] = 0     # [C, H, W]
        # get bounding box
        BOX_THRESHOLD = 0.3
        TEXT_THRESHOLD = 0.22 # can be tuned for different tasks
        all_boxes, all_logits, all_phrases = [], [], []
        for text in texts:
            boxes, logits, phrases = predict(
                model=self.dino,
                image=image_transformed,
                caption=text[1],
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            id = torch.argmax(logits)
            all_boxes.append(boxes[id])
            all_logits.append(logits[id])
            all_phrases.append(phrases[id])
        boxes, logits, phrases = torch.stack(all_boxes, dim=0), torch.stack(all_logits, dim=0), all_phrases
        
        if visualize:
            annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
            # %matplotlib inline
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB
            annotated_frame_pil = Image.fromarray(annotated_frame)
            if logdir is not None:
                annotated_frame_pil.save(os.path.join(logdir, 'bbox.png'))
            plt.imshow(annotated_frame_pil)
            plt.show()
        return boxes, logits, phrases

    def filter_masks(self, boxes, scores, phrases, iou_threshold=0.5):
        # filter out masks by IOU
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([self.W, self.H, self.W, self.H])
        keep_indices = nms(boxes_xyxy, scores, iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        phrases = [phrases[i] for i in keep_indices.tolist()]
        return boxes, scores, phrases

    def get_segmentation_masks(self, image, boxes, logits, phrases, visualize=False, save_path=None):
        image_np, _ = self._load_pil_image(image)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([self.W, self.H, self.W, self.H])
        # get segmentation based on bounding box
        sam_predictor = SamPredictor(self.sam)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2]).to(self.device)
        sam_predictor.set_image(image_np)
        with torch.no_grad():
            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            masks = masks.cpu().numpy()
        if visualize:
            annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
            # %matplotlib inline
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB
            mask_annotated_img = self._show_mask(masks, annotated_frame)
            mask_annotated_img_pil = Image.fromarray(mask_annotated_img)
            plt.imshow(mask_annotated_img_pil)
            plt.show()
            if save_path is not None:
                mask_annotated_img_pil.save(save_path)
        return masks[:, 0]