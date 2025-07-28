from importmod import *
from utils.util import get_device
import matplotlib.pyplot as plt
from deploy import BCHW2colormap, post_process
from transformers import Mask2FormerForUniversalSegmentation

class DFPmodel(torch.nn.Module):
    def __init__(self,pretrained=True,freeze=True):
        super(DFPmodel,self).__init__()
        self.device = get_device()

        # Initialize Mask2Former
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-panoptic")

        # Define output layers to match original model's output dimensions
        # Mask2Former typically outputs masks and class logits. We need to map these to room types and boundaries.
        # Assuming Mask2Former outputs masks of size [batch_size, num_queries, H, W]
        # and class logits of size [batch_size, num_queries, num_classes]
        # We need to map these to [batch_size, num_room_types, H, W] and [batch_size, num_boundary_types, H, W]
        # For simplicity, let's assume num_queries can be mapped to num_room_types and num_boundary_types.
        # The original model outputs 9 channels for room types and 3 for boundaries.
        C= 256
        self.room_type_head = nn.Conv2d(C, 9, kernel_size=1)
        self.boundary_head = nn.Conv2d(C, 3, kernel_size=1)
    


    def forward(self, x):
        original_size = x.shape[-2:]
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, C, H, W)
        if x.shape[1] == 3 and x.max() > 1:
            x = x / 255.0

        x = x.to(self.device)

        # Mask2Former expects pixel_values (normalized)
        outputs = self.mask2former(pixel_values=x)

        # outputs.pred_masks: (B, num_queries, H_mask, W_mask)
        # outputs.pred_logits: (B, num_queries, num_classes)
        pred_masks = outputs.masks_queries_logits  # Shape: (B, Q, H, W)
        pred_logits = outputs.class_queries_logits  # (B, Q, C)

        # Convert pred_logits to softmax to get confidence per query
        class_confidence = torch.softmax(pred_logits, dim=-1)  # (B, Q, C)
        confidence_scores, predicted_classes = class_confidence.max(dim=-1)  # (B, Q)

        # Select top-k masks by confidence
        top_k = 10
        topk_values, topk_indices = torch.topk(confidence_scores, k=top_k, dim=1)  # (B, k)

        batch_size, _, H, W = pred_masks.shape
        aggregated_mask = torch.zeros((batch_size, 1, H, W), device=self.device)

        for b in range(batch_size):
            selected_masks = pred_masks[b][topk_indices[b]]  # (k, H, W)
            combined_mask = selected_masks.mean(dim=0).unsqueeze(0)  # (1, H, W)
            aggregated_mask[b] = combined_mask

        features = outputs.pixel_decoder_last_hidden_state  # (B, C, H, W)
        # Upsample hoặc dùng Conv2d để chuyển thành segmentation
        logits_r = self.room_type_head(features)
        logits_cw = self.boundary_head(features)

        return logits_r, logits_cw


if __name__ == "__main__":

    with torch.no_grad():
        
        sample = np.load("")
        image = sample["image"]
        model_input = torch.Tensor(image.astype(np.float32) / 255.0)

        model = DFPmodel()
        model.load_state_dict(torch.load('log/store2/checkpoint.pt'))
        
        model.eval()
        logits_r,logits_cw = model(model_input)
        predroom = BCHW2colormap(logits_r)
        predboundary = BCHW2colormap(logits_cw)
        predroom = post_process(predroom,predboundary)
        rgb = ind2rgb(predroom,color_map=floorplan_fuse_map)
        plt.subplot(1,3,1); plt.imshow(image[:,:,::-1])
        plt.subplot(1,3,2); plt.imshow(rgb)
        plt.subplot(1,3,3); plt.imshow(predboundary)
        plt.show()


