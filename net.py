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
        self.room_type_head = nn.Conv2d(self.mask2former.config.num_queries, 9, kernel_size=1)
        self.boundary_head = nn.Conv2d(self.mask2former.config.num_queries, 3, kernel_size=1)
    


    def forward(self,x):
        # Resize input to Mask2Former expected size (e.g., 512x512 or 1024x1024)
        # Mask2Former expects input in the range [0, 1] and normalized
        # Assuming x is already normalized to [0, 1]
        original_size = x.shape[-2:]
        
        # Mask2FormerForUniversalSegmentation expects pixel_values as input
        outputs = self.mask2former(pixel_values=x)

        # Mask2Former outputs: pred_masks (logits for masks), pred_logits (logits for classes)
        # pred_masks: (batch_size, num_queries, H_mask, W_mask)
        # pred_logits: (batch_size, num_queries, num_labels)

        # We need to convert these to the original model's output format:
        # logits_r (room types): (batch_size, 9, H, W)
        # logits_cw (boundaries): (batch_size, 3, H, W)

        # For simplicity, let's take the predicted masks and apply our heads
        # First, resize pred_masks to original_size if necessary
        pred_masks = F.interpolate(outputs.masks_queries_logits, size=original_size, mode="bilinear", align_corners=False)

        # Apply the custom heads to get the desired output channels
        # Note: This is a simplified mapping. A more sophisticated approach might involve
        #       interpreting pred_logits to select relevant masks or combining them.
        logits_r = self.room_type_head(pred_masks)
        logits_cw = self.boundary_head(pred_masks)

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


