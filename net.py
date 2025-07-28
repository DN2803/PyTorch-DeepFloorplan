from importmod import *
from utils.util import get_device
import matplotlib.pyplot as plt
from deploy import BCHW2colormap, post_process
from transformers import Mask2FormerForUniversalSegmentation
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DFPmodel(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(DFPmodel, self).__init__()
        self.device = get_device()

        # Load pretrained Mask2Former
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-coco-panoptic"
        )

        # Output feature dimension from pixel decoder (default = 256)
        self.C = 256  # Swin-small uses 256 as pixel decoder output

        # Room type and boundary heads using Conv2d on feature map
        self.room_type_head = nn.Conv2d(self.C, 9, kernel_size=1)
        self.boundary_head = nn.Conv2d(self.C, 3, kernel_size=1)

    def forward(self, x):
        original_size = x.shape[-2:]

        # Preprocess input
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 3 and x.max() > 1:
            x = x / 255.0

        x = x.to(self.device)

        # Pass through Mask2Former
        outputs = self.mask2former(pixel_values=x)

        # Use feature map from pixel decoder
        features = outputs.pixel_decoder_last_hidden_state  # Shape: (B, C, H, W)

        # Segmentation heads
        logits_r = self.room_type_head(features)     # Room type logits
        logits_cw = self.boundary_head(features)     # Boundary logits

        # Optional: resize to match original input size if needed
        if logits_r.shape[-2:] != original_size:
            logits_r = F.interpolate(logits_r, size=original_size, mode='bilinear', align_corners=False)
            logits_cw = F.interpolate(logits_cw, size=original_size, mode='bilinear', align_corners=False)

        return logits_r, logits_cw

if __name__ == "__main__":
    with torch.no_grad():
        sample = np.load("")  # Add correct file path
        image = sample["image"]
        model_input = torch.Tensor(image.astype(np.float32)).permute(2, 0, 1)  # Convert to (C, H, W)

        model = DFPmodel()
        model.load_state_dict(torch.load('log/store2/checkpoint.pt', map_location='cpu'))  # Modify map_location if needed

        model.eval()
        logits_r, logits_cw = model(model_input)
        predroom = BCHW2colormap(logits_r)
        predboundary = BCHW2colormap(logits_cw)
        predroom = post_process(predroom, predboundary)
        rgb = ind2rgb(predroom, color_map=floorplan_fuse_map)

        plt.subplot(1, 3, 1)
        plt.imshow(image[:, :, ::-1])
        plt.subplot(1, 3, 2)
        plt.imshow(rgb)
        plt.subplot(1, 3, 3)
        plt.imshow(predboundary)
        plt.show()
