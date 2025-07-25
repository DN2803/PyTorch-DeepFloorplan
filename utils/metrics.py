import torch

def calculate_iou(pred, target, num_classes):
    # pred and target are expected to be class indices (long type)
    # pred: (N, H, W)
    # target: (N, H, W)

    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            iou_per_class.append(torch.tensor(float('nan')))  # Avoid division by zero
        else:
            iou_per_class.append(intersection / union)
    
    # Filter out NaN values (classes not present) and calculate mean IoU
    valid_ious = [iou for iou in iou_per_class if not torch.isnan(iou)]
    if len(valid_ious) == 0:
        return torch.tensor(0.0) # No valid classes found
    return torch.mean(torch.stack(valid_ious))

def calculate_dice(pred, target, num_classes):
    # pred and target are expected to be class indices (long type)
    # pred: (N, H, W)
    # target: (N, H, W)

    dice_per_class = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        sum_masks = pred_mask.sum().float() + target_mask.sum().float()

        if sum_masks == 0:
            dice_per_class.append(torch.tensor(float('nan'))) # Avoid division by zero
        else:
            dice_per_class.append((2. * intersection) / sum_masks)
    
    # Filter out NaN values (classes not present) and calculate mean Dice
    valid_dices = [dice for dice in dice_per_class if not torch.isnan(dice)]
    if len(valid_dices) == 0:
        return torch.tensor(0.0) # No valid classes found
    return torch.mean(torch.stack(valid_dices))
