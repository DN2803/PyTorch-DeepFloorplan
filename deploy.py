import sys
from utils.rgb_ind_convertor import *
from utils.util import get_device, fill_break_line, flood_fill, refine_room_region
from utils import *
import cv2
from net import *
from data import *
import argparse 
import matplotlib.pyplot as plt

def BCHW2colormap(tensor,earlyexit=False):
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    result = tensor.squeeze().permute(1,2,0).cpu().detach().numpy()
    if earlyexit:
        return result
    result = np.argmax(result,axis=2)
    return result

def initialize(args):
    # device
    device = get_device()
    # data
    trans = transforms.Compose([transforms.ToTensor()])
    orig = cv2.imread(args.image_path)
    orig = cv2.resize(orig,(512,512))
    image = trans(orig.astype(np.float32)/255.)
    image = image.unsqueeze(0).to(device)
    # model
    model = DFPmodel()
    model.load_state_dict(torch.load(args.loadmodel, map_location=device))
    model.to(device)
    return device,orig,image,model

def post_process(rm_ind,bd_ind):
    hard_c = (bd_ind>0).astype(np.uint8)
    # region from room prediction 
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind>0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)

    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask>=1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask//255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask,rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask*new_rm_ind

    return new_rm_ind
def init_detect_model(args): 
    
    # dnn model
    if args.dnn_model == 'yolo':
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package is not installed.")
        print(f"Loading YOLO model from {args.dnn_model_path}")
        model = YOLO(args.dnn_model_path)
        return model
import matplotlib.patches as mpatches
import random

def get_color_palette(num_classes):
    random.seed(42)  # để màu ổn định giữa các lần
    return {
        i: tuple([random.randint(0, 255) for _ in range(3)])  # (B, G, R)
        for i in range(num_classes)
    }

def draw_detection(orig, boxes, scores, labels, label_names):
    unique_labels = list(set(labels.astype(int)))
    color_map = get_color_palette(max(unique_labels) + 1)

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:
            label = int(label)
            color = color_map[label]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
            cv2.putText(orig, f"{label_names[label]}: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return orig, color_map
def main(args):
    device, orig,image,model = initialize(args)
    detect_model = init_detect_model(args)

    # run
    with torch.no_grad():
        model.eval()
        logits_r,logits_cw = model(image)
        predroom = BCHW2colormap(logits_r)
        predboundary = BCHW2colormap(logits_cw)
    if args.postprocess:
        # postprocess
        predroom = post_process(predroom,predboundary)
    rgb = ind2rgb(predroom,color_map=floorplan_fuse_map)

    with torch.no_grad():
        if args.dnn_model == 'yolo':
            results = detect_model(image)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy()
   
    label_names = {
        0: "Door",
        1: "ArmChair",
        2: "Bed",
        3: "CoffeeTable",
        4: "DiningTable",
        5: "LargeSink",
        6: "LargeSofa",
        7: "RoundTable",
        8: "Sink",
        9: "SmallSink",
        10: "SmallSofa",
        11: "Tub",
        12: "Twin_Sink",
        13: "wall",
        14: "windown"
    }
    orig, color_map = draw_detection(orig, boxes, scores, labels, label_names)
    # Tạo chú thích (legend)
    # Chuyển BGR sang RGB và chuẩn hóa về [0, 1]
    patches = [
        mpatches.Patch(
            color=[color_map[i][2]/255, color_map[i][1]/255, color_map[i][0]/255],
            label=label_names[i]
        )
        for i in color_map.keys() if i in label_names
    ]


    plt.figure(figsize=(15,5))
    plt.subplot(1, 4, 1); plt.imshow(orig[:, :, ::-1]); plt.axis("off"); plt.title("Detection")
    plt.subplot(1, 4, 2); plt.imshow(rgb); plt.axis("off"); plt.title("Room Map")
    plt.subplot(1, 4, 3); plt.imshow(predboundary); plt.axis("off"); plt.title("Boundary")

    plt.subplot(1, 4, 4); plt.axis('off'); plt.legend(handles=patches, loc='center')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--loadmodel',type=str,default="log/store2/checkpoint.pt")
    p.add_argument('--postprocess',type=bool,default=True)
    p.add_argument('--dnn_model',type=str,default='yolo')
    p.add_argument('--dnn_model_path',type=str,default='content/runs/detect/train2/weights/best.pt')
    p.add_argument('--image_path',type=str,default="dataset/newyork/test/47545145.jpg")
    args = p.parse_args()

    main(args)




