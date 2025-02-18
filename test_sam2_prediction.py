import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2  #, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matplotlib.patches import Rectangle


def calc_iou(pred_mask, gt_mask):
    inter = (gt_mask * (pred_mask > 0.5)).sum(1).sum(1) # intersection
    iou = inter / (gt_mask.sum(1).sum(1) + (pred_mask > 0.5).sum(1).sum(1) - inter)

    return iou


def resize_image(img):
    img_arr = np.array(img)

    r = np.min([1024 / img_arr.shape[1], 1024 / img_arr.shape[0]])
    img_arr = cv2.resize(img_arr, (int(img_arr.shape[1] * r), int(img_arr.shape[0] * r)))

    if img_arr.shape[0] < 1024:
        img_arr = np.concatenate([img_arr,np.zeros([1024 - img_arr.shape[0], img_arr.shape[1],3],
                                                   dtype=np.uint8)],axis=0)
    if img_arr.shape[1] < 1024:
        img_arr = np.concatenate([img_arr, np.zeros([img_arr.shape[0] , 1024 - img_arr.shape[1], 3],
                                                    dtype=np.uint8)],axis=1)
    return Image.fromarray(img_arr)


def show_image(image, mask, frame, points=None, bboxes=None):
    fig, ax = plt.subplots()
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape
    mask_highlighted =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(image)
    ax.imshow(mask_highlighted)
    ax.set_title(f"Frame {frame}")
    if points is not None:
        for point in points:
            ax.scatter(point[0, 0], point[0, 1], color='green', marker='*',
                       s=100, edgecolor='white', linewidth=1.25)

    if bboxes is not None:
        for box in bboxes:
            w0, h0, w1, h1 = tuple(box)
            rect = Rectangle((w0, h0), w1-w0, h1-h0, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
    plt.show()


def plot_masks(gt_mask, pred_mask):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(pred_mask, cmap="copper")
    axarr[1].imshow(gt_mask, cmap="copper")

    axarr[0].axis('off')
    axarr[1].axis('off')

    axarr[0].set_title("Predicted Mask")
    axarr[1].set_title('Ground Truth')
    plt.show()


def predict_next_frame(predictor, best_mask, frame, volume_f3):
    inline = volume_f3[frame].T
    arr_image = np.stack((inline, inline, inline), axis=2)
    image = Image.fromarray(arr_image, "RGB")

    predictor.set_image(image)
        
    masks, scores, logits = predictor.predict(
        mask_input = best_mask[None, :, :]
    )
    ind_max = np.argmax(scores)
    best_mask = logits[ind_max, :, :]

    show_image(np.array(image), masks[ind_max], frame)

PARENT_DIR = os.path.dirname(os.getcwd())

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Load model you need to have pretrained model already made
sam2_checkpoint = os.path.join(PARENT_DIR, "checkpoints", "sam2.1_hiera_tiny.pt") # path to model weight
model_cfg = os.path.join(PARENT_DIR, "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml") # model config

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

# Build net and load weights
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load("./checkpoints/best_model.pth"))

# mean_iou = np.mean(iou_stacked.cpu().detach().numpy())
ref_frame = 580

IMG_PATH = "./imagens/inputs"
LABEL_PATH = "./imagens/labels"

iou_values = []

for frame in tqdm(range(100)):
    current_file = np.random.randint(1000)

    selected_file = os.listdir(IMG_PATH)[current_file]

    with torch.no_grad():
        image_file = f"{IMG_PATH}/{selected_file}"
        label_file = f"{LABEL_PATH}/{selected_file}"

        #selected_points = np.array([[100, 100], [100, 100]])
        image = Image.open(image_file).convert("RGB")
        # image = resize_image(image)
        arr_image = np.array(image).astype(np.uint8)

        label = Image.open(label_file)
        # label = resize_image(label)
        arr_label = np.array(label)

        idx_labels = np.argwhere(arr_label == 1)

        selected_points = idx_labels[np.random.randint(0, len(idx_labels), 1)]

        # Todos os selected_points são um ponto positivo
        selected_points = selected_points.reshape(
            selected_points.shape[0], 1, selected_points.shape[1])

        input_boxes = None
        # input_boxes = np.array([[560, 384, 600, 462] for _ in range(len(selected_points))])

        arr_label *= 255
        arr_label = np.stack((arr_label, arr_label, arr_label), axis=2)

        # show_image(arr_image, np.zeros((arr_image.shape[0], arr_image.shape[1])), 0)
        # show_image(arr_image, np.zeros((arr_image.shape[0], arr_image.shape[1])), frame, selected_points)

        predictor.set_image(image)
        pred_masks, scores, logits = predictor.predict(
            point_coords = selected_points,
            point_labels = np.ones([selected_points.shape[0], 1]),
            box=input_boxes,
            multimask_output=False, # True,
        )
        # pred_masks = pred_masks[:, 0]
        # logits = logits[:, 0]

        pred_mask = pred_masks[ np.argsort(scores[0])][::-1][0]
        best_mask_ref = logits[ np.argsort(scores[0])][::-1][0]

        gt_mask = (arr_label / 255)[:, :, 0]

        iou_values.append( calc_iou(gt_mask[None, :, :], pred_mask[None, :, :])[0] )

        #plot_masks(gt_mask, pred_mask)
        #show_image(arr_label, np.zeros((arr_label.shape[0], arr_label.shape[1])), frame)
        #show_image(arr_image, np.zeros((arr_image.shape[0], arr_image.shape[1])), frame, selected_points)
        #show_image(arr_image, shorted_mask, frame)
        best_mask = np.copy(best_mask_ref)

    current_file += 1

mean_iou = np.mean(iou_values)

print("IOU = ", mean_iou)

"""
size_propagation = 10

with torch.no_grad():
    for frame in range(ref_frame-1, ref_frame-size_propagation, -1):
        predict_next_frame(predictor, best_mask, frame, seismic_volume)

    best_mask = np.copy(best_mask_ref)

    for frame in range(ref_frame+1, ref_frame+size_propagation):
        predict_next_frame(predictor, best_mask, frame, seismic_volume)
"""