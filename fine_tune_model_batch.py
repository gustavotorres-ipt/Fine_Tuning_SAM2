import json
import random
from cv2.cuda import EVENT_BLOCKING_SYNC
import numpy as np
import torch
import cv2 # type: ignore
import os
import matplotlib.pyplot as plt
import pycocotools
from pycocotools.coco import COCO # por algum motivo precisa dar import nisso
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matplotlib.patches import Rectangle

N_IMAGES_TRAINING = 1000
N_IMAGES_VAL = 100
N_EPOCHS = 120
BATCH_SIZE = 4

IMGS_BASE_DIR = "imagens_parihaka"
INPUTS_DIR = f"./{IMGS_BASE_DIR}/seismic"
LABELS_DIR = f"./{IMGS_BASE_DIR}/annotations"

FIRST_EPOCH = 0
POSSIBLE_CONFIGS = {"tiny": ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml"), # 33 min
                    "small": ("sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml"), # 37 min
                    "base_plus": ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml"), # 50 min
                    "large": ("sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml")} # 1h38 min

BASE_MODEL_CONFIG = POSSIBLE_CONFIGS["tiny"]
# CHECKPOINT_NAME = "sam2.1_hiera_small_seismic_100_epochs.pth"


def show_image(image, points=None, bbox=None, is_mask=False):
    _, ax = plt.subplots()

    ax.imshow(image)
    if is_mask:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = image.shape
        mask_highlighted = image.reshape(h, w, 1) * color.reshape(1, 1, -1)

        ax.imshow(mask_highlighted)

    if points is not None:
        for point in points:
            ax.scatter(point[0], point[1], color='green', marker='*', s=100,
                       edgecolor='white', linewidth=1.25)
    if bbox is not None:
        # w0, h0, w1, h1 = tuple(bbox)
        # rect = Rectangle((w0, h0), w1-w0, h1-h0, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        # ax.add_patch(rect)
        point1 = (int(bbox[0]), int(bbox[1]))
        point2 = (int(bbox[2]), int(bbox[3]))
        # cv2.rectangle(image_batch[i], point1, point2, (0,255,0), 2)
        rect = Rectangle((point1[0], point1[1]), point2[0]-point1[0], point2[1]-point1[1],
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def plot_masks(image_batch, gt_mask, pred_mask, bboxes=None):
    for i in range(len(image_batch)):
        _, axarr = plt.subplots(1, 2)
        segm = 255 * gt_mask[i].astype('uint8')
        segm = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
        blend_gt = cv2.addWeighted(segm, 0.2, image_batch[i], 0.8, 0.0)

        segm = 255 * pred_mask[i].astype('uint8')
        segm = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
        blend_pred = cv2.addWeighted(segm, 0.2, image_batch[i], 0.8, 0.0)

        axarr[0].imshow(blend_pred)
        axarr[1].imshow(blend_gt)

        axarr[0].axis("off")
        axarr[1].axis('off')

        axarr[0].set_title("Predicted Mask")
        axarr[1].set_title('Ground Truth')

        if bboxes is not None:

            point1 = (int(bboxes[i][0]), int(bboxes[i][1]))
            # point2 = (int(bboxes[i][0] + bboxes[i][2]), int(bboxes[i][1] + bboxes[i][3]))
            point2 = (int(bboxes[i][2]), int(bboxes[i][3]))
            # cv2.rectangle(image_batch[i], point1, point2, (0,255,0), 2)
            rect = Rectangle((point1[0], point1[1]), point2[0]-point1[0], point2[1]-point1[1],
                             linewidth=1, edgecolor='r', facecolor='none')
            axarr[1].add_patch(rect)
        plt.show()
        plt.close()


def logscaler(x, a):
    x = x.copy()
    x[x>0] = a * np.log(1 + x[x>0]/a)
    x[x<0] = - a * np.log(1 - x[x<0]/a)
    return x


def normalize_and_add_color_channels(image):
    # image = logscaler(image, 128)
    image = 255 * (image - image.min()) / (image.max() - image.min())
    return np.stack([image, image, image], axis=2).astype(np.uint8)


def decode_segmentation_mask(segm_entry):
    encoded_mask = {'size': segm_entry['size'],
                    'counts': segm_entry['counts']}
    return pycocotools.mask.decode(encoded_mask)

def get_random_point(mask):
    point_yx = mask[np.random.randint(mask.shape[0])]
    point_xy = point_yx.copy()
    point_xy[0] = point_yx[1]
    point_xy[1] = point_yx[0]
    return point_xy

def read_from_image(data, idx_image):
    N_PONTOS_POSITIVOS = 1

    # selected_image = np.random.randint(len(data))
    ent  = data[idx_image] # choose random entry
    bbox = None

    if ".json" in ent["annotation"]:
        image = normalize_and_add_color_channels(np.load(ent["image"]))
        ann_map = decode_segmentation_mask(ent['segmentation'])
        bbox = [int(coord) for coord in ent['bbox']]
    else:
        image = cv2.imread(ent["image"])  # read image
        ann_map = cv2.imread(
            ent["annotation"], cv2.IMREAD_GRAYSCALE
        ).astype(np.uint8) # read annotation

    # bbox = np.array(bbox)
    # temp = bbox[2]
    # bbox[2] = bbox[1]
    # bbox[1] = temp
    # print(ent["image"])
    # plot_masks(image[None, ...], ann_map[None, ...], ann_map[None, ...], bbox[None, ...])

    r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
    image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
    ann_map = cv2.resize(
        ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    if bbox is not None:
        bbox = [int(bbox[0] * r), int(bbox[1] * r),
                int((bbox[0] + bbox[2]) * r), int((bbox[1] + bbox[3]) * r)]
        # bbox = [r *  bbox[0], r * bbox[1], r * (bbox[0] + bbox[2]), r * (bbox[1] + bbox[3])]

    if image.shape[0] < 1024:
        image = np.concatenate(
            [image, np.zeros([1024 - image.shape[0], image.shape[1], 3], dtype=np.uint8)], axis=0)
        ann_map = np.concatenate(
            [ann_map, np.zeros( [1024 - ann_map.shape[0], ann_map.shape[1]], dtype=np.uint8)], axis=0)

    if image.shape[1] < 1024:
        image = np.concatenate(
            [image, np.zeros([image.shape[0] , 1024 - image.shape[1], 3], dtype=np.uint8)], axis=1)
        ann_map = np.concatenate([
            ann_map, np.zeros([ann_map.shape[0] , 1024 - ann_map.shape[1]], dtype=np.uint8)], axis=1)

    inds_mask = np.argwhere(ann_map > 0)
    masks = np.array(ann_map)

    pontos_aleatorios = [#inds_mask[np.random.randint(inds_mask.shape[0])]
                         get_random_point(inds_mask)
                         for _ in range(N_PONTOS_POSITIVOS)][0]
    pontos_aleatorios = np.array(pontos_aleatorios).reshape(N_PONTOS_POSITIVOS, 2)

    # show_image(masks, pontos_aleatorios, is_mask=True)
    return image, masks, pontos_aleatorios, bbox #, np.ones([N_PONTOS_POSITIVOS, 1])


def read_batch(data, batch_size=4):
    limage = []
    lmask = []
    linput_points = []
    lboxes = []

    for i in range(batch_size):
        image, mask, input_points, bbox = read_from_image(data, i)
        limage.append(image)
        lmask.append(mask)
        linput_points.append(input_points)
        lboxes.append(bbox)

    return limage, np.array(lmask), np.array(linput_points), np.ones([batch_size, 1]), np.array(lboxes)


def run_sam2_iter(predictor, training_images):
    image_batch, masks, input_points, input_labels, bboxes = read_batch(
                    training_images, BATCH_SIZE) # load data batch
    if masks.shape[0]==0: return # ignore empty batches

    predictor.set_image_batch(image_batch) # apply SAM image encoder to the image

    bboxes = torch.tensor(bboxes.astype(np.float16)).cuda()
    input_points = torch.tensor(input_points.astype(np.float16)).cuda()

    # Insere os prompts para a imagem
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_points, input_labels, box=bboxes, mask_logits=None, normalize_coords=True
                )
    # Calcula os embeddings da imagem
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=bboxes, masks=None,)

    #high_res_features = [feat_level[-1].unsqueeze(0)
    #                     for feat_level in predictor._features["high_res_feats"]]
    high_res_features = predictor._features["high_res_feats"]
    image_embeded = predictor._features["image_embed"]#[-1].unsqueeze(0)

    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=image_embeded,
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                # Upscale the masks to the original image resolution
    prd_masks = predictor._transforms.postprocess_masks(
                    low_res_masks, predictor._orig_hw[-1])

                # Segmentaion Loss caclulation
    gt_mask = torch.tensor(masks.astype(np.float32)).cuda()
    prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) -
                            (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

                # Calcula O loss total iou
    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1) # intersection
    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
    loss = seg_loss + score_loss*0.05  # mix losses

    return loss, iou, prd_mask, gt_mask, image_batch, bboxes


def run_training(predictor, training_data, validation_data):

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(),
                                lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler() # set mixed precision
    
    mean_loss_values_train = []
    mean_loss_values_val = []

    mean_iou_values_train = []
    mean_iou_values_val = []

    for epoch in range(FIRST_EPOCH, FIRST_EPOCH + N_EPOCHS):
        mean_iou_train = 0
        iou_values = []
        loss_values = []

        # images_data = all_data[:N_IMAGES_TRAINING]
        training_images = training_data

        print(f"EPOCH {epoch+1}/{N_EPOCHS}...")

        for itr in tqdm(range(len(training_data) // BATCH_SIZE)):
            with torch.cuda.amp.autocast(): # cast to mix precision
                loss, iou, _, _, _, bboxes = run_sam2_iter(predictor, training_images)

                iou_values.append(iou)
                loss_values.append(loss)

                # apply back propogation
                predictor.model.zero_grad() # empty gradient
                scaler.scale(loss).backward()  # Backpropogate
                scaler.step(optimizer)
                scaler.update() # Mix precision

            training_images = training_images[BATCH_SIZE:]

        iou_stacked = torch.stack(iou_values, dim=0)
        mean_iou_train = float(np.mean(iou_stacked.cpu().detach().numpy()))

        loss_stacked = torch.stack(loss_values, dim=0)
        mean_loss_train = float(np.mean(loss_stacked.cpu().detach().numpy()))

        mean_loss_values_train.append(mean_loss_train)
        mean_iou_values_train.append(mean_iou_train)

        if (epoch + 1) % 10 == 0 : 
            name_output = f"checkpoints/{BASE_MODEL_CONFIG[0][:-3]}_seismic_{epoch+1}_epochs.pth"
            torch.save(predictor.model.state_dict(), name_output)
            print(name_output, "saved successfully.")

        print(f"Epoch {epoch+1} - Training Accuracy (IOU) = ", mean_iou_train)

        mean_loss_val, mean_iou_val = run_validation(predictor, validation_data)
        mean_loss_values_val.append(mean_loss_val)
        mean_iou_values_val.append(mean_iou_val)
 
    return mean_loss_values_train, mean_loss_values_val, mean_iou_values_train, mean_iou_values_val


def run_validation(predictor, validation_data, n_batches_to_plot=0):
    iou_values = []
    loss_values = []

    validation_size = len(validation_data)

    for itr in tqdm(range(validation_size // BATCH_SIZE)):

        with torch.cuda.amp.autocast(): # cast to mix precision
            loss, iou, prd_mask, gt_mask, image_batch, bboxes = run_sam2_iter(predictor, validation_data)

            iou_values.append(iou)
            loss_values.append(loss)

        validation_data = validation_data[BATCH_SIZE:]

        # Plot images during validation for visualization
        if n_batches_to_plot > 0 and itr < n_batches_to_plot:
            pred_mask_plot = (prd_mask > 0.5).cpu().detach().numpy()
            gt_mask_plot = gt_mask.cpu().detach().numpy()

            # for b in range(BATCH_SIZE):
            #     show_image(gt_mask_plot, input_points.cpu().detach().numpy()[b], is_mask=True)
            #     show_image(pred_mask_plot, input_points.cpu().detach().numpy()[b], is_mask=True)

            print(iou_values[-1])
            plot_masks(image_batch, gt_mask_plot, pred_mask_plot, bboxes.cpu().detach().numpy())

    loss_stacked = torch.stack(loss_values, dim=0)
    mean_loss = np.mean(loss_stacked.cpu().detach().numpy())

    # Calcula IOU médio
    iou_stacked = torch.stack(iou_values, dim=0)
    mean_iou = np.mean(iou_stacked.cpu().detach().numpy())

    print("Validation Accuracy (IOU) = ", mean_iou)
    # Mostrar imagens
    return float(mean_loss), float(mean_iou)


all_data=[] # list of files in dataset

def get_training_val_data():
    for _, nome_arquivo in enumerate(os.listdir(INPUTS_DIR)):  # go over all folder annotation
        path_imagem = f"{INPUTS_DIR}/{nome_arquivo}"

        if "parihaka" in INPUTS_DIR :
            nome_label = f'{".".join(nome_arquivo.split(".")[:-1])}.json'
        else:
            nome_label = nome_arquivo
        path_label  = f"{LABELS_DIR}/{nome_label}"

        if ".json" in path_label:
            # Lê o arquivo json e conta a quantidade de máscaras
            annotations_imagem = json.load(open(path_label))

            for annotation_info in annotations_imagem:
                all_data.append(
                    {"image": path_imagem, "annotation": path_label,
                    "bbox": annotation_info["bbox"],
                    "segmentation": annotation_info["segmentation"]}
                )
        else:
            all_data.append({"image": path_imagem, "annotation": path_label})

    # At least 20 images are used for validation
    max_training_images = int(0.9 * len(all_data)) \
        if N_IMAGES_TRAINING > (len(all_data) - 20) else N_IMAGES_TRAINING

    random.seed(42)
    random.shuffle(all_data)

    training_data = all_data[:max_training_images]
    validation_data = all_data[max_training_images :
                               (max_training_images + N_IMAGES_VAL)]
    
    return training_data, validation_data


def main():
    sam2_checkpoint = os.path.join("checkpoints", BASE_MODEL_CONFIG[0]) # path to model weight
    model_cfg = os.path.join("configs", "sam2.1", BASE_MODEL_CONFIG[1]) # model config

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
    predictor = SAM2ImagePredictor(sam2_model) # load net

    predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder 
    predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

    # predictor.model.load_state_dict(torch.load(f"./checkpoints/{CHECKPOINT_NAME}"))

    os.makedirs("checkpoints", exist_ok=True)

    training_data, validation_data = get_training_val_data()

    loss_training, loss_validation, iou_training, iou_validation = run_training(
        predictor, training_data, validation_data)

    training_info = {
        "loss_train": loss_training,        
        "loss_val": loss_validation,
        "iou_train": iou_training,
        "iou_val": iou_validation,
    }
    output_file = f"resultados_{BASE_MODEL_CONFIG[0][:-3]}_seismic.json"

    with open(output_file, "w") as f:
        json.dump(training_info, f, indent=4)
        print(output_file, "salvo com sucesso.")

if __name__ == "__main__":
    main()
