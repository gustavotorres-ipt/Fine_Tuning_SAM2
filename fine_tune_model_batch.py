import json
import random
import numpy as np
import torch
import cv2 # type: ignore
import os
import matplotlib.pyplot as plt
import pycocotools
import matplotlib.pyplot as plt
from pycocotools.coco import COCO # por algum motivo precisa dar import nisso
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matplotlib.patches import Rectangle


N_IMAGES_TRAINING = 1000
N_IMAGES_VAL = 100
N_EPOCHS = 100
BATCH_SIZE = 4

IMGS_BASE_DIR = "imagens_parihaka"
INPUTS_DIR = f"./{IMGS_BASE_DIR}/seismic"
LABELS_DIR = f"./{IMGS_BASE_DIR}/annotations"

def show_image(image, points, is_mask=False):
    fig, ax = plt.subplots()

    ax.imshow(image)
    if is_mask:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = image.shape
        mask_highlighted = image.reshape(h, w, 1) * color.reshape(1, 1, -1)

        ax.imshow(mask_highlighted)
    if points is not None:
        for point in points:
            ax.scatter(point[1], point[0], color='green', marker='*', s=100,
                       edgecolor='white', linewidth=1.25)
    plt.show()


def add_color_channels(image):
    return np.stack([image, image, image], axis=2)


def decode_segmentation_mask(segm_entry):
    encoded_mask = {'size': segm_entry['size'],
                    'counts': segm_entry['counts']}
    return pycocotools.mask.decode(encoded_mask)


def read_from_image(data, idx_image):
    N_PONTOS_POSITIVOS = 1

    # selected_image = np.random.randint(len(data))
    ent  = data[idx_image] # choose random entry

    if ".json" in ent["annotation"]:
        image = add_color_channels(np.load(ent["image"])).astype(np.uint8)
        ann_map = decode_segmentation_mask(ent['segmentation'])
        bbox = [int(coord) for coord in ent['bbox']]
    else:
        image = cv2.imread(ent["image"])  # read image
        ann_map = cv2.imread(
            ent["annotation"], cv2.IMREAD_GRAYSCALE
        ).astype(np.uint8) # read annotation

    r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
    image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
    ann_map = cv2.resize(
        ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

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

    pontos_aleatorios = [inds_mask[np.random.randint(inds_mask.shape[0])]
                         for _ in range(N_PONTOS_POSITIVOS)][0]
    pontos_aleatorios = np.array(pontos_aleatorios).reshape(N_PONTOS_POSITIVOS, 2)

    # show_image(image, pontos_aleatorios)
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


def run_training(predictor, training_data, validation_data):

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(),
                                lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler() # set mixed precision

    for epoch in range(N_EPOCHS):
        mean_iou = 0
        iou_values = []
        # images_data = all_data[:N_IMAGES_TRAINING]
        training_images = training_data

        print(f"EPOCH {epoch+1}/{N_EPOCHS}...")

        for itr in tqdm(range(len(training_data) // BATCH_SIZE)):
            with torch.cuda.amp.autocast(): # cast to mix precision
                loss, iou, _, _ = run_sam2_iter(predictor, training_images)

                iou_values.append(iou)

                # apply back propogation
                predictor.model.zero_grad() # empty gradient
                scaler.scale(loss).backward()  # Backpropogate
                scaler.step(optimizer)
                scaler.update() # Mix precision

            training_images = training_images[BATCH_SIZE:]

        iou_stacked = torch.stack(iou_values, dim=0)
        mean_iou = np.mean(iou_stacked.cpu().detach().numpy())

        if (epoch + 1) % 10 == 0 : 
            name_output = f"checkpoints/seismic_model_{epoch+1}.pth"
            torch.save(predictor.model.state_dict(), name_output)
            print(name_output, "saved successfully.")

        print(f"Epoch {epoch+1} - Training Accuracy (IOU) = ", mean_iou)

        run_validation(predictor, validation_data)


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

    return loss, iou, prd_mask, gt_mask


def plot_masks(gt_mask, pred_mask):
    f, axarr = plt.subplots(2, BATCH_SIZE)

    for i in range(BATCH_SIZE):
        axarr[0, i].imshow(pred_mask[i], cmap="copper")
        axarr[1, i].imshow(gt_mask[i], cmap="copper")

        axarr[0, i].axis('off')
        axarr[1, i].axis('off')

        axarr[0, i].set_title("Predicted Mask")
        axarr[1, i].set_title('Ground Truth')
    plt.show()


def run_validation(predictor, validation_data, n_batches_to_plot=0):
    iou_values = []

    validation_size = len(validation_data)

    for itr in tqdm(range(validation_size // BATCH_SIZE)):

        with torch.cuda.amp.autocast(): # cast to mix precision
            loss, iou, prd_mask, gt_mask = run_sam2_iter(predictor, validation_data)

            iou_values.append(iou)

        validation_data = validation_data[BATCH_SIZE:]

        # Plot images during validation for visualization
        if n_batches_to_plot > 0 and itr < n_batches_to_plot:
            pred_mask_plot = (prd_mask > 0.5).cpu().detach().numpy()
            gt_mask_plot = gt_mask.cpu().detach().numpy()

            # for b in range(BATCH_SIZE):
            #     show_image(masks[b], input_points.cpu().detach().numpy()[b], is_mask=True)
            #     show_image(preds_to_plot[b], input_points.cpu().detach().numpy()[b], is_mask=True)

            plot_masks(gt_mask_plot, pred_mask_plot)
            print(iou_values[-1])

    # Calcula IOU médio
    iou_stacked = torch.stack(iou_values, dim=0)
    mean_iou = np.mean(iou_stacked.cpu().detach().numpy())

    print("Validation Accuracy (IOU) = ", mean_iou)
    # Mostrar imagens


all_data=[] # list of files in dataset

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

parent = os.path.dirname(os.getcwd())

sam2_checkpoint = os.path.join("checkpoints", "sam2.1_hiera_tiny.pt") # path to model weight
model_cfg = os.path.join("configs", "sam2.1", "sam2.1_hiera_t.yaml") # model config

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
predictor = SAM2ImagePredictor(sam2_model) # load net

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder 
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

os.makedirs("checkpoints", exist_ok=True)

# At least 20 images are used for validation
max_training_images = int(0.9 * len(all_data)) \
    if N_IMAGES_TRAINING > (len(all_data) - 20) else N_IMAGES_TRAINING

random.shuffle(all_data)

training_data = all_data[:max_training_images]
validation_data = all_data[max_training_images :
                           (max_training_images + N_IMAGES_VAL)]

run_training(predictor, training_data, validation_data)

# predictor.model.load_state_dict(torch.load("./checkpoints/best_model.pth"))
# run_validation(predictor, validation_data, n_batches_to_plot=20)