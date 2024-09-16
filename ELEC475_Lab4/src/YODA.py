import torch
import os
import cv2
import argparse
import model
import numpy as np
from KittiDataset import KittiDataset
from KittiAnchors import Anchors
from torchvision import transforms


save_ROIs = False
max_ROIs = -1
shapes = [(150, 150)]

# build batch from the ROIs
def batch_ROIs(ROIs, shape):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(shape, antialias=True)
    ])
    batch = torch.empty(size=(len(ROIs), 3, shape[0], shape[1]))
    for i in range(len(ROIs)):
        ROI = ROIs[i]
        # print('break 650: ', ROI.shape, ROI.dtype)
        ROI = transform(ROI)
        ROI = torch.swapaxes(ROI, 1, 2)
        # print('break 57: ', ROI.shape)
        batch[i] = ROI
        # print('break 665: ', i, batch.shape)
    return batch

# calculate IOU score
def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def calc_max_IoU(ROI, ROI_list):
    max_IoU = 0
    for i in range(len(ROI_list)):
        max_IoU = max(max_IoU, calc_IoU(ROI, ROI_list[i]))
    return max_IoU

def main():
    label_file = 'labels_new.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    parser.add_argument('-o', metavar='output_dir', type=str, help='output dir (./)')
    parser.add_argument('-IoU', metavar='IoU_threshold', type=float, help='[0.02]')
    parser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    parser.add_argument('-m', metavar='mode', type=str, help='[train/test]')
    parser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')
    parser.add_argument('-c', metavar='classifier_weight', type=str, help='classifier weight file')
    args = parser.parse_args()

    input_dir = None
    if args.i != None:
        input_dir = args.i

    output_dir = None
    if args.o != None:
        output_dir = args.o

    IoU_threshold = 0.02
    if args.IoU != None:
        IoU_threshold = float(args.IoU)

    show_images = False
    if args.d != None:
        if args.d == 'y' or args.d == 'Y':
            show_images = True

    training = True
    if args.m == 'test':
        training = False

    use_cuda = False
    if args.cuda != None:
        if args.cuda == 'y' or args.cuda == 'Y':
            use_cuda = True

    labels = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = KittiDataset(input_dir, training)
    anchors = Anchors()

    classifier_weight = args.c

    net = model.Net()
    net.resnet.load_state_dict(torch.load(classifier_weight, map_location='cpu'))
    net.to(device)
    net.eval()

    i = 0
    for item in enumerate(dataset):
        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        # print(i, idx, label)

        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)  # ground truth?

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        if show_images:
            image1 = image.copy()
            for j in range(len(anchor_centers)):
                x = anchor_centers[j][1]
                y = anchor_centers[j][0]
                cv2.circle(image1, (x, y), radius=4, color=(255, 0, 255))

        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        # print('break 555: ', boxes)

        #  build a batch from the ROIs
        batch = batch_ROIs(ROIs=ROIs, shape=shapes[0])
        predicted_ROIs_idx = []

        with torch.no_grad():
            #  passed the batch through the classifier
            batch = batch.to(device)
            # print(batch.shape)
            output = net.forward(batch)
            _, predicted = output.max(1)
            # print(predicted)
            for idx in range(len(predicted)):
                if predicted[idx] == 1:
                    predicted_ROIs_idx += [idx]
            print(predicted_ROIs_idx)
            # box_predicted = boxes[idx]

        ROI_IoUs = []
        for idx in range(len(predicted_ROIs_idx)):
            print(boxes[idx], car_ROIs)
            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

        print(ROI_IoUs)

        # for k in range(len(boxes)):
        #     filename = str(i) + '_' + str(k) + '.png'
        #     if save_ROIs == True:
        #         cv2.imwrite(os.path.join(output_dir, filename), ROIs[k])
        #     name_class = 0
        #     name = 'NoCar'
        #     if ROI_IoUs[k] >= IoU_threshold:
        #         name_class = 1
        #         name = 'Car'
        #     labels += [[filename, name_class, name]]

        if show_images:
            cv2.imshow('image', image1)
        # key = cv2.waitKey(0)
        # if key == ord('x'):
        #     break

        # if show_images:
        #     image2 = image1.copy()
        #
        #     for k in range(len(boxes)):
        #         if ROI_IoUs[k] > IoU_threshold:
        #             box = boxes[k]
        #             pt1 = (box[0][1], box[0][0])
        #             pt2 = (box[1][1], box[1][0])
        #             cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))
        #
        # if show_images:
        #     cv2.imshow('boxes', image2)
        #     key = cv2.waitKey(0)
        #     if key == ord('x'):
        #         break

        i += 1

        if max_ROIs > 0 and i >= max_ROIs:
            break
    #
    # print(labels)
    #
    # if save_ROIs == True:
    #     with open(os.path.join(output_dir, label_file), 'w') as f:
    #         for k in range(len(labels)):
    #             filename = labels[k][0]
    #             name_class = str(labels[k][1])
    #             name = labels[k][2]
    #             box = boxes[k]
    #             pt1 = [box[0][1], box[0][0]]  # [0][1]: top left x; [0][0]: top left y
    #             pt2 = [box[1][1], box[1][0]]  # [1][1]: bottom right x; [1][0]: bottom right y
    #             f.write(filename + ' ' + name_class + ' ' + name + ' '
    #                     + str(pt1[0]) + ' ' + str(pt1[1]) + ' ' + str(pt2[0]) + ' ' + str(pt2[1]) + ' ' + '\n')
    #     f.close()


###################################################################

main()
