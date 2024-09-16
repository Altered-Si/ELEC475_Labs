from main import *
import cv2
import numpy as np
from PIL import Image
if __name__ == "__main__":
    # parameters
    n_epochs = args.e
    batch_size = args.b
    model_weight = args.s
    mode = args.m
    show_images = True if args.d == 'y' or args.d == 'Y' else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weight = args.s
    net = Net()
    net.load_state_dict(torch.load(model_weight))
    net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)

            output = net.forward(image)  # predicted nose central coordinate
            if show_images:
                # convert tensor object to PIL image
                image = transforms.ToPILImage()(image[0])
                # convert PIL image to NumPy array
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                image1 = image.copy()

                ground_truth_nose = (label[0][0].item(), label[0][1].item())
                # convert ground truth x and y to integer (pixel form)
                ground_truth_nose = (int(ground_truth_nose[0]), int(ground_truth_nose[1]))
                predicted_nose = (output[0][0].item(), output[0][1].item())
                # convert predicted x and y to integer (pixel form)
                predicted_nose = (int(predicted_nose[0]), int(predicted_nose[1]))
                # print('predicted_nose', predicted_nose)
                # draw a red circle with radius of 2 around the prediction
                cv2.circle(image, predicted_nose, 2, (0, 0, 255), 1)
                # draw a green circle with radius of 8 around the ground truth
                cv2.circle(image1, ground_truth_nose, 8, (0, 255, 0), 1)
                cv2.imshow('prediction', image)
                cv2.imshow('ground truth', image1)
                key = cv2.waitKey(0)
                # press Q to exit
                if key == ord('q'):
                    exit(0)
                cv2.destroyAllWindows()
