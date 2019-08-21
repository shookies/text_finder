
import cv2
import time
import numpy as np
import argparse as ap
from imutils.object_detection import non_max_suppression


def parse_arguments():

    parser = ap.ArgumentParser()
    parser.add_argument("-i","--image", type=str, help="path for input image")
    parser.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
    parser.add_argument("-c", "--min-confidence", type=float, default = 0.5, help="minimum confidence required"
                                                                                  "to inspect a region")
    parser.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be a multiple of 32)")
    parser.add_argument("--h", "--height", type=int, default=320, help="resized image height (should be a "
                                                                       "multiple of 32")

    args = vars(parser.parse_args())
    return args


def load_and_resize_image(im_path, newHeight, newWidth):

    #TODO original = image.copy() outside of function
    image = cv2.imread(im_path)
    (height, width) = image.shape[:2]
    ratio_width, ratio_height = width / float(newWidth), height / float(newHeight)    #calculates ratio for later use
    ratio = (ratio_width, ratio_height)

    image = cv2.resize(image, (newWidth, newHeight))
    return [image, (image.shape[:2]), ratio]

def detect(east_path, image):

    model = cv2.dnn.readNet(east_path)
    width, height = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor= 1.0, size= (width,height), mean= (123.68,116.78, 103.94),
                                 swapRB= True, crop= False)     #preprocess the image for detection
    model.setInput(blob)
    scores, geometry = model.forward([
        "feature_fusion/Conv_7/Sigmoid",        #probability of each region containing text
        "feature_fusion/concat_3"])             #used for deriving the bounding boxes

    boxes, confidences =

# args = parse_arguments()
# im = cv2.imread(args["image"])
# cv2.imshow("meow", im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
