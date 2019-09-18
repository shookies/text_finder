
import numpy as np
import cv2, math, pdf2image, os, tempfile
import argparse as ap
from imutils.object_detection import non_max_suppression

import CharDetector

def round_down(num, divisor):

    return num - (num%divisor)

def parse_arguments():

    parser = ap.ArgumentParser()
    parser.add_argument("-i","--image", type=str, help="path for input image")
    parser.add_argument("-east", "--east", type=str, help="path for EAST text detector")
    parser.add_argument("-c", "--min_confidence", type=float, default = 0.7, help="minimum confidence required"
                                                                                  "to inspect a region")
    # parser.add_argument("--w", "--width", type=int, default=320, help="resized image width (should be a "
    #                                                                   "multiple of 32)")
    # parser.add_argument("--h", "--height", type=int, default=320, help="resized image height (should be a "
    #                                                                    "multiple of 32)")
    parser.add_argument("--nms","--non_max_suppression_thresh",type=float, default=0.4,
                        help= "Non - maximum suppression threshold")

    args = parser.parse_args()

    return args


def load_and_resize_image(im_path):

    #TODO handle multiple pages
    #TODO handle file path error
    if im_path.endswith(".pdf"):
        # args.image = pdf2image.convert_from_path(args.image)
        with tempfile.TemporaryDirectory() as path:
            images_from_path = pdf2image.convert_from_path(im_path, output_folder=path, last_page=1, first_page =0)


        mat_from_image = np.array(images_from_path[0])
        image = cv2.cvtColor(mat_from_image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(im_path)

    original = image.copy()
    height, width = image.shape[:2]
    newHeight = height if (height % 32 == 0) else round_down(height,32)     #rounds down to the nearest multiple of 32
    #in order for EAST to work with the image
    newWidth = width if (width % 32 == 0) else round_down(width,32)
    # print(newHeight, newWidth)
    ratio_width = width / float(newWidth)        #calculates ratio for later use (upscaling image back)
    ratio_height = height / float(newHeight)
    ratio = (ratio_width, ratio_height)

    image = cv2.resize(image, (newWidth, newHeight))
    return [image, ratio, original, newWidth, newHeight]

def detect(east_path, image, minConfidence, inputHeight, inputWidth):

    model = cv2.dnn.readNet(east_path)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor= 1.0, size= (width,height), mean= (123.68,116.78, 103.94),
                                 swapRB= True, crop= False)     #preprocess the image for detection
    model.setInput(blob)
    scores, geometry = model.forward([
        "feature_fusion/Conv_7/Sigmoid",        #probability of each region containing text
        "feature_fusion/concat_3"])             #used for deriving the bounding boxes


    rects, confidences = decode(scores, geometry, minConfidence)

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    H, W = image.shape[:2]
    rW = W / float(inputWidth)
    rH = H / float(inputHeight)

    #sort for testing
    def dist(box):

        return math.sqrt(box[0]**2 + box[1]**2)
    foxes = boxes.tolist()
    foxes = sorted(foxes, key=dist)
    boxes = np.array(foxes)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:

        # print(startX, startY)

        dilation_kernel, padding, char_padding, debris_thresh = define_parameters(image,[startX, startY, endX, endY])
        # padding, debris_thresh = define_parameters(image,[startX, startY, endX, endY])
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW) - padding
        startY = int(startY * rH) - padding
        endX = int(endX * rW) + padding
        endY = int(endY * rH) + padding
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        word = image[startY:endY, startX:endX]
        chars, BB = CharDetector.detect_chars(word.astype('uint8'), dilation_kernel, debris_thresh)
        # char_boxes.append(chars)
        for box in BB:
            cv2.rectangle(image,(startX + box[0] - char_padding,startY + box[2]), (startX + box[1] + char_padding, startY + box[3]), (255,0,0),1)

    cv2.imshow("meow",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def define_parameters(image, BB):

    #TODO use only height of text. width = difference between BB of characters in x
    ratio = calculate_ratio(image,BB)
    height_constant = 200
    width_constant = 80
    #
    # padding_constant = 12
    debris_threshold_constant = 20
    # print(ratio)
    h = int(ratio * height_constant)
    w = int(ratio * width_constant)
    # ker_dims = (20, 2)
    ker_dims = (h, w)
    print(ker_dims)
    dilation_kernel = np.ones(ker_dims, np.uint8)
    # padding = int(ratio * padding_constant)
    debris_threshold = int(ratio * debris_threshold_constant)   #threshold is 0 for small enough words

    # dilation_kernel = np.ones((2000,2),np.uint8)
    padding = 2
    char_padding = 1
    # debris_threshold = 1

    return [dilation_kernel, padding, char_padding, debris_threshold]


def calculate_ratio(image,BB):

    #Assumes axis-aligned bounding box (not rotated)
    im_h, _ = image.shape[:2]
    bb_h = BB[3] - BB[1]

    return float(bb_h / im_h)

def decode(scores, geometry, minConfidence):

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    numRows, numCols = scores.shape[2:4]
    rects = []
    confidences = []

    #numrows = height, numcols = width

    for y in range(0, numRows):
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]    #h_upper from offset
        xData1 = geometry[0,1,y]    #w_right from offset            offset == some middle point in rectangle
        xData2 = geometry[0,2,y]    #h_lower_from offset
        xData3 = geometry[0,3,y]    #w_left from offset
        anglesData = geometry[0,4,y]


        for x in range(0, numCols):
            score = scoresData[x]
            if score < minConfidence:
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cosA * xData1[x]) + (sinA * xData2[x]))
            endY = int(offsetY - (sinA * xData1[x]) + (cosA * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            confidences.append(float(score))
            rects.append((startX,startY,endX,endY))

    return [rects, confidences]


def main():
    args = parse_arguments()

    im_path = args.image
    east_path = args.east
    minConfidence = args.min_confidence
    image, ratio, orig, w, h = load_and_resize_image(im_path)
    detect(east_path, image, minConfidence, h, w)

    return 0

main()

# test_im = np.ones((100,100))
# test_bb = [25,25,74,76]
# print(calculate_ratio(test_im,test_bb))


# -i /cs/usr/shookies/Desktop/text_finder/stop_sign.jpg -east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb
#-i /cs/usr/shookies/Desktop/text_finder/test_im.png -east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb
# --i /cs/usr/shookies/Desktop/text_finder/test2.png --east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb
# --i /cs/usr/shookies/Desktop/text_finder/lines.png --east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb

# -i C:\Users\User\Desktop\CS\text_finder\blind.png -east C:\Users\User\Desktop\CS\text_finder\east_text_detection.pb
# -i C:\Users\User\Desktop\CS\text_finder\gilayon.pdf -east C:\Users\User\Desktop\CS\text_finder\east_text_detection.pb