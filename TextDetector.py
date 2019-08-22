
import cv2
import numpy as np
import argparse as ap
from imutils.object_detection import non_max_suppression


def parse_arguments():

    parser = ap.ArgumentParser()
    parser.add_argument("-i","--image", type=str, help="path for input image")
    parser.add_argument("-east", "--east", type=str, help="path for EAST text detector")
    parser.add_argument("-c", "--min-confidence", type=float, default = 0.5, help="minimum confidence required"
                                                                                  "to inspect a region")
    parser.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be a multiple of 32)")
    parser.add_argument("--h", "--height", type=int, default=320, help="resized image height (should be a "
                                                                       "multiple of 32)")

    args = vars(parser.parse_args())
    return args


def load_and_resize_image(im_path, newHeight, newWidth):

    image = cv2.imread(im_path)
    original = image.copy()
    (height, width) = image.shape[:2]
    ratio_width, ratio_height = width / float(newWidth), height / float(newHeight)    #calculates ratio for later use
    ratio = (ratio_width, ratio_height)

    image = cv2.resize(image, (newWidth, newHeight))
    return [image, ratio, original]

def detect(east_path, image, minConfidence):

    model = cv2.dnn.readNet(east_path)
    width, height = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor= 1.0, size= (width,height), mean= (123.68,116.78, 103.94),
                                 swapRB= True, crop= False)     #preprocess the image for detection
    model.setInput(blob)
    scores, geometry = model.forward([
        "feature_fusion/Conv_7/Sigmoid",        #probability of each region containing text
        "feature_fusion/concat_3"])             #used for deriving the bounding boxes


    rects, confidences = decode(scores, geometry, minConfidence)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    return boxes


def decode(scores, geometry, minConfidence):

    numRows, numCols = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]    #h_upper from offset
        xData1 = geometry[0,1,y]    #w_right from offset            offset == some middle point in rectangle
        xData2 = geometry[0,2,y]    #h_lower_from offset
        xData3 = geometry[0,3,y]    #w_left from offset
        anglesData = geometry[0,4,y]

        for x in range(0, numCols):
            if scoresData[x] < minConfidence:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return [rects, confidences]

def show_boxes_and_image(image, boxes, ratio):

    for (sX, sY, eX, eY) in boxes:
        sX = int(sX* ratio[0])
        sY = int(sY * ratio[1])
        eX = int(eX * ratio[0])
        eY = int(eY * ratio[1])
        cv2.rectangle(image, (sX, sY), (eX,eY), (255,30,30), 2)

    cv2.imshow("meow",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # parser = ap.ArgumentParser()
    # parser.add_argument("-i","--image", type=str, help="path for input image")
    # parser.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
    # parser.add_argument("-c", "--min-confidence", type=float, default = 0.5, help="minimum confidence required"
    #                                                                               "to inspect a region")
    # parser.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be a multiple of 32)")
    # parser.add_argument("--h", "--height", type=int, default=320, help="resized image height (should be a "
    #                                                                    "multiple of 32")


# args = parse_arguments()
#-i /cs/usr/shookies/Desktop/text_finder/test_im.png -east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb -w 608 -h 288
east_path = "/cs/usr/shookies/Desktop/text_finder/east_text_detection.pb"
im_path = "/cs/usr/shookies/Desktop/text_finder/stop_sign.jpg"
h = 320
w = 672
print("meowwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
image, ratio, orig = load_and_resize_image(im_path, h, w)
boxes = detect(east_path, image, 0.5)
show_boxes_and_image(image, boxes, ratio)
# cv2.imshow("meow", im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
