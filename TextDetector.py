
import numpy as np
import cv2, math, pdf2image, os, tempfile
import argparse as ap

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

    if im_path.endswith(".pdf"):
        # args.image = pdf2image.convert_from_path(args.image)
        with tempfile.TemporaryDirectory() as path:
            images_from_path = pdf2image.convert_from_path(im_path, output_folder=path, last_page=1, first_page =0)

                                                                #TODO handle multiple pages
        mat_from_image = np.array(images_from_path[0])
        image = cv2.cvtColor(mat_from_image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(im_path)

    original = image.copy()
    height, width = image.shape[:2]
    newHeight = height if (height % 32 == 0) else round_down(height,32)     #rounds down to the nearest multiple of 32
                                                                            #in order for EAST to work with the image
    newWidth = width if (width % 32 == 0) else round_down(width,32)
    print(newHeight, newWidth)
    ratio_width = width / float(newWidth)        #calculates ratio for later use (upscaling image back)
    ratio_height = height / float(newHeight)
    ratio = (ratio_width, ratio_height)

    image = cv2.resize(image, (newWidth, newHeight))
    return [image, ratio, original, newWidth, newHeight]

def detect(east_path, image, minConfidence, nmsThresh, inputHeight, inputWidth, dilation_kernel):

    model = cv2.dnn.readNet(east_path)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor= 1.0, size= (width,height), mean= (123.68,116.78, 103.94),
                                 swapRB= True, crop= False)     #preprocess the image for detection
    model.setInput(blob)
    scores, geometry = model.forward([
        "feature_fusion/Conv_7/Sigmoid",        #probability of each region containing text
        "feature_fusion/concat_3"])             #used for deriving the bounding boxes


    rects, confidences = decode(scores, geometry, minConfidence)
    indices = cv2.dnn.NMSBoxesRotated(rects, confidences, minConfidence, nmsThresh)
    word_boxes = []
    char_boxes = []
    for i in indices:
        vertices = cv2.boxPoints(rects[i[0]])   # get 4 corners of the rotated rect

        for j in range(4):      # scale the bounding box coordinates based on the respective ratios
            vertices[j][0] *= (width / float(inputWidth))
            vertices[j][1] *= (height / float(inputHeight))
        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            cv2.line(image, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
        word_boxes.append(vertices)

    for BB in word_boxes:
        # cv2.imshow("word", word)
        word = image[BB[0]:BB[1], BB[2]:BB[3]]
        chars = CharDetector.detect_chars(word.astype('uint8'),dilation_kernel)
        char_boxes.append(chars)
        for char in chars:
            cv2.rectangle(image,(char[0],char[1]), (char[2], char[3]), (255,0,0),1)


    #REMINDER: opencv points are plotted as (x,y) NOT (row,column), and the y axis is positive downward.


    cv2.imshow("meow",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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

            # Calculate offset
            offset = ([offsetX + cosA * xData1[x] + sinA * xData2[x], offsetY - sinA * xData1[x] + cosA * xData2[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            rects.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    return [rects, confidences]


def main():
    args = parse_arguments()

    im_path = args.image
    east_path = args.east
    nms = args.nms
    minConfidence = args.min_confidence
    image, ratio, orig, w, h = load_and_resize_image(im_path)
    dilation_kernel = np.ones((2,1), np.uint8)
    detect(east_path, image, minConfidence, nms, h, w, dilation_kernel)

    return 0

main()

# -i /cs/usr/shookies/Desktop/text_finder/stop_sign.jpg -east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb
#-i /cs/usr/shookies/Desktop/text_finder/test_im.png -east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb
# --i /cs/usr/shookies/Desktop/text_finder/test2.png --east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb --width 512 --height 256
# --i /cs/usr/shookies/Desktop/text_finder/lines.png --east /cs/usr/shookies/Desktop/text_finder/east_text_detection.pb
#
#
# # Import required modules
# import cv2 as cv
# import math
# import argparse
#
# parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# # Input argument
# parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# # Model argument
# parser.add_argument('--model', default="frozen_east_text_detection.pb",
#                     help='Path to a binary .pb file of model contains trained weights.'
#                     )
# # Width argument
# parser.add_argument('--width', type=int, default=320,
#                     help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
#                     )
# # Height argument
# parser.add_argument('--height',type=int, default=320,
#                     help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
#                     )
# # Confidence threshold
# parser.add_argument('--thr',type=float, default=0.5,
#                     help='Confidence threshold.'
#                     )
# # Non-maximum suppression threshold
# parser.add_argument('--nms',type=float, default=0.4,
#                     help='Non-maximum suppression threshold.'
#                     )
#
# args = parser.parse_args()
#
#
# ############ Utility functions ############
# def decode(scores, geometry, scoreThresh):
#     detections = []
#     confidences = []
#
#     ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
#     assert len(scores.shape) == 4, "Incorrect dimensions of scores"
#     assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
#     assert scores.shape[0] == 1, "Invalid dimensions of scores"
#     assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
#     assert scores.shape[1] == 1, "Invalid dimensions of scores"
#     assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
#     assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
#     assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
#     height = scores.shape[2]
#     width = scores.shape[3]
#     for y in range(0, height):
#
#         # Extract data from scores
#         scoresData = scores[0][0][y]
#         x0_data = geometry[0][0][y]
#         x1_data = geometry[0][1][y]
#         x2_data = geometry[0][2][y]
#         x3_data = geometry[0][3][y]
#         anglesData = geometry[0][4][y]
#         for x in range(0, width):
#             score = scoresData[x]
#
#             # If score is lower than threshold score, move to next x
#             if(score < scoreThresh):
#                 continue
#
#             # Calculate offset
#             offsetX = x * 4.0
#             offsetY = y * 4.0
#             angle = anglesData[x]
#
#             # Calculate cos and sin of angle
#             cosA = math.cos(angle)
#             sinA = math.sin(angle)
#             h = x0_data[x] + x2_data[x]
#             w = x1_data[x] + x3_data[x]
#
#             # Calculate offset
#             offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])
#
#             # Find points for rectangle
#             p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
#             p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
#             center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
#             detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
#             confidences.append(float(score))
#
#     # Return detections and confidences
#     return [detections, confidences]
#
# if __name__ == "__main__":
#     # Read and store arguments
#     confThreshold = args.thr
#     nmsThreshold = args.nms
#     inpWidth = args.width
#     inpHeight = args.height
#     model = args.model
#
#     # Load network
#     net = cv.dnn.readNet(model)
#
#     # Create a new named window
#     kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
#     cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
#     outputLayers = []
#     outputLayers.append("feature_fusion/Conv_7/Sigmoid")
#     outputLayers.append("feature_fusion/concat_3")
#
#     # Open a video file or an image file or a camera stream
#     cap = cv.VideoCapture(args.input if args.input else 0)
#
#     while cv.waitKey(1) < 0:
#         # Read frame
#         hasFrame, frame = cap.read()
#         if not hasFrame:
#             cv.waitKey()
#             break
#
#         # Get frame height and width
#         height_ = frame.shape[0]
#         width_ = frame.shape[1]
#         rW = width_ / float(inpWidth)
#         rH = height_ / float(inpHeight)
#
#         # Create a 4D blob from frame.
#         blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
#
#         # Run the model
#         net.setInput(blob)
#         output = net.forward(outputLayers)
#         t, _ = net.getPerfProfile()
#         label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
#
#         # Get scores and geometry
#         scores = output[0]
#         geometry = output[1]
#         [boxes, confidences] = decode(scores, geometry, confThreshold)
#         # Apply NMS
#         indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
#         for i in indices:
#             # get 4 corners of the rotated rect
#             vertices = cv.boxPoints(boxes[i[0]])
#             # scale the bounding box coordinates based on the respective ratios
#             for j in range(4):
#                 vertices[j][0] *= rW
#                 vertices[j][1] *= rH
#             for j in range(4):
#                 p1 = (vertices[j][0], vertices[j][1])
#                 p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
#                 cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
#                 # cv.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
#
#         # Put efficiency information
#         cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
#
#         # Display the frame
#         cv.imshow(kWinName,frame)
#         cv.imwrite("out-{}".format(args.input),frame)