import numpy as np
import cv2, os, tempfile, pdf2image
import argparse as ap


#TODO change preprocess values for words instead of characters (args)

def parse_args():

    parser = ap.ArgumentParser()
    parser.add_argument("-i","-im_path", type=str,help="path for input image")
    args = parser.parse_args()
    im_path = args.i

    return im_path


def load_image(im_path):

    if im_path.endswith(".pdf"):
        # args.image = pdf2image.convert_from_path(args.image)
        with tempfile.TemporaryDirectory() as path:
            images_from_path = pdf2image.convert_from_path(im_path, output_folder=path, last_page=1, first_page =0)

            #TODO handle multiple pages
        mat_from_image = np.array(images_from_path[0])
        image = cv2.cvtColor(mat_from_image, cv2.COLOR_RGB2BGR)

    else:
        image = cv2.imread(im_path)

    return image



#####NOTE#####: The following functions were changed to support the creation of the dilation_kernel here instead of
#               in TextDetector.py. For som reason when the kernel was created there it would not have any effect on
#               preprocessing.

def preprocess(im, BB): #TODO fix dilation for smaller fonts. maybe try something other than that.

    #grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


    #binary
    _ , thresh = cv2.threshold(gray, 127,255,cv2.THRESH_BINARY_INV)


    #dilation
    dilation_kernel = create_kernel(im,BB)
    print(dilation_kernel.shape)
    im_dilation = cv2.dilate(thresh, dilation_kernel, iterations=1)

    #erosion
    ker_dims = (1,1)
    erosion_kernel = np.ones(ker_dims,np.uint8)
    im_erosion = cv2.erode(im_dilation,erosion_kernel,iterations=1)

    return im_erosion

def detect_chars(image, word, BB):


    dilation_kernel, padding, debris_threshold = define_parameters(image,BB)    #TODO wasnt working: dilation_kernel was made here and was a parameter for preprocess.
    processed_im = preprocess(word.copy(), BB)
    # cv2.imshow("processed",processed_im)
    return find_BB(processed_im, word, debris_threshold)

def create_kernel(image, BB):

    ratio = calculate_ratio(image,BB)
    height_constant = 15
    width_constant = 2
    ker_dims = (int(ratio * height_constant), int(ratio * width_constant))
    return np.ones(ker_dims, np.uint8)

def define_parameters(image, BB):

    ratio = calculate_ratio(image,BB)
    height_constant = 10000000
    width_constant = 2

    padding_constant = 12
    debris_threshold_constant = 5
    # print(ratio * height_constant)
    print(ratio)
    # ker_dims = (int(ratio * height_constant), int((ratio ** (-1)) * (ratio ** (-1))))
    ker_dims = (0,0)
    dilation_kernel = np.ones(ker_dims, np.uint8)
    padding = int(ratio * padding_constant)
    debris_threshold = int(ratio * debris_threshold_constant)

    return [dilation_kernel, padding, debris_threshold]

def calculate_ratio(image,BB):

    #Assumes axis-aligned bounding box (not rotated)
    im_h, im_w = image.shape[:2]
    bb_h = BB[3] - BB[1]
    bb_w = BB[2] - BB[0]
    im_area = im_h * im_w
    bb_area = bb_h * bb_w

    return float(bb_h / im_h)


##############################################################################################

def find_BB(im, orig, threshold):

    ctrs, heir = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda cont: cv2.boundingRect(cont)[0])    #sorts contours by upper left point

    boxes = []
    chars = []

    for i, cont in enumerate(sorted_ctrs):

        x, y, w, h = cv2.boundingRect(cont)     #get bounding box
        ROI = orig[y:y+h, x-1:x+w+1]


        if w > threshold and h > threshold:   #eliminating debris
            chars.append(ROI)
            boxes.append([x,x+w, y, y+h])

    return [chars, boxes]

# def detect_chars(image, dilation_kernel, debris_threshold):
#
#     # im_path = parse_args()
#     # im = load_image(im_path)
#     # kernel = np.ones((2,1), np.uint8)
#     processed_im = preprocess(image.copy(), dilation_kernel)
#     # cv2.imshow("processed",processed_im)
#     return find_BB(processed_im, image, debris_threshold)


# detect_chars()