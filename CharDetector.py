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

def preprocess(im, dilation_kernel):

    #grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


    #binary
    _ , thresh = cv2.threshold(gray, 127,255,cv2.THRESH_BINARY_INV)


    #dilation
    im_dilation = cv2.dilate(thresh, dilation_kernel, iterations=1)

    return im_dilation


def find_BB(im, orig, threshold):

    ctrs, heir = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda cont: cv2.boundingRect(cont)[0])    #sorts contours by upper left point

    boxes = []
    chars = []

    for i, cont in enumerate(sorted_ctrs):

        x, y, w, h = cv2.boundingRect(cont)     #get bounding box
        ROI = orig[y:y+h, x:x+w]


        if w > threshold and h > threshold:   #eliminating debris
            chars.append(ROI)
            boxes.append([x,x+w, y, y+h])

    return [chars, boxes]

def detect_chars(image, dilation_kernel, debris_threshold):

    # im_path = parse_args()
    # im = load_image(im_path)
    # kernel = np.ones((2,1), np.uint8)
    processed_im = preprocess(image.copy(), dilation_kernel)
    # cv2.imshow("processed",processed_im)
    return find_BB(processed_im, image, debris_threshold)


# detect_chars()