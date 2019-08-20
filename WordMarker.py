
###################### Imports ###########################
import pytesseract as pt
import numpy as np
import cv2




###################### Constants ###########################



###################### Functions ###########################
pt.pytesseract.tesseract_cmd = '/cs/usr/punims/Desktop/tesseract-ocr-setup-3.02.02/tesseract.exe'

def mark_words(original_image, word_list, word):
    """

    :param original_image: original image with words that need to be marked.
    :param word_list: list of all the word objects with their coordinates written
    :param word: the word needed to be marked within the list.
    :return: augmented image with marked bounding boxes around the word
    """
    filename = t3qWG.png

    img = cv2.imread(filename)
    h, w = img.shape

    boxes = pt.image_to_boxes(img)
    print(pt.image_to_string(img))

    for b in boxes:
        b = b.split()
        cv2.rectangle(img, ((int(b[1]), h - int(b[2]))), ((int(b[3]), h - int(b[4]))), (0, 255, 0), 2)
    return

def calculate_bounding_box():
    return

def placeholder():
    return

def main():
    mark_words(None, None, None)

if __name__ == '__main__':
    main()



