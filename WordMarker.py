
###################### Imports ###########################
import pytesseract as pt
import numpy as np
from pytesseract import Output
import cv2




###################### Constants ###########################



###################### Functions ###########################
pt.pytesseract.tesseract_cmd = 'C:\\Users\\Chaim\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

def mark_words(original_image, word_list, word):
    """

    :param original_image: original image with words that need to be marked.
    :param word_list: list of all the word objects with their coordinates written
    :param word: the word needed to be marked within the list.
    :return: augmented image with marked bounding boxes around the word
    """
    filename = 't3qWG.png'

    img = cv2.imread(filename)
    d = pt.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    print(pt.image_to_string(img))


    for i in range(n_boxes):
        (x, y, w, h) = (
        d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def calculate_bounding_box():
    return

def placeholder():
    return

def main():
    mark_words(None, None, None)

if __name__ == '__main__':
    main()



