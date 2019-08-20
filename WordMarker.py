
###################### Imports ###########################
import pytesseract as pt
import numpy as np
from pytesseract import Output
import cv2




###################### Constants ###########################



###################### Functions ###########################
pt.pytesseract.tesseract_cmd = 'C:\\Users\\Chaim\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

def mark_words(original_image, word_list):
    """

    :param original_image: original image with words that need to be marked.
    :param word_list: list of all the word objects with their coordinates written
    :param word: the word needed to be marked within the list.
    :return: augmented image with marked bounding boxes around the word
    """

    for i in range(len(word_list)):
        (x, y), (w, h) = word_list[i].start, word_list[i].end
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('image', original_image)
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



