
###################### Imports ###########################
import pytesseract as pt
import numpy as np
from pytesseract import Output
import cv2
import WordFinder
import WordMarker




###################### Constants ###########################



###################### Functions ###########################
pt.pytesseract.tesseract_cmd = 'C:\\Users\\Chaim\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

def func():

    filename = 't3qWG.png'

    img = cv2.imread(filename)
    d = pt.image_to_data(img, output_type=Output.DICT)
    word_list =d['text']
    print(pt.image_to_string(img))

    for i in range(len(word_list)):
        WordFinder.add_word(word_list[i], (d['left'][i], d['top'][i]), (d['width'][i], d['height'][i]))
    unique_word_list = WordFinder.get_words("Adobe")
    WordMarker.mark_words(img, unique_word_list)





def main():
    func()

main()

