import os
import string

def DeleteFiles(path):
    lower_ascii_list = [char for char in string.ascii_lowercase]
    numbers_to_delete = [3,4,5,6,173,174,175,218,219,220,221,278,279,280,281,312,313,314,315,342,417,418,419,420,
                         797,798,799,800,801,802,803,804,880,881,882,883,936,937,938,939,943,946]
    for file in lower_ascii_list:
        for num in numbers_to_delete:
            x = path + '/' + file + '/' + str(num) + '.jpg'
            if os.path.isfile(path + '/' + file + '/' + str(num) + '.jpg'):
                os.remove(path + '/' + file + '/' + str(num) + '.jpg')



DeleteFiles('/cs/usr/punims/Desktop/OCRProject/Preprocessed_Images')