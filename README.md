# text_finder -- WORK IN PROGRESS --
We started this project in the midst of summer while on break from university. The object of this project
is to use our aquired skills from courses in order to create a simple chrome extension for finding text in images
(Mainly PDF scans of text). 

The code is mainly divided into 3 modules:

-TextDetector.py: 
  Recieves input jpeg (or converts pdf file to jpg) and pre-processes it for detection. using the native OpenCV
  text detection CNN (EAST) the module detects words in the image and places bounding boxes around them in order
  to convert them into strings.

-CharDetector.py:
  Detects characters in words using classic image processing applications.

-MLmodels.py:
  A CNN model recieving 32x32 images of letters and converts them to characters in order to concatenate them into strings
  for searching.


# created by: Tom Marom, Edan Patt
