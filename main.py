import TextDetector, MLmodels, WordFinder
import tensorflow.keras as tf







def main():

    # load model
    model = tf.load_model('pathname')
    # summarize model.
    model.summary()

    #get number:

    model.predict() # probably returns one hot encoded vector.

    args = TextDetector.parse_arguments()
    im_path = args.image
    east_path = args.east
    minConfidence = args.min_confidence
    image, ratio, orig, w, h = TextDetector.load_and_resize_image(im_path)
    chars, words = TextDetector.detect(east_path, image, minConfidence, h, w)    #chars = 32x32 chars, words = [word_matrix, num_of_characters]
    # a = [[1,2],[3,4],[5,6]]
    # b = np.array(a)
    # print(b.shape)
    # print()
    # print(b)
    # print("padding:")
    # c = np.pad(b,((1,2),(1,2)),mode="constant",constant_values=0)
    # print(c)

    return 0