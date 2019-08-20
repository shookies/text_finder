

############# Constants #############


#TODO add substring functionality

############# Word class #############


class word:


    def __init__(self, theWord, xs, ys, xe, ye, page):

        self.start = (xs, ys)
        self.end = (xe, ye)
        self._string = theWord
        self.pageNum = page

    def get_coors(self): return [self.start, self.end]  #<----- list of tuples

    def get_string(self): return self._string

    def get_substring(self, substr):
        if (substr in self._string):
            pass    #TODO handle substring coordinates





############# Global Variables #############


word_list = [] #keys: the words themselves as strings. values: the word class wrapper


############# Functions #############

def add_word(word_string, start, end, pageNum = 1):

    #TODO input checks?
    _word = word(word_string, start[0], start[1], end[0], end[1], pageNum)
    word_list.append(_word)

def get_words(word_string):

    ret_list = []

    for i in range (len(word_list)):
        if word_string in word_list[i].get_string():
            ret_list.append(word_list[i])

    return ret_list



