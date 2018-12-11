STUDENT={'name': 'Coral Malachi_Daniel Braunstein',
         'ID': '314882853_312510167'}

import help_funcs as helper


import numpy as np
from numpy import linalg as la

###############################################################
# Function Name:load_dev_data_set
# Function input:dev_file
# Function output:dev data set
# Function Action: loading the dev data set
#
################################################################

def main():
    print  str(most_similar("dog", 5))
    print  str(most_similar("england", 5))
    print  str(most_similar("john", 5))
    print  str(most_similar("explode", 5))
    print  str(most_similar("office", 5))


###############################################################
# Function Name:function
# Function input:word, k
# Function output:list
# Function Action:our function creates a list of the k most similar words
#and return it
################################################################
def most_similar(word, k):
    distances = []# set the distance array
    # get the word
    words = helper.dictionary_embed('vocab.txt', 'wordVectors.txt')
    word = words[word]


    # loop over words
    for i in words:
        # calc the distance between word and words[i]
        dist = cosine_distance(word, words[i])
        # add the distance to array
        distances.append([i, dist])

    # sort the array
    distances = sorted(distances, key=get_distance)
    # get top k
    m_top_k = sorted(distances, key=get_distance, reverse=True)[1:k + 1]
    m_top_k = [item[0] for item in m_top_k]
    return m_top_k

###############################################################
# Function Name:cosine_distance
# Function input:x, y
# Function output:distance
# Function Action:the function compute The distance between
#x and y and return it
################################################################
def cosine_distance(x, y):


    return (np.dot(x, y)) / (np.max([float(la.norm(x, 2) * la.norm(y, 2)), 1e-8]))

###############################################################
# Function Name:get_distance
# Function input:word_dist
# Function output:distance
# Function Action: return he distance
#
################################################################
def get_distance(word_dist):
    return word_dist[1]

if __name__ == "__main__":
    main()
