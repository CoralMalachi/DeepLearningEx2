STUDENT={'name': 'Coral Malachi_Daniel Braunstein',
         'ID': '314882853_312510167'}

import help_funcs as helper


import numpy as np
from numpy import linalg as la



def main():
    print  str(most_similar("dog", 5))
    print  str(most_similar("england", 5))
    print  str(most_similar("john", 5))
    print  str(most_similar("explode", 5))
    print  str(most_similar("office", 5))



def get_distance(word_dist):
    return word_dist[1]

def most_similar(word, k):
 

    # get the word
    words = helper.dictionary_embed('vocab.txt', 'wordVectors.txt')
    word = words[word]
    # set the distance array
    distances = []
    # loop over words
    for i in words:
        # calc the distance between word and words[i]
        dist = cosine_distance(word, words[i])
        # add the distance to array
        distances.append([i, dist])

    # sort the array
    distances = sorted(distances, key=get_distance)
    # get top k
    top_k = sorted(distances, key=get_distance, reverse=True)[1:k + 1]
    top_k = [item[0] for item in top_k]
    return top_k







def cosine_distance(x, y):

    return (np.dot(x, y)) / (np.max([float(la.norm(x, 2) * la.norm(y, 2)), 1e-8]))


if __name__ == "__main__":
    main()
