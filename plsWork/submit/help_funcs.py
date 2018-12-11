STUDENT={'name': 'Coral Malachi_Daniel Braunstein',
         'ID': '314882853_312510167'}

import numpy as np
#this file contains helper functions to deal with the training set and the test set

#let's define some global list:
Representation_Of_Indexes_By_Words = {}
Representation_Of_Words_By_Indexes = {}
Representation_Of_classes_By_Indexes ={}
Representation_Of_Indexes_By_classes ={}
#define set let us assign an index to each unique word - avoid giving the same word many indexes
Dictionary_of_classes = set()
Dictionary_of_words = set()


#more to define:
UNK = "UUUNKKK"
WINDOW_START = "START_WIN"
WINDOW_END = "END_WIN"
NEW_LINE = "\n"
TAB = "\t"

###############################################################
# Function Name:bring_test_data
# Function input:
# Function output:none
# Function Action:
#
################################################################
def bring_test_data(test_data_set):
    word_sequences = reading_test_dataset(test_data_set)
    return divide_word_sequence_into_windows(word_sequences)

###############################################################
# Function Name:
# Function input:
# Function output:none
# Function Action:
#
################################################################
def add_class_and_word_to_dics(m_class,word):
    global Dictionary_of_words
    global Dictionary_of_classes
    Dictionary_of_classes.add(m_class)
    Dictionary_of_words.add(word)

###############################################################
# Function Name:read_train_data
# Function input:file_name
# Function output:none
# Function Action:training data
#
################################################################
def read_train_data(train_dataset):

    # We use global keyword to read and write a global variable inside a function.
    global Dictionary_of_words
    global Dictionary_of_classes

    m_seqs = []#declere an empty list

    with open(train_dataset) as file:
        m_lines = file.readlines()
        sequence = []
        for m_line in m_lines:
            if NEW_LINE == m_line:
                m_seqs.append(sequence)
                # clean buffer
                sequence =[]
                # continue yo next iteration of the loop
                continue
            m_line = clean_line(m_line)
            word, tag  = m_line.split()
            add_class_and_word_to_dics(tag, word)
            sequence.append((word, tag))

    #add unkown word to both dicts
    Dictionary_of_classes.add(UNK)
    Dictionary_of_words.add(UNK)
    return m_seqs

###############################################################
# Function Name:clean_line
# Function input:line
# Function output:none
# Function Action:# remove new line from string
# # all leading and trailing whitespaces are removed from the string.
#
################################################################
def clean_line(line):
    return line.strip(NEW_LINE).strip().strip(TAB)

###############################################################
# Function Name:read_dev_data
# Function input:file_name
# Function output:none
# Function Action:
#
################################################################
def read_dev_data(file_name):
    m_seq_with_class = []

    # We use global keyword to read and write a global variable inside a function.
    global Dictionary_of_words
    global Dictionary_of_classes

    with open(file_name) as file:
        #call readlines function
        m_lines = file.readlines()
        m_seq = []
        for line in m_lines:
            if NEW_LINE == line:
                m_seq_with_class.append(m_seq)
                # clean buffer
                m_seq = []
                # continue yo next iteration of the loop
                continue
            line = clean_line(line)
            word, m_class = line.split()
            m_seq.append((word, m_class))

    return m_seq_with_class


###############################################################
# Function Name:updating_dictionaries_set
# Function input:none
# Function output:none
# Function Action:update all the dcts we created above
#
################################################################
def updating_dictionaries_set():
    # We use global keyword to read and write a global variable inside a function.
    global Representation_Of_Words_By_Indexes
    global Representation_Of_Indexes_By_Words
    global Representation_Of_classes_By_Indexes
    global Representation_Of_Indexes_By_classes
    print("here2")
    print(len(Dictionary_of_words))
    Dictionary_of_words.update(set([WINDOW_START, WINDOW_END]))
    print("after2")
    print(len(Dictionary_of_words))
    Representation_Of_Words_By_Indexes = {
        m_word : m_index for m_index, m_word in enumerate(Dictionary_of_words)
    }
    Representation_Of_Indexes_By_Words = {
        m_index : m_word for m_word, m_index in Representation_Of_Words_By_Indexes.iteritems()
    }
    Representation_Of_classes_By_Indexes = {
        m_class : m_index for m_index, m_class in enumerate(Dictionary_of_classes)
    }
    Representation_Of_Indexes_By_classes = {
        m_index : m_class for m_class, m_index in Representation_Of_classes_By_Indexes.iteritems()
    }

###############################################################
# Function Name:updating_dictionaries_set
# Function input:none
# Function output:none
# Function Action:
#
################################################################

def get_word_embeddings_dict_from_file(m_words, m_vec):

    word_embeddings_dict = {}
    #use for loop
    for m_word, m_vec in izip(open(m_words), open(m_vec)):
        m_word = m_word.strip("\n").strip()
        m_vec = m_vec.strip("\n").strip().split(" ")
        #assign to dictionary
        word_embeddings_dict[m_word] = np.asanyarray(map(float,m_vec))
    return word_embeddings_dict

###############################################################
# Function Name:divide_word_class_sequence_into_windows
# Function input:word_sequences
# Function output:none
# Function Action:return an array of windows when each window contain 5 words (as requiredin the assigment)
#
################################################################
def divide_word_class_sequence_into_windows(word_sequences):
    # define an empty array to return
    windows_array = []
    classes = []
    for sentence in word_sequences:
        words_classes_matrix = []
        #words_classes_matrix = [(WINDOW_START, WINDOW_START), (WINDOW_START, WINDOW_START)]
        words_classes_matrix.extend([(WINDOW_START, WINDOW_START),
                                     (WINDOW_START, WINDOW_START)])
        words_classes_matrix.extend(sentence)
        words_classes_matrix.extend([(WINDOW_END, WINDOW_END), (WINDOW_END, WINDOW_END)])
        for i, (cure_word,tag) in enumerate(words_classes_matrix):
            #check that cure word is not start or end ( we created them )
            if cure_word!=WINDOW_START and cure_word !=WINDOW_END:
                #create a new window
                new_window = []
                new_window.append(convert_word_to_index(words_classes_matrix[i - 2][0]))
                new_window.append(convert_word_to_index(words_classes_matrix[i - 1][0]))
                new_window.append(convert_word_to_index(cure_word))
                new_window.append(convert_word_to_index(words_classes_matrix[i + 1][0]))
                new_window.append(convert_word_to_index(words_classes_matrix[i - 2][0]))

                windows_array.append(new_window)
                classes.append(Representation_Of_classes_By_Indexes[tag])
    return windows_array, classes
###############################################################
# Function Name:divide_word_sequence_into_windows
# Function input:word_sequences
# Function output:return an array of windows when each window contain 5 words (as requiredin the assigment)
# Function Action:for test set which not has the class for each word
#
################################################################
def divide_word_sequence_into_windows(word_sequences):
    # define an empty array to return
    windows_array = []
    for sequence in word_sequences:
        # words = [WINDOW_START,WINDOW_START]
        # words.extend(sentence)
        # words.extend([WINDOW_END,WINDOW_END])
        words = []
        # adds start*2, end*2 for each sentence for appropriate windows
        words.append(WINDOW_START)
        words.append(WINDOW_START)
        words.extend(sequence)
        words.append(WINDOW_END)
        words.append(WINDOW_END)
        for index, (cure_word) in enumerate(words):
            if cure_word != WINDOW_START and cure_word != WINDOW_END:
                new_window = []
                new_window.append(convert_word_to_index(words[index - 2]))
                new_window.append(convert_word_to_index(words[index - 1]))
                new_window.append(convert_word_to_index(cure_word))
                new_window.append(convert_word_to_index(words[index + 1]))
                new_window.append(convert_word_to_index(words[index - 2]))

                windows_array.append(new_window)
    return windows_array

###############################################################
# Function Name:convert_word_to_index
# Function input:word_to_convert
# Function output:index
# Function Action:dwon explain
#
################################################################
def convert_word_to_index(word_to_convert):
    """
       The first step in using an embedding layer is to encode this sentence by indices.
        In this case we assign an index to each unique word

       :param word_to_convert:
       :return: return the index which represent the word.
       """
    if word_to_convert not in Representation_Of_Words_By_Indexes:
        #if the word is not in the list its index is UNK - Unknown
        #words we have not seen in the training action we'll map them to the same vector in
        #the embedding matrix - unk
        return Representation_Of_Words_By_Indexes[UNK]
    else:
        #if the word is in the list, return its index
        return Representation_Of_Words_By_Indexes[word_to_convert]




###############################################################
# Function Name:get_dev_data
# Function input:dev_file
# Function output:none
# Function Action:the function return the dev data
#
################################################################
def get_dev_data(dev_file):
    # We use global keyword to read and write a global variable inside a function.
    global Dictionary_of_words
    global Dictionary_of_classes
    word_sequences = read_dev_data(dev_file)
    #call the divide_word_class_sequence_into_windows
    sequence, m_class = divide_word_class_sequence_into_windows(word_sequences)
    return sequence, m_class


###############################################################
# Function Name:reading_test_dataset
# Function input:dataset_to_read
# Function output:test data
# Function Action:the function return the test data
#
################################################################
def reading_test_dataset(dataset_to_read):
    # define an empty array to return
    word_sequences = []

    with open(dataset_to_read) as dataset:
        content = dataset.readlines()
        sequence = []
        for cure_line in content:
            if NEW_LINE == cure_line:
                word_sequences.append(sequence)
                # clean buffer
                sequence =[]
                # continue yo next iteration of the loop
                continue
            word_in_sequence = cure_line.strip(NEW_LINE).strip()
            sequence.append(word_in_sequence)
    return word_sequences


###############################################################
# Function Name:get_train_data
# Function input:m_dataset
# Function output:return the train data
# Function Action:the function return the train data
#
################################################################
def get_train_data(m_dataset):


    # We use global keyword to read and write a global variable inside a function.
    global Dictionary_of_words
    global Dictionary_of_classes
    #call read_train_data
    train_data = read_train_data(m_dataset)
    #call updating_dictionaries_set function
    updating_dictionaries_set()
    #call divide_word_class_sequence_into_windows functin
    windows_array, classes = divide_word_class_sequence_into_windows(train_data)
    return windows_array, classes







