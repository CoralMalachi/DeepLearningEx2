
STUDENT={'name': 'Coral Malachi_Daniel Braunstein',
'ID': '314882853_312510167'}
import numpy as np

import help_funcs as helper
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from torch.utils.data import TensorDataset

import sys

NEW_LINE = "\n"
TAB = "\t"
# let's define our parameters:

learning_rate = 0.005
window_size = 5  # 2 neighbors + target word
DIM_HIDDEN_LAYER = 150
tags_number = 10
suffix_size = 3
prefix_size = 3
DIM_INPUT = 250  # window x vector_size = 5 x 50 =250
NUM_OF_EPOCHS = 3
VECTOR_EMBEDDINGS_DIM = 50

BATCH_NORMALIZATION_SIZE = 1024  # denotes the number of samples contained in each generated batch.


def create_paths_for_preds(ner_or_pos):
    """

    :param ner_or_pos:
    :return: the func return paths to the test dataset and to the test pred
    """
    path_to_test_file = ner_or_pos + "/test"
    path_to_pred_file = "test3." + ner_or_pos

    return path_to_test_file,path_to_pred_file

###############################################################
# Function Name:load_test_set
# Function input:the file to load contains the test data
# Function output:return the test data
# Function Action: get the test data
#
################################################################

def load_test_set(file_name):

    data_set_test = helper.bring_test_data(file_name)
    return data_set_test


class Part1Model(object):

    ###############################################################
    # Function Name: init function on the class - constractor
    # Function input: kind_of_optim, training_set, nn_model, test_data,dev_dataset
    # Function output:a model object with the next methods
    # Function Action:The __init__ method is roughly what represents a constructor in Python    #
    ################################################################
    def __init__(self, kind_of_optim, training_set, nn_model, test_data, dev_dataset):

        self.kind_of_optim = kind_of_optim
        self.training_set = training_set
        self.nn_model = nn_model
        self.test_data = test_data
        self.dev_dataset = dev_dataset


        ###############################################################
        # Function Name:forward_test_data
        # Function input:ner_or_pos
        # Function output:none
        # Function Action:This is the forward propagation function
        #     # #feed the neural networks firts time, and use activition function
        #     # #to convert values to be between 0 to 1
        ################################################################

    def forward_test_data(self, ner_or_pos):
        self.nn_model.eval()
        path_to_test_file, path_to_pred_file = create_paths_for_preds(ner_or_pos)
        pred_list = []
        for data in self.test_data:
            the_model = self.nn_model(torch.LongTensor(data))

            # call get_y_tag to find y_hat

            pred = get_y_tag(the_model)

            # append y_hat in the file of preds
            pred_list.append(pred.item())

        # define the path to the test file - from where we'll get the data
        pred_list = self.Get_Representation_Of_Indexes_By_classes(pred_list)
        # path_to_test_file = ner_or_pos + "/test"
        self.create_predictions_file(path_to_test_file, path_to_pred_file, pred_list)


    ###############################################################
    # Function Name:Get_Representation_Of_Indexes_By_classes
    # Function input:lst_of_index_to_convert
    # Function output:return the class which the index is
    # Function Action:    return the class which the index is
    ################################################################

    def Get_Representation_Of_Indexes_By_classes(self, lst_of_index_to_convert):
        return [helper.Representation_Of_Indexes_By_classes[i] for i in lst_of_index_to_convert]

    ###############################################################
    # Function Name:Start_Action
    # Function input:ner_or_pos - a info which tell what kind of dataset we are dealing with
    # Function output:none
    # Function Action:the function runing a loop (epoch) which in each iteration it train the
    # network and The function produces two graphs that describe the loss of the function and
    # its accuracy percentage for the training set and test set at each iteration
    ################################################################
    def Start_Action(self, ner_or_pos):
        m_precent_of_accuracy_on_dev = {}
        # define 3 empty list to save our loss data suring the loop
        m_loss_training = {}
        m_loss_dev = {}
        # runing in a loop as number of epoch we defined
        for epoch in range(NUM_OF_EPOCHS):
            print "epoch number " + str(epoch)
            # call the tairn function to start the training progress
            self.train_neural_network(epoch, m_loss_training)
            self.feedforward_the_dev_set(epoch, m_loss_dev, m_precent_of_accuracy_on_dev, ner_or_pos)
        create_match_ploting(m_loss_dev, m_precent_of_accuracy_on_dev)
        # call the forward_test_data function to find the predictions for unseen data
        self.forward_test_data(ner_or_pos)

    ###############################################################
    # Function Name:train_neural_network
    # Function input:model, trainng and validation sets
    # Function output:none
    # Function Action:train on the training set and then test the
    # network on the test set. This has the network make predictions on data it has never seen
    ################################################################

    def train_neural_network(self, epoch, avg_train_loss_per_epoch_dict):

        self.nn_model.train()
        train_loss = 0
        m_success = 0

        # looping the training set
        for data, labels in self.training_set:
            self.kind_of_optim.zero_grad()
            m_model = self.nn_model(data)
            # call help function to compute the predicted y
            y_hat = get_y_tag(m_model)
            m_success += y_hat.eq(labels.data.view_as(y_hat)).cpu().sum().item()
            # computing the loss
            loss = F.nll_loss(m_model, labels)
            train_loss += loss
            # start back action
            loss.backward()
            # updating parameters
            self.kind_of_optim.step()

        train_loss /= (len(self.training_set))
        avg_train_loss_per_epoch_dict[epoch] = train_loss
        # calculat length
        length = len(self.training_set) * BATCH_NORMALIZATION_SIZE
        #calculat accuracy
        accuracy = 100. * m_success / (len(self.training_set) * BATCH_NORMALIZATION_SIZE)

        print_message_each_epoch(1, length, epoch, train_loss, m_success, BATCH_NORMALIZATION_SIZE, accuracy)

    ###############################################################
    # Function Name:feedforward_the_dev_set
    # Function input:iter_number, dev_loss_dictionary,dev_precents_of_accuracy_dictionary, nerOrPos
    # Function output:none
    # Function Action:This is the forward propagation function
    # #feed the neural networks firts time, and use activition function
    # #to convert values to be between 0 to 1
    ################################################################

    def feedforward_the_dev_set(self, iter_number, dev_loss_dictionary,
                                dev_precents_of_accuracy_dictionary, nerOrPos):

        # let the model know to switch to eval mode by calling .eval() on the model
        self.nn_model.eval()
        # define varibles for loss, and number of correct prediction
        m_loss = 0
        m_success = 0
        m_count = 0
        for data, target in self.dev_dataset:
            # let the model know to switch to eval mode by calling .eval() on the model
            m_model = self.nn_model(data)
            m_loss = F.nll_loss(m_model, target, size_average=False).item() + m_loss
            y_hat = get_y_tag(m_model)
            if nerOrPos == 'ner':
                if helper.Representation_Of_Indexes_By_classes[y_hat.cpu().sum().item()] != 'O' or \
                        helper.Representation_Of_Indexes_By_classes[target.cpu().sum().item()] != 'O':
                    m_success = y_hat.eq(target.data.view_as(y_hat)).cpu().sum().item() + m_success
                    m_count = 1 + m_count
            else:
                m_count = 1 + m_count
                m_success = y_hat.eq(target.data.view_as(y_hat)).cpu().sum().item() + m_success


        m_loss /= len(self.dev_dataset)
        dev_loss_dictionary[iter_number] = m_loss

        #calculate accuracy of data
        accuracy = 100. * m_success / m_count
        dev_precents_of_accuracy_dictionary[iter_number] = accuracy

        print_message_each_epoch(0, m_count, iter_number, m_loss, m_success, 1, accuracy)




    ###############################################################
    # Function Name:create_predictions_file
    # Function input:test_file_name, output_file_name, list_of_y_hats
    # Function output:none
    # Function Action:#save resluts into predictions file
    #
    ################################################################

    def create_predictions_file(self, test_file_name, output_file_name, list_of_y_hats):

        with open(test_file_name, 'r') as unseen_data, open(output_file_name, 'w') as preds_file:
            m_lines_of_file = unseen_data.readlines()
            m_count = 0#set count to zero
            # do a for loop
            for m_cure_line in m_lines_of_file:

                # check if current line is a new line
                if m_cure_line == NEW_LINE:
                    # if so add this to the predictions file
                    preds_file.write(m_cure_line)
                else:
                    preds_file.write(m_cure_line.strip(NEW_LINE) + " " + list_of_y_hats[m_count] + NEW_LINE)
                    m_count = 1 + m_count#update the count value!


class NeuralNet(nn.Module):

    def __init__(self, trained, input_size):
        super(NeuralNet, self).__init__()

        self.trained = trained

        if trained:

            self.E = nn.Embedding(helper.E.shape[0], helper.E.shape[1])
            self.E.weight.data.copy_(torch.from_numpy(helper.E))
        else:
            self.E = nn.Embedding(len(helper.Dictionary_of_words), VECTOR_EMBEDDINGS_DIM)

        #calc and set the input size
        self.input_size = window_size * VECTOR_EMBEDDINGS_DIM
        self.fc0 = nn.Linear(input_size, DIM_HIDDEN_LAYER)

        self.fc1 = nn.Linear(DIM_HIDDEN_LAYER, len(helper.Dictionary_of_classes))


        self.m_pref = {word[:prefix_size] for word in helper.Dictionary_of_words}
        self.m_suffs = {word[:-suffix_size] for word in helper.Dictionary_of_words}
        self.m_pref = list(self.m_pref)
        self.m_suffs = list(self.m_suffs)

        self.prefix_to_index = {suff: i for i, suff in enumerate(self.m_pref)}

        self.suffix_to_index = {suff: i for i, suff in enumerate(self.m_suffs)}

        self.E_suff = nn.Embedding(len(self.m_suffs), VECTOR_EMBEDDINGS_DIM)

        self.E_pref = nn.Embedding(len(self.m_pref), VECTOR_EMBEDDINGS_DIM)




    ###############################################################
    # Function Name:forward
    # Function input:x
    # Function output:none
    # Function Action:This is the forward propagation function
    # #feed the neural networks firts time, and use activition function
    # #to convert values to be between 0 to 1
    ################################################################
    def forward(self, x):


        suffix_of_wind = x.data.numpy().copy()

        prefix_of_wind = x.data.numpy().copy()

        prefix_of_wind = prefix_of_wind.reshape(-1)

        suffix_of_wind = suffix_of_wind.reshape(-1)

        prefix_of_wind = [self.m_pref[self.prefix_to_index[helper.Representation_Of_Indexes_By_Words[index][:3]]]
                        for index in prefix_of_wind]

        suffix_of_wind = [self.m_suffs[self.suffix_to_index[helper.Representation_Of_Indexes_By_Words[index][:-3]]]
                        for index in suffix_of_wind]
        # get lists of the indices for the prefixes/suffixes
        prefix_of_wind = [self.prefix_to_index[pref] for pref in prefix_of_wind]
        suffix_of_wind = [self.suffix_to_index[suff] for suff in suffix_of_wind]

        #convert to np array
        prefix_of_wind = np.asanyarray(prefix_of_wind)
        suffix_of_wind = np.asanyarray(suffix_of_wind)
        #reshape
        #print(x.data.dtype)
        #print(x.data.shape)
        #print(prefix_of_wind.shape)
        #print(np.unique(x.data))
        prefix_of_wind = torch.from_numpy(prefix_of_wind.reshape(x.data.shape)).type(torch.LongTensor)
        suffix_of_wind = torch.from_numpy(suffix_of_wind.reshape(x.data.shape)).type(torch.LongTensor)
        #prefix_vectors, suffix_vectors = prefix_vectors.type(tr.LongTensor), suffix_vectors.type(tr.LongTensor)

        x = (self.E(x) + self.E_pref(prefix_of_wind) + self.E_suff(suffix_of_wind)).view(-1, self.input_size)
        # use the tanh as the activation function
        x = F.tanh(self.fc0(x))
        x = self.fc1(x)
        # call softmax function
        x_softmax = F.log_softmax(x, dim=1)
        return x_softmax


###############################################################
# Function Name:create_match_ploting
# Function input:m_dev_loss, m_precents_correct_dev
# Function output:none
# Function Action: the function uses plot library to plot graph shes the learning and scores of the model#
################################################################

def create_match_ploting(m_dev_loss, m_precents_correct_dev):


    line1, = plt.plot(m_dev_loss.keys(), m_dev_loss.values(), "red",
                      label='Dev avg loss')

    # add a title to the plots
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

    # show the plot
    plt.show()

    # add a title to the plots
    line2, = plt.plot(m_precents_correct_dev.keys(), m_precents_correct_dev.values(),
                      label='Dev avg accuracy')
    # drawing name of the graphs
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=4)})
    plt.show()



###############################################################
# Function Name:load_training_data_set
# Function input:train_set_file
# Function output:the training data set
# Function Action:get the train data
#
################################################################

def load_training_data_set(train_set_file):
    windows_array, tags = helper.get_train_data(train_set_file)
    # Convert a list into a numpy array and set data-type to float32
    windows_array = np.asarray(windows_array, np.float32)

    # Convert a list into a numpy array and set data-type to int32
    tags = np.asarray(tags, np.int32)

    """
    Creates a Tensor from a numpy.ndarray
    The returned tensor and ndarray share the same memory. Modifications to the tensor will
     be reflected in the ndarray and vice versa. The returned tensor is not resizable.
    """
    tags = torch.from_numpy(tags)
    windows_array = torch.from_numpy(windows_array)

    # make sure the words and tag have same size, in order to pass it through TensorDataset
    tags = tags.type(torch.LongTensor)
    windows_array = windows_array.type(torch.LongTensor)

    data = TensorDataset(windows_array, tags)

    """
    shuffle is set to True, we will get a new order of exploration at each pass.
    Shuffling the order in which examples are fed to the classifier is helpful so
     that batches between epochs do not look alike. Doing so will eventually make our model more robust.
    """
    return DataLoader(batch_size=BATCH_NORMALIZATION_SIZE, shuffle=True, dataset=data)


###############################################################
# Function Name:load_dev_data_set
# Function input:dev_file
# Function output:dev data set
# Function Action: loading the dev data set
#
################################################################

def load_dev_data_set(dev_file):

    windows_array, tags = helper.get_dev_data(dev_file)
    # Convert a list into a numpy array and set data-type to float32
    windows_array = np.asarray(windows_array, np.float32)

    # Convert a list into a numpy array and set data-type to int32
    tags = np.asarray(tags, np.int32)

    """
    Creates a Tensor from a numpy.ndarray
    The returned tensor and ndarray share the same memory. Modifications to the tensor will
     be reflected in the ndarray and vice versa. The returned tensor is not resizable.
    """
    tags = torch.from_numpy(tags)
    windows_array = torch.from_numpy(windows_array)

    # make sure the words and tag have same size, in order to pass it through TensorDataset
    tags = tags.type(torch.LongTensor)
    windows_array = windows_array.type(torch.LongTensor)

    data = TensorDataset(windows_array, tags)

    """
    shuffle is set to True, we will get a new order of exploration at each pass.
    Shuffling the order in which examples are fed to the classifier is helpful so
     that batches between epochs do not look alike. Doing so will eventually make our model more robust.
    """
    return DataLoader(batch_size=1, shuffle=True, dataset=data)

#we want to define to model which help function file to use? of part 1 or part 2
use_help_2 = bool(int(sys.argv[2]))
if use_help_2:
    import helper_funcs2 as helper
else:
    import help_funcs as helper



###############################################################
# Function Name:define_path_for_txt
# Function input:folder_name
# Function output:paths for all datasets we need to load
# Function Action:the function load all the data sets wee need for the assignment
#create a moel of our neural network and start tarining action
################################################################
def define_path_for_txt(folder_name):
    path_test = folder_name + '/test'
    path_train = folder_name + '/train'
    path_dev = folder_name + '/dev'

    return path_test,path_train,path_dev


###############################################################
# Function Name:define_learning_rate
# Function input:folder_name
# Function output:none
# Function Action: update the learning rate parameter according to the folder name
################################################################
def define_learning_rate(folder_name):
    global learning_rate
    if 'ner' == folder_name:
        learning_rate = 0.01

###############################################################
# Function Name:main
# Function input:argv
# Function output:none
# Function Action:the function load all the data sets wee need for the assignment
#create a moel of our neural network and start tarining action
################################################################

def main(argv):
    # ner or pos (user input)
    folder_name_input = argv[0]
    global learning_rate



    define_learning_rate(folder_name_input)

    # define a path for each dataset file
    path_test, path_train, path_dev = define_path_for_txt(folder_name_input)

    set_of_training = load_training_data_set(path_train)
    print(len(helper.Dictionary_of_words))

    set_of_dev = load_dev_data_set(path_dev)

    dataset_test = load_test_set(path_test)


    # create a nueral net work object
    my_neural_network_model = NeuralNet(use_help_2, input_size=DIM_INPUT)
    optimizer = optim.RMSprop(my_neural_network_model.parameters(), lr=learning_rate)

    # create a training object
    training_object = Part1Model(optimizer, set_of_training, my_neural_network_model, dataset_test, set_of_dev)
    # call start action object to start training the model
    training_object.Start_Action(folder_name_input)


###############################################################
# Function Name: get_y_tag
# Function input: model
# Function output:return the model prediction tag
# Function Action: the function return the prediction
# by getting  the index of the max log-probability
################################################################

def get_y_tag(model):
    return model.data.max(1, keepdim=True)[1]


###############################################################
# Function Name:print_message_each_epoch
# Function input:kind_of_set,length of set, loss of model, number
# of correct predictions of model and size of batch
# Function output:none
# Function Action:the function print a message to help the user
# follow the network progress
# Function Action:the function calculate the loss of the model
# for each epochs,and print write message
################################################################

def print_message_each_epoch(is_training_set, m_total, epoch, m_loss, m_success, size_of_batch, accuracy):
    if (is_training_set == 1):
        title = "Training Set"
    else:
        title = "Validation set"

    print('\n' + title + ': Epoch number is:{}  The average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
                                                                                                              m_loss,
                                                                                                              m_success,
                                                                                                              m_total,
                                                                                                              accuracy))


if __name__ == "__main__":
    main(sys.argv[1:])