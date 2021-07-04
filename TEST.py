import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib import gridspec
import cv2
import os
import time
from preprocessing import PreProcessing
from model import TripletLoss
from PIL import Image


# Read Dataset. Split into training and test set
def get_train_test_dataset(train_test_ratio):
    data_src = r'C:\OData_low'# OData_low TTT_all TTT
    X = []
    y = []
    for directory in os.listdir(data_src):
        try:
            for pic in os.listdir(os.path.join(data_src, directory)):
                img = cv2.imread(os.path.join(data_src, directory, pic))
                X.append(np.squeeze(np.asarray(img)))
                y.append(directory)
        except:
            pass

    labels = list(set(y))
    label_dict = dict(zip(labels, range(len(labels))))
    Y = np.asarray([label_dict[label] for label in y])
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = []
    y_shuffled = []
    for index in shuffle_indices:
        x_shuffled.append(X[index])
        y_shuffled.append(Y[index])

    size_of_dataset = len(x_shuffled)
    n_train = int(np.ceil(size_of_dataset * train_test_ratio))
    n_test = int(np.ceil(size_of_dataset * (1 - train_test_ratio)))
    return np.asarray(x_shuffled[0:n_train]), np.asarray(x_shuffled[n_train + 1:size_of_dataset]), np.asarray(y_shuffled[0:n_train]), np.asarray(y_shuffled[
                                                                                                  n_train + 1:size_of_dataset])

train_images, test_images, train_label, test_label = get_train_test_dataset(0.8)


#helper function to plot image
def show_image(idxs, data):
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs])
    fig = plt.figure()
    gs = gridspec.GridSpec(1,len(idxs))
    for i in range(len(idxs)):
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(data[idxs[i],:,:,:])
        ax.axis('off')
    plt.show()

model_path = r"model_triplet\model.ckpt"
model = TripletLoss()

img_placeholder = tf.placeholder(tf.float32, [None, 128, 128, 3], name='img')
net = model.conv_net(img_placeholder, reuse=False)

# Generate random index from test_images corpus and display the image
idx = np.random.randint(0, len(test_images))
im = test_images[idx]
img = cv2.imread(r'NRR.png')

'''print("********** QUERY IMAGE **********")
show_image(idx, test_images)'''



def generate_db_normed_vectors():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, model_path)#saver.restore(sess, model_path+"model.ckpt")
        train_vectors = sess.run(net, feed_dict={img_placeholder:train_images})
    normalized_train_vectors = train_vectors/np.linalg.norm(train_vectors,axis=1).reshape(-1,1)
    return normalized_train_vectors

def find_k_nn(normalized_train_vectors, vec, k):
    dist_arr = np.matmul(normalized_train_vectors, vec.T)
    return np.argsort(-dist_arr.flatten())[:k]

normalized_training_vectors = generate_db_normed_vectors()

#helper function to plot imageS
def show_top_k_images(indx_list, test_image_indexes, train_data, test_data):
    fig = plt.figure(figsize=(20, 40))
    gs = gridspec.GridSpec(len(indx_list), len(indx_list[0])+2)
    for i in range(len(indx_list)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(test_data[test_image_indexes[i], :, :, :])
        ax.axis('off')
        for j in range(len(indx_list[0])):
            ax = fig.add_subplot(gs[i, j+2])
            ax.imshow(train_data[indx_list[i][j], :, :, :])
            ax.axis('off')
    plt.savefig(r'.\figures\similar_images.jpg')
    plt.show()


K = 100
N = 1
indx_list = []
test_image_indexes = []
_test_images = []
for i in range(N):
    idx = i
    test_image_indexes.append(idx)
    _test_images.append(test_images[idx])
    # run the test image through the network to get the test features
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, model_path)
    search_vectors = sess.run(net, feed_dict={img_placeholder: _test_images})

normalized_search_vecs = search_vectors / np.linalg.norm(search_vectors, axis=1).reshape(-1, 1)
for i in range(len(normalized_search_vecs)):
    candidate_index = find_k_nn(normalized_training_vectors, normalized_search_vecs[i], K)
    indx_list.append(candidate_index)
    print(i, "SAME")
    for j in range(len(candidate_index)):
        if j < 6:
            print(np.linalg.norm(normalized_search_vecs[i] - normalized_training_vectors[j]), ',')

    print(i, "NOT SAME")
    for k in range(len(candidate_index)):
        if k >= 6:
            print(np.linalg.norm(normalized_search_vecs[i] - normalized_training_vectors[k]), ',')
    #min = np.linalg.norm(normalized_search_vecs[i] - normalized_training_vectors[0])
    #for j in range(len(candidate_index)):
    #    if (min > np.linalg.norm(normalized_search_vecs[i] - normalized_training_vectors[j])):
    #        min = np.linalg.norm(normalized_search_vecs[i] - normalized_training_vectors[j])
    #print("DISTANCE ", i, ": ", min)


print('**Query Image**       *************************** Top %d Similar Images  ***************************' % K)

show_top_k_images(indx_list, test_image_indexes, train_images, test_images)