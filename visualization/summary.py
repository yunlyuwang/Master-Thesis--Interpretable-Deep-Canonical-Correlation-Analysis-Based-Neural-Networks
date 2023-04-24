import numpy as np
import tensorflow as tf
from visualization.plot import plot_maps, nor_img, plot_to_image,compare_plot_mapes
from visualization.visualways import calculateGradients, get_pooled_grads_last_conv_out, calculate_heatmap

def image_summary_ori(training_data_e01, img_idx):
    input_img_list = []
    for i in range(len(img_idx)):
        input_img = tf.reshape(training_data_e01['nn_input_0'][img_idx[i]],[28,28])
        input_img_list.append(input_img)
    return input_img_list

def find_box_ori_img(ori, box_imgs, img_idx):
    ori_img_list = []
    box_list = []
    for i in range(len(img_idx)):
        ori_img = tf.reshape(ori[img_idx[i]],[28,28])
        ori_img_list.append(ori_img)
        
        box = tf.reshape(box_imgs[img_idx[i]],[28,28])
        box_list.append(box)
    return ori_img_list, box_list