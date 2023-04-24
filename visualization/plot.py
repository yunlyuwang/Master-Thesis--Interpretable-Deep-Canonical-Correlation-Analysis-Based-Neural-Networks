import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import io
import itertools

def plot_maps(img1, img2, mix_val=2):
    f = plt.figure(figsize=(15,45))
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(img1*mix_val+img2/mix_val)
    plt.axis("off")

def plot_maps02(img1, img2, img3):
    f = plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(img3)
    plt.axis("off")

def compare_plot_mapes(img1, img2):
    
    f = plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.axis("off")

    return f

def nor_img(x):
    x = tf.maximum(x, 0)#only positive neccesary
    max = tf.reduce_max(x)
    min = tf.reduce_min(x)
    return (x- min)/(max - min)

def nor_img01 (x):
    return tf.maximum(x, 0)/ tf.reduce_max(x)

def nor_img02 (x):  #normalized
    max = tf.reduce_max(x)
    min = tf.reduce_min(x)
    return (x- min)/(max - min)

def nor_img03(x):
    relu_x = tf.maximum(x, 0) 
    return nor_img02(relu_x)


def t_SNE_image(data_x, y):
    tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300)
    tsne_test= tsne.fit_transform(data_x)
    
    figure = plt.figure(figsize=(7,5))
    plt.scatter(tsne_test[:, 0], tsne_test[:, 1], s= 5, c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('Visualizing MNIST through t-SNE', fontsize=16)
    return figure

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def t_SNE_2images(ori_x, proc_x, y):
    tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300)

    tsne_ori= tsne.fit_transform(ori_x)
    tsne_proc= tsne.fit_transform(proc_x)

    
    figure = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(tsne_ori[:, 0], tsne_ori[:, 1], s= 5, c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('Visualizing original MNIST through t-SNE', fontsize=16)

    plt.subplot(1,2,2)
    plt.scatter(tsne_proc[:, 0], tsne_proc[:, 1], s= 5, c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('Visualizing processing MNIST through t-SNE', fontsize=16)

    return figure


def t_SNE_2images_value(ori_x, proc_x, y):
    tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300)

    tsne_ori= tsne.fit_transform(ori_x)
    tsne_proc= tsne.fit_transform(proc_x)

    return tsne_ori, tsne_proc

def tensor_image_summmary(img3,img4,img5,img6,img7,box_att_img):
  
    #img3-ori example mnist
    #img4-saliency map mnist
    #img5-nosie ori mnist
    #img6-smooth map mnist
    #img7-gradcam map
    #box attension
    figure = plt.figure(figsize=(8,12))
    plt.subplot(3,2,1)
    plt.imshow(img3)
    plt.axis("off")
    plt.title('Original MNIST Example', fontsize=5)

    plt.subplot(3,2,2)
    plt.imshow(img4)
    plt.axis("off")
    plt.title('Saliency Map', fontsize=5)

    plt.subplot(3,2,3)
    plt.imshow(img5)
    plt.axis("off")
    plt.title('Noise MNIST Example', fontsize=5)

    plt.subplot(3,2,4)
    plt.imshow(img6)
    plt.axis("off")
    plt.title('Smooth Saliency Map', fontsize=5)

    plt.subplot(3,2,5)
    plt.imshow(img3)
    plt.axis("off")
    plt.title('Original MNIST Example', fontsize=5)

    plt.subplot(3,2,6)
    plt.imshow(img7)
    plt.axis("off")
    plt.title('GradCam Map with box attention{}'. format(float(box_att_img)), fontsize=8)

    return figure

def tensor_image_summmary_dnn(img3,img4,img5,img6):
    figure = plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plt.imshow(img3)
    plt.axis("off")
    plt.title('Original MNIST Example', fontsize=5)

    plt.subplot(2,2,2)
    plt.imshow(img4)
    plt.axis("off")
    plt.title('Saliency Map', fontsize=5)

    plt.subplot(2,2,3)
    plt.imshow(img5)
    plt.axis("off")
    plt.title('Noise MNIST Example', fontsize=5)

    plt.subplot(2,2,4)
    plt.imshow(img6)
    plt.axis("off")
    plt.title('Smooth Saliency Map', fontsize=5)

    return figure

def show2images(img1,img2):
    figure = plt.figure(figsize=(4,4))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.axis("off")
    plt.title('Original Image', fontsize=5)

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.axis("off")
    plt.title('GradCam Map', fontsize=5)
    
    return figure