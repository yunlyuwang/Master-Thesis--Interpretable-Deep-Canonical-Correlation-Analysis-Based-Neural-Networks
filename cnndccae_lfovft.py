import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from keras.utils.data_utils import get_file
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
import io
import itertools

################## define the model
from dccamodel.losses import compute_recons_mse, compute_loss, CCA, compute_aver_corr_out_data
from dccamodel.dcca_model import Encoder, Decoder, TwoEncoders
from cnnmodel.cnn_model_test import CnnEncoder, TwoCnnEncoders
from visualization.plot import plot_maps, nor_img,t_SNE_image,plot_to_image,compare_plot_mapes, t_SNE_2images, plot_maps02
from visualization.visualways import calculateGradients, get_pooled_grads_last_conv_out, calculate_heatmap, calculate_heatmap02, box_attention, Guided_Backpropagation
from NNclassifier import Fullconnection
from visualization.summary import image_summary_ori

for repeat in [1]:
    # Create the neural networks
    model = TwoCnnEncoders() #只编写最后输出层的大小，其余cnn运算已内置写好
    #decode
    recon_model = Decoder(decoder_config=[(1024,'relu'),(1024,'relu'),(1024,'relu'),(784,None)])
    #define the inner classifer for saliency map
    FL = Fullconnection(18)
    FL.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam()

    #Define the parameters for tensorboard
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    performance_log_dir = 'logs/DNNvsCNN/CNN'+ current_time 
    performance_summary_writer = tf.summary.create_file_writer(performance_log_dir) #corr, loss, accuracy

    ##################box-noise data generate
    from dataset.mnist_corr import BoxNoiseData, shuffle_instances, box_augment, augment_batch, filter_classes, preprocess_dataset, generate_noisy_images
    num_classes=10
    view1_train, view1_eval, view1_test, view2_train, view2_eval, view2_test = BoxNoiseData.generate(batch_size = 10, num_boxes = 1, data_idx = 0)  #first try to use 1 boxes

    print("view1_train[0]'s shape is {}".format(view1_train[0].shape)) #(50000, 784)。view_里面放了1个元组，view1_test[0]对应的是图片，view1_test[1]对应的是labels
    print("view1_eval[0]'s shape is {}".format(view1_eval[0].shape))   #(10000, 784)
    print("view1_test[0]'s shape is {}".format(view1_test[0].shape))  #(10000, 784)

    ##################package data for putting into model
    #train model
    view01_train = tf.convert_to_tensor(view1_train[0])
    view02_train = tf.convert_to_tensor(view2_train[0])

    #using for fit classifier
    view01_eval01 = tf.convert_to_tensor(view1_eval[0][:800]) #if using a few classes, take care the nurmbers
    view02_eval01 = tf.convert_to_tensor(view2_eval[0][:800])
    #fit svm by view1
    Y_view01 = view1_eval[1][:800]
    view01_box = view1_eval[2][:800]       #box
    training_data_e01 = dict(nn_input_0 = view01_eval01, nn_input_1 = view02_eval01)
    #noise data
    view01_eval01_noise = generate_noisy_images(view1_eval[0][:600], 0.2)
    view01_eval01_noise = tf.convert_to_tensor(view01_eval01_noise)
    view02_eval01_noise = generate_noisy_images(view2_eval[0][:600], 0.2)
    view02_eval01_noise = tf.convert_to_tensor(view02_eval01_noise)
    training_data_e01_noise = dict(nn_input_0 = view01_eval01_noise, nn_input_1 = view02_eval01_noise)
    #evaluate classifier
    view01_eval02 = tf.convert_to_tensor(view1_eval[0][600:1200])
    view02_eval02 = tf.convert_to_tensor(view2_eval[0][600:1200])
    #evaluate by view1
    eval_Y_view01 = view1_eval[1][600:1200]
    eval_view01_box = view1_eval[2][600:1200]       #box
    training_data_e02 = dict(nn_input_0 = view01_eval02, nn_input_1 = view02_eval02)

    ###############training and evaluating the model
    print("training and evaluating model is runing now")

    for epoch in range(1001):
        # Train one epoch
        with tf.GradientTape() as tape:
            # Feed forward
            training_data_t = dict(nn_input_0= view01_train, nn_input_1= view02_train)  # Insert data here, makes sure its of tensorflow data type
            network_output = model(training_data_t)
            # decode
            recons = recon_model(network_output)
            # Compute loss
            loss, corr = compute_loss(network_output, recons, training_data_t)
            #check the correlation between network_output and orginal data, for finding when it startes overfitting
            aver_corr_out_data = compute_aver_corr_out_data(training_data_t, network_output) 

        with performance_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch)
            tf.summary.scalar('corr', tf.reduce_mean(corr), step=epoch)
                
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch% 50 == 0: 
            network_output_eval01 = model(training_data_e01)
            X = network_output_eval01['latent_view_0']
            #FL.fit(X, Y_view01, epochs=10)

            # classif  = svm.LinearSVC(C=0.01, dual=False)
            classif = OneVsRestClassifier(SVC(kernel="linear"))
            classif.fit(X, Y_view01)

            network_output_eval02 = model(training_data_e02)
            eval_X = network_output_eval02['latent_view_0']
            # FL_accu = FL.evaluate(eval_X, eval_Y_view01)
            # print("FL_{}'s accuracy is {}". format(int(epoch), format(float(FL_accu[1]))))
            # with performance_summary_writer.as_default():
            #     tf.summary.scalar('flaccu', FL_accu[1], step = epoch)
            
            p = classif.predict(eval_X)
            cnn_accu = accuracy_score(eval_Y_view01, p) #classif.score(eval_X, eval_Y_view01) 
            print("svm_{}'s accuracy is {}". format(int(epoch), format(float(cnn_accu))))

            with performance_summary_writer.as_default():
                tf.summary.scalar('svmaccu', cnn_accu, step=epoch)
            #——————————————————————————————————————————————————————
            #cluster
            # clust_labels = SpectralClustering(
            #                 n_clusters=num_classes,
            #                 assign_labels='kmeans',
            #                 affinity='nearest_neighbors',
            #                 random_state=33,
            #                 n_init=10).fit_predict(X)  #利用X预测有多少个cluster

            # prediction = np.zeros_like(clust_labels)
            # for i in range(num_classes):
            #     ids = np.where(clust_labels == i)[0]
            #     prediction[ids] = np.argmax(np.bincount(Y_view01[ids])) #利用labels的量去判断

            # cluster_acc = accuracy_score(Y_view01, prediction)
            # with performance_summary_writer.as_default():
            #     tf.summary.scalar('cluster', cluster_acc, step = epoch)
            
    print("{} time training and evaluating model is done". format(int(repeat)))
