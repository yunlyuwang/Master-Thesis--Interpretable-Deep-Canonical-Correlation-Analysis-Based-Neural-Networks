import numpy as np
import tensorflow as tf

#SM
def calculateGradients(input_multi, model01, model02, viewindex):
    with tf.GradientTape() as tape:
        tape.watch(input_multi)
        output_multi = model01(input_multi)
        if viewindex == 1:
            fl_y = model02(output_multi['latent_view_0'])
            fl_y_max = tf.reduce_max(fl_y, axis=1)
            grads_multi = tape.gradient(fl_y_max, input_multi)
        else:
            fl_y = model02(output_multi['latent_view_1'])
            fl_y_max = tf.reduce_max(fl_y, axis=1)
            grads_multi = tape.gradient(fl_y_max, input_multi)

    return grads_multi
    
#SmoothSM
def generate_noisy_images(images, num_samples, noise):
    repeated_images = np.repeat(images, num_samples, axis=0)  
    noise = np.random.normal(0, noise, repeated_images.shape).astype(np.float32)

    return repeated_images + noise

def calculateSmoothGradients(input_multi, model01, model02, viewindex):
    noise_inputs_list = []
    smooth_noi_map_list = []
    for i in range(4):
        noise_inputs = generate_noisy_images(input_multi, 1, 0.4)
        noise_inputs_list.append(noise_inputs)

        with tf.GradientTape() as tape:
            tape.watch(noise_inputs)
            output_multi = model01(noise_inputs)
            if viewindex == 1:
                fl_y = model02(output_multi['latent_view_0'])
                fl_y_max = tf.reduce_max(fl_y, axis=1)
                grads_multi = tape.gradient(fl_y_max, noise_inputs)
            else:
                fl_y = model02(output_multi['latent_view_1'])
                fl_y_max = tf.reduce_max(fl_y, axis=1)
                grads_multi = tape.gradient(fl_y_max, noise_inputs)

            smooth_noi_map_list.append(grads_multi)
            smooth_average = tf.keras.layers.Average()(smooth_noi_map_list)

    return smooth_average

#Grad-CAM
def get_pooled_grads_last_conv_out(model,input_multi, viewindex):
    model.cnnencoder_v0.model().summary()
    map_model = tf.keras.Model(
        [model.cnnencoder_v0.layers[0].input],
        [model.cnnencoder_v0.layers[0].output,
        model.cnnencoder_v0.layers[1].output, 
        model.cnnencoder_v0.layers[-1].output]
        )
    map_model.summary() #output:[first conv out, second conv out, real out]

    if viewindex == 1:
        map_input = tf.cast(tf.reshape(input_multi['nn_input_0'],[-1, 28, 28, 1]), dtype=tf.float32)
    else:
        map_input = tf.cast(tf.reshape(input_multi['nn_input_1'],[-1, 28, 28, 1]), dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(map_input)
        map_output = map_model(map_input) #[first conv out, second conv out, output]
        grads = tape.gradient(map_output[2], map_output[1])#[5000, 14, 14, 8]
        pooled_grads = tf.reduce_mean(grads, axis=(1,2)) #[5000, 8]

    return grads, pooled_grads, map_output


def calculate_heatmap(pooled_grads, last_conv_output, img_idx):
    a = tf.multiply(pooled_grads[img_idx], last_conv_output[img_idx])
    heatmap = tf.reduce_mean(a, axis=-1)

    heatmap = np.maximum(heatmap, 0) #normalize the img
    heatmap /= np.max(heatmap)

    return heatmap

#try to calculato all the heatmap, before only plot the examples.
def calculate_heatmap02(pooled_grads, last_conv_output):
    heatmap = np.zeros_like(last_conv_output)
    for i in range(pooled_grads.shape[0]) :
        heatmap[i] = tf.multiply(pooled_grads[i], last_conv_output[i]) #(8) * (14,14,8)=(14,14,8)
    heatmap = tf.reduce_mean(heatmap, axis=-1)    #(5k,14,14)
    heatmap = tf.math.maximum(heatmap, 0) #only positive values matter

    return heatmap #heatmap(5k,14,14) 

def box_attention(box, heatmap): 
    heatmap = tf.reshape(heatmap, [-1,14,14,1]) #([5000, 14, 14,1])
    heatmap = tf.image.resize(heatmap,[28,28]) #([5000, 28, 28, 1])
    heatmap = tf.reshape(heatmap, [-1,784]) #([5000, 784]) box(5k,784)
    box_attention = tf.math.multiply(box, heatmap)#(5000,784)
    #normaize the box attention
    nor_box_attention = []
    for i in range(box_attention.shape[0]): #每张图片做nor处理
        box_max = tf.reduce_max(box_attention[i])
        box_min = tf.reduce_min(box_attention[i])
        nor = tf.math.divide_no_nan((box_attention[i]-box_min),(box_max - box_min))
        nor_box_attention.append(nor)
    #average all images
    nor_average = tf.reduce_mean(nor_box_attention)
    return nor_box_attention, nor_average, box_attention


#Guided Backpropagation
class Guided_Backpropagation:
    def __init__(self, model, input_multi, viewindex):
        self.model = model
        self.input_multi = input_multi
        self.viewindex = viewindex

    @tf.custom_gradient
    def guidedRelu(a):
        def grad(dy):
            return tf.cast(dy>0,"float32") * tf.cast(a>0, "float32") * dy
        return tf.nn.relu(a), grad 

    def GBP_grads(self):
        bpg_model = tf.keras.Model(
            self.model.cnnencoder_v0.layers[0].input,
            self.model.cnnencoder_v0.layers[-1].output
        )

        layer_dict = [layer for layer in bpg_model.layers if hasattr(layer,'activation')]

        for l in layer_dict: #更换了conv的激活层
            if l.activation == tf.keras.activations.relu:
                l.activation = guidedRelu

        if self.viewindex == 1:
            map_input = tf.cast(tf.reshape(self.input_multi['nn_input_0'],[-1, 28, 28, 1]), dtype=tf.float32)
        else:
            map_input = tf.cast(tf.reshape(self.input_multi['nn_input_1'],[-1, 28, 28, 1]), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(map_input)
            map_output = bpg_model(map_input)
        GBP_grads = tape.gradient(map_output,map_input)

        return GBP_grads


