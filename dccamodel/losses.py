import tensorflow as tf
#import tensorflow_probability as tfp

def compute_recons_mse(recons, training_data):
    y_true_0 = tf.cast(training_data['nn_input_0'], dtype=tf.float32)
    y_pred_0 = tf.cast(recons['latent_out_view_0'], dtype=tf.float32)
    recons_mse_0 = tf.reduce_sum(tf.square(y_true_0 - y_pred_0)) 
    #recons_mse_0 = tf.keras.losses.MeanSquaredError(y_true_0, y_pred_0)

    y_true_1 = tf.cast(training_data['nn_input_1'], dtype=tf.float32)
    y_pred_1 = tf.cast(recons['latent_out_view_1'], dtype=tf.float32)
    recons_mse_1 = tf.reduce_sum(tf.square(y_true_1 - y_pred_1))
    #recons_mse_1 = tf.keras.losses.MeanSquaredError(y_true_1, y_pred_1)

    N = training_data['nn_input_0'].shape[0]

    recons_mse = (recons_mse_0 + recons_mse_1) / N

    return recons_mse
    

def compute_loss(network_output,recons, data, lambda_1 = 1e-5):  
        # Compute CCA loss
        # we expect the latent space with 5 dimensions here
        B0, B1, epsilon, omega, D = CCA(
            network_output['latent_view_0'],
            network_output['latent_view_1'],
            num_shared_dim=10
        )
        recons_mse = compute_recons_mse(recons, data)
        cca_loss = -1*tf.reduce_sum(D)/10 + lambda_1 * recons_mse 
        
        return cca_loss, D

def CCA(view1, view2, num_shared_dim, r1=1e-4, r2=1e-4):      #r1 r2 are for avoiding the small size of dimensions. also always set small.
    # Compute CCA with tensorflow variables to allow for automatic back-prop
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)

    assert V1.shape[0] == V2.shape[0]
    M = tf.constant(V1.shape[0], dtype=tf.float32)
    ddim_1 = tf.constant(V1.shape[1], dtype=tf.int16)
    ddim_2 = tf.constant(V2.shape[1], dtype=tf.int16)
    
    # check mean and variance
    mean_V1 = tf.reduce_mean(V1, 0)  #take every rows sums
    mean_V2 = tf.reduce_mean(V2, 0)

    V1_bar = tf.subtract(V1, tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
    V2_bar = tf.subtract(V2, tf.tile(tf.convert_to_tensor(mean_V2)[None], [M, 1]))

    Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
    Sigma11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) + r1 * tf.eye(ddim_1)
    Sigma22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) + r2 * tf.eye(ddim_2)

    Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
    Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
    Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

    C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
    D, U, V = tf.linalg.svd(C, full_matrices=True)

    #corr = tf.sqrt(tf.linalg.trace(tf.matmul(C, C, transpose_a=True)))  #本质是一样的

    A = tf.matmul(tf.transpose(U)[:num_shared_dim], Sigma11_root_inv)
    B = tf.matmul(tf.transpose(V)[:num_shared_dim], Sigma22_root_inv)

    epsilon = tf.matmul(A, tf.transpose(V1_bar))
    omega = tf.matmul(B, tf.transpose(V2_bar))

    return A, B, epsilon, omega, D


def compute_aver_corr_out_data(training_data, network_output):
    ori_view_0 = training_data['nn_input_0']            #50,784
    ori_view_1 = training_data['nn_input_1']            #50,784 
    net_out_view_0 = network_output['latent_view_0']    #50,20
    net_out_view_1 = network_output['latent_view_1']    #50,20

    #tensorflow variables
    ori_view0 = tf.cast(ori_view_0, dtype=tf.float32)
    ori_view1 = tf.cast(ori_view_1, dtype=tf.float32)
    netout_view0 = tf.cast(net_out_view_0, dtype=tf.float32)
    netout_view1 = tf.cast(net_out_view_1, dtype=tf.float32)

    num_samples = tf.constant(net_out_view_0.shape[0], dtype=tf.float32) #50

    #S_x前两列和N_x的前两列求，得到R_nxsx
    #view2同理

    #R_ori_out = 1/N * out * ori'
    R_ori_out_0 = tf.linalg.matmul(tf.transpose(netout_view0[:2]), ori_view0[:2]) /num_samples
    R_ori_out_1 = tf.linalg.matmul(tf.transpose(netout_view1[:2]), ori_view1[:2]) /num_samples

    #！！！该参数实际上可以直接将view_out和原始信号进行CCA处理，得到的corr，再去平均就可以得到我们检测数据。
    # A_R_0, B_R_0, epsilon_0, omega_0, R_ori_out_0 = CCA(tf.transpose(netout_view0[:2]), ori_view0[:2], num_shared_dim = num_samples)
    # A_R_1, B_R_1, epsilon_1, omega_1, R_ori_out_1 = CCA(tf.transpose(netout_view1[:2]), ori_view1[:2], num_shared_dim = num_samples)
    # #average
    R_ori_out = tf.reduce_mean(R_ori_out_0 + R_ori_out_1)

    return R_ori_out