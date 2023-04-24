import tensorflow as tf

from dcca_model import TwoEncoders
from losses import compute_loss

# Create the neural networks
model = TwoEncoders(
    encoder_config=[(256,'relu'),(256,'relu'),(5,None)]
)

# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# Load training data once
training_data = dict(nn_input_0=..., nn_input_1=...) # Insert data here, makes sure its of tensorflow data type

# Iterate over 1000 epochs
for epoch in range(1000):
    # Train one epoch
    with tf.GradientTape() as tape:
        # Feed forward
        network_output = model(training_data)
        # Compute loss
        loss = compute_loss(network_output, training_data)

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

