import keras
import tensorflow as tf # Or jax, or torch depending on your preferred backend

# Define your model using Keras 3 API
inputs = keras.Input(shape=(64,), name="input_2")
x = keras.layers.Reshape((4, 16), name="reshape_2")(inputs)
x = keras.layers.Permute((2, 1), name="permute_2")(x)

# Block 1 (Conv1D -> BN -> ReLU)
conv1d_3 = keras.layers.Conv1D(
    32, 1, padding='same',
    kernel_regularizer=keras.regularizers.l2(9.999999747378752e-06), # L2 regularizer
    name="conv1d_3"
)(x)
bn1 = keras.layers.BatchNormalization(
    momentum=0.99, epsilon=0.001, center=True, scale=True,
    name="batch_normalization_1"
)(conv1d_3)
act1 = keras.layers.Activation('relu', name="activation_1")(bn1)

# Block 2 (Conv1D -> BN -> ReLU)
conv1d_4 = keras.layers.Conv1D(
    32, 3, padding='same',
    kernel_regularizer=keras.regularizers.l2(9.999999747378752e-06),
    name="conv1d_4"
)(act1)
bn2 = keras.layers.BatchNormalization(
    momentum=0.99, epsilon=0.001, center=True, scale=True,
    name="batch_normalization_2"
)(conv1d_4)
act2 = keras.layers.Activation('relu', name="activation_2")(bn2)

# Block 3 (Conv1D -> BN -> ReLU)
conv1d_5 = keras.layers.Conv1D(
    32, 3, padding='same',
    kernel_regularizer=keras.regularizers.l2(9.999999747378752e-06),
    name="conv1d_5"
)(act2)
bn3 = keras.layers.BatchNormalization(
    momentum=0.99, epsilon=0.001, center=True, scale=True,
    name="batch_normalization_3"
)(conv1d_5)
act3 = keras.layers.Activation('relu', name="activation_3")(bn3)

# Add Layer (Skip connection)
# This combines the output of the first activation block (act1) with the third (act3)
add_layer_output = keras.layers.Add(name="add_1")([act1, act3])

# Dense Blocks
flatten = keras.layers.Flatten(name="flatten_2")(add_layer_output)

dense4 = keras.layers.Dense(
    64, activation='linear', # Activation is 'linear' for Dense layers before BN/ReLU in your config
    kernel_regularizer=keras.regularizers.l2(9.999999747378752e-06),
    name="dense_4"
)(flatten)
bn4 = keras.layers.BatchNormalization(
    momentum=0.99, epsilon=0.001, center=True, scale=True,
    name="batch_normalization_4"
)(dense4)
act4 = keras.layers.Activation('relu', name="activation_4")(bn4)

dense5 = keras.layers.Dense(
    64, activation='linear',
    kernel_regularizer=keras.regularizers.l2(9.999999747378752e-06),
    name="dense_5"
)(act4)
bn5 = keras.layers.BatchNormalization(
    momentum=0.99, epsilon=0.001, center=True, scale=True,
    name="batch_normalization_5"
)(dense5)
act5 = keras.layers.Activation('relu', name="activation_5")(bn5)

outputs = keras.layers.Dense(
    1, activation='sigmoid', # Final output layer activation
    kernel_regularizer=keras.regularizers.l2(9.999999747378752e-06),
    name="dense_6"
)(act5)

# Create the Keras Model
model = keras.Model(inputs=inputs, outputs=outputs, name="model_2_v3_no_weights")

# Compile the model (necessary if you plan to train it)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Model architecture successfully defined in Keras 3:")
model.summary()

keras3_model_path = "my_keras3_model_architecture_only.keras"
model.save(keras3_model_path)
print(f"\nModel architecture saved to {keras3_model_path} in Keras 3 format.")

# You can load this model later in Keras 3:
# loaded_model_v3 = keras.models.load_model(keras3_model_path)
# loaded_model_v3.summary()