from tensorflow.keras import layers, models

# Residual block
def residual_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Add the input_tensor (residual connection) to the output of the convolution block
    x = layers.add([x, input_tensor])
    x = layers.ReLU()(x)

    return x

# Attention gate
def attention_gate(input_tensor, gate_tensor, num_filters):
    # Reshape gate tensor to match the input_tensor shape
    gate_resized = layers.Conv2D(num_filters, 1, padding='same')(gate_tensor)
    gate_resized = layers.BatchNormalization()(gate_resized)

    # Add the gate to the input_tensor
    x = layers.add([input_tensor, gate_resized])
    x = layers.Activation('relu')(x)

    # Sigmoid activation to get the attention coefficients
    attention = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    x = layers.multiply([input_tensor, attention])

    return x

def conv_block(input_tensor, num_filters):
    x = residual_block(input_tensor, num_filters)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

# Modify the decoder_block to include an attention gate
def decoder_block(input_tensor, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)

    # Apply attention gate to the skip features before concatenating
    attention_skipped = attention_gate(skip_features, x, num_filters)
    
    x = layers.concatenate([x, attention_skipped])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = models.Model(inputs, outputs)
    return model
 