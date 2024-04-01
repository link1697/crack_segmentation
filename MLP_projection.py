import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def contrastive_loss(projections, temperature=0.1):
    batch_size = tf.shape(projections)[0]
    labels = tf.range(batch_size)

    projections = tf.math.l2_normalize(projections, axis=1)
    logits = tf.matmul(projections, projections, transpose_b=True)
    logits /= temperature
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=labels, y_pred=logits, from_logits=True
    )

    mask = tf.eye(batch_size)
    loss = tf.boolean_mask(loss, ~tf.cast(mask, tf.bool))
    loss = tf.reduce_mean(loss)

    return loss

input_shape = (112, 112, 3)
unet_model = build_unet(input_shape)
feature_shape = (7, 7)   
mlp_head = build_mlp_head(feature_shape, projection_dim=128)
optimizer = optimizers.Adam()


for epoch in range(num_epochs):
    for step, (images, _) in enumerate(train_dataset):  
        with tf.GradientTape() as tape:
            features = unet_model(images, training=True)
        
            projections = mlp_head(features, training=True)
            loss = contrastive_loss(projections)

        gradients = tape.gradient(loss, unet_model.trainable_variables + mlp_head.trainable_variables)
        optimizer.apply_gradients(zip(gradients, unet_model.trainable_variables + mlp_head.trainable_variables))
        
        if step % log_every_n_steps == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.numpy()}")
