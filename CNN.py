import numpy as np
import struct
import tensorflow as tf

def load_data():
    with open('train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open('train-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        train_images = np.fromfile(imgs, dtype=np.uint8).reshape(num,784)
    with open('t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open('t10k-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        test_images = np.fromfile(imgs, np.uint8).reshape(num,784)
    return train_images, train_labels, test_images, test_labels

def cnn_model_fn(features, labels, mode):
    input_layer = tf.cast(tf.reshape(features['x'], [-1, 28, 28, 1]), tf.float16)

    conv1 = tf.layers.conv2d(inputs=input_layer,
                            filters=16,
                            kernel_size=[5,5],
                            padding='same',
                            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1,
                            filters=32,
                            kernel_size=[5,5],
                            padding='same',
                            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size = [2,2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=10)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')        
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = { 'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
if __name__ == '__main__':
    training_data, training_labels, testing_data, testing_labels = load_data()
    num_epochs = 10

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                        model_dir='tmp/')

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_data},
        y=training_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)
    
    for i in range(num_epochs):
        classifier.train(input_fn=input_fn, steps=1000)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': testing_data},
        y=testing_labels,
        shuffle=False)
    
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print('these are the results of my evaluations')
    print(eval_results)

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': testing_data},
        y=testing_labels,
        num_epochs=1,
        shuffle=False)
    
    pred_results = classifier.predict(input_fn=pred_input_fn)
    predicted_classes = [p['classes'] for p in pred_results]      