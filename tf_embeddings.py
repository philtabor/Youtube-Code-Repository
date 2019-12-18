import io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

#embedding_layer = layers.Embedding(1000, 5)

#result = embedding_layer(tf.constant([1,2,3]))

#print(result.numpy())
#print(result.numpy().shape)
def get_batch_data():
    (train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',
                                    split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                    with_info=True, as_supervised=True)

    encoder = info.features['text'].encoder
    #print(encoder.subwords[:20])
    padded_shapes = ([None], ())
    train_batches = train_data.shuffle(1000).padded_batch(10,
                                                padded_shapes=padded_shapes)
    test_batches = test_data.shuffle(1000).padded_batch(10,
                                                padded_shapes=padded_shapes)
    return train_batches, test_batches, encoder

def get_model(encoder, embedding_dim=16):

    model = keras.Sequential([
                    layers.Embedding(encoder.vocab_size, embedding_dim),
                    layers.GlobalAveragePooling1D(),
                    layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,9))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim((0.5, 1))
    plt.show()

def retrieve_embeddings(model, encoder):
    out_vectors = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_metadata = io.open('meta.tsv', 'w', encoding='utf-8')
    weights = model.layers[0].get_weights()[0]

    for num, word in enumerate(encoder.subwords):
        vec = weights[num+1]
        out_metadata.write(word + '\n')
        out_vectors.write('\t'.join([str(x) for x in vec]) + '\n')
    out_vectors.close()
    out_metadata.close()

train_batches, test_batches, encoder = get_batch_data()
model = get_model(encoder)
history = model.fit(train_batches, epochs=10, validation_data=test_batches,
                    validation_steps=20)
plot_history(history)
retrieve_embeddings(model, encoder)
