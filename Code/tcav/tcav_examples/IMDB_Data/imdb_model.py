import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def tokenizer(training_sentences,vocab_size):
    #Tokenize
    oov_tok = "<OOV>"
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    return tokenizer

def model_creation(data):
    #Get data ready
    train_data, test_data = data['train'], data['test']
    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []
    for s,l in train_data:
        training_sentences.append(str(s.numpy()))
        training_labels.append(l.numpy())
    
    for s,l in test_data:
        testing_sentences.append(str(s.numpy()))
        testing_labels.append(l.numpy())
    
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    #Setting important variables
    embedding_dim = 16
    max_length = 100
    trunc_type='post'
    vocab_size = 10000

    tknz = tokenizer(training_sentences,vocab_size)
    word_index = tknz.word_index

    #Modify data to fit model
    sequences = tknz.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
    testing_sequences = tknz.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
    #create model with tensorflow
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    num_epochs = 10
    history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

    return model

def testModel(model):
    max_length = 100
    trunc_type='post'
    vocab_size = 10000
    oov_tok = "<OOV>"
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    new_sentences = [
    'I loved this movie.',
    'This film is so boring.',
    'This movie is so hilarious. I had a really great time!',
    'Very linear scenario, no surprises at all',
    'Another amazing addition to the franchise with good story arcs and standalone episodes.',
    'Not for the hardened not even the casual fans.'
    ]
    new_sequences = tokenizer.texts_to_sequences(new_sentences)
    padded=pad_sequences(new_sequences, maxlen=max_length,truncating=trunc_type)
    output=model.predict(padded)
    for i in range(0,len(new_sentences)):
        print('Review:'+new_sentences[i]+' '+'sentiment:'+str(output[i])+'\n')




def create_model():
    dir_path = "/code/tcav/tcav_examples/IMDB_Data/" #If you are using docker no need to change this
    model_name = "imdb_model.h5"
    imdb,info = tfds.load("imdb_reviews",with_info=True,as_supervised=True,data_dir=dir_path,download=True)
    if(os.path.exists(os.path.join(dir_path,model_name))):
        return tf.keras.models.load_model(os.path.join(dir_path,model_name))
    else:
        model = model_creation(imdb)
        #testModel(model)
        model.save(os.path.join(dir_path,model_name))
        return model

