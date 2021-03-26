import random
import time
import webbrowser
import os
import json
import pickle
import string
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import nltk
from nltk.stem import WordNetLemmatizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Optimizer
from tensorflow.keras.callbacks import History
from constants import Constants, Filepaths

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


class Chatbot(object):
    
    def __init__(self, intents_file: str) -> None:
        self.intents_file = intents_file
        self.intents = self._load_intents_from_file()
        
    def _load_intents_from_file(self) -> None:
        with open(self.intents_file, "r") as f:
            self.intents = json.load(f)["intents"]
    
    def train(self):
        pass
    
    def _get_words_and_tags(self) -> tuple:
        tags = set()
        words = []
        tags_to_words = []
        
        lemmatizer = WordNetLemmatizer()
        for intent in self.intents:
            tags.add(intent["tag"])
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list
                            if word not in Constants.IGNORE_LETTERS]
                words.extend(word_list)
                tags_to_words.append((intent["tag"], word_list))
                
        words = sorted(set(words))
        tags = sorted(tags)
        
        return(tags, words, tags_to_words)
    
    def _save_words_and_tags(self, tags: list, words: list) -> None:
        with open(Filepaths.TAGS_AND_WORDS, "wb") as f:
            pickle.dump((tags, words), f)
            
    def _load_words_and_tags(self) -> tuple:
        with open(Filepaths.TAGS_AND_WORDS, "rb") as f:
            tags, words = pickle.load(f)
        return (tags, words)
            
    def _create_training_data(self, tags: list, words: list, tags_to_words: list) -> tuple:
        train = []
        for tag, word_list in tags_to_words:
            bag_for_word_list = [1 if word in word_list else 0 for word in words]
            bag_for_tag = [0] * len(tags)
            bag_for_tag[tags.index(tag)] = 1
            train.append([bag_for_tag, bag_for_word_list])
            
        random.shuffle(train)
        train = np.array(train, dtype=object)
        X_train = np.array(list(train[:, 1])) # word_lists
        y_train = np.array(list(train[:, 0])) # tags
        return (X_train, y_train)
    
    def _train_model(self, X_train: np.array, y_train: np.array,
                     optimizer: Optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                     loss: str = "categorical_crossentropy", epochs: int = 200, batch_size: int = 5
                     ) -> History:
        model = Sequential()
        model.add(Dense(128, input_shape=(X_train.shape[1],), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(y_train.shape[1], activation="softmax"))

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return history