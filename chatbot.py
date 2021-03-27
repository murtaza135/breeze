import random
import time
import webbrowser
import os
import pathlib
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
from tensorflow.keras.models import load_model
from exceptions import NoModelTrainedError

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


class Chatbot:
    IGNORE_LETTERS = [char for char in string.punctuation] + ["'s"]
    
    def __init__(self, name: str = "Chatbot") -> None:
        self.name = name
        self._intents = None
        self._tags = set()
        self._words = []
        self._tags_to_words = []
        self._model = None
        
    @property
    def model(self) -> Sequential:
        return self._model
    
    
    def train(self, intents_file: str,
              optimizer: Optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              loss: str = "categorical_crossentropy", epochs: int = 200, batch_size: int = 5):
        self._load_intents(intents_file)
        self._extract_words_and_tags()
        X_train, y_train = self._create_training_data()
        self._train_model(X_train, y_train, optimizer, loss, epochs, batch_size)
    
    def _load_intents(self, intents_file: str) -> None:
        with open(intents_file, "r") as f:
            self._intents = json.load(f)["intents"]
        
    def _extract_words_and_tags(self) -> None:
        lemmatizer = WordNetLemmatizer()
        for intent in self._intents:
            self._tags.add(intent["tag"])
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list
                            if word not in Chatbot.IGNORE_LETTERS]
                self._words.extend(word_list)
                self._tags_to_words.append((intent["tag"], word_list))
                
        self._words = sorted(set(self._words))
        self._tags = sorted(self._tags)
        
    def _create_training_data(self) -> tuple:
        train = []
        for tag, word_list in self._tags_to_words:
            bag_for_word_list = [1 if word in word_list else 0 for word in self._words]
            bag_for_tag = [0] * len(self._tags)
            bag_for_tag[self._tags.index(tag)] = 1
            train.append([bag_for_tag, bag_for_word_list])
            
        random.shuffle(train)
        train = np.array(train, dtype=object)
        X_train = np.array(list(train[:, 1])) # word_lists
        y_train = np.array(list(train[:, 0])) # tags
        return (X_train, y_train)
    
    def _train_model(self, X_train: np.array, y_train: np.array,
                     optimizer: Optimizer, loss: str, epochs: int, 
                     batch_size: int):
        self._model = Sequential()
        self._model.add(Dense(128, input_shape=(len(self._words),), activation="relu"))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(64, activation="relu"))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(len(self._tags), activation="softmax"))

        self._model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        self._model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        print("===== Fininished training model =====")
        
        
    def save(self, path: str, name: str = None) -> None:
        if self._model is None:
            raise NoModelTrainedError("You must train the chatbot using an intents file before saving it")
        
        if os.path.exists(path) and os.path.isdir(path):
            name = self.name if name is None else name
            new_path = os.path.join(path, name)
            
            model_path = os.path.join(new_path, "model.h5")
            self._model.save(model_path)
            
            words_and_tags_path = os.path.join(new_path, "words_and_tags.pickle")
            with open(words_and_tags_path, "wb") as f:
                pickle.dump((self._tags, self._words, self._tags_to_words), f)
                
            intents_path = os.path.join(new_path, "intents.pickle")
            with open(intents_path, "wb") as f:
                pickle.dump(self._intents, f)
        else:
            raise NotADirectoryError("Could not save chatbot model")
                
    
    def load(self, path: str) -> None:
        model_path = os.path.join(path, "model.h5")
        self._model = load_model(model_path)
        
        words_and_tags_path = os.path.join(path, "words_and_tags.pickle")
        with open(words_and_tags_path, "rb") as f:
            self._tags, self._words, self._tags_to_words = pickle.load(f)
            
        intents_path = os.path.join(path, "intents.pickle")
        with open(intents_path, "rb") as f:
            self._intents = pickle.load(f)
            
        name = os.path.basename(path)
        self.name = name
    
    @classmethod
    def load_model(cls, path: str):
        model_path = os.path.join(path, "model.h5")
        model = load_model(model_path)
        
        words_and_tags_path = os.path.join(path, "words_and_tags.pickle")
        with open(words_and_tags_path, "rb") as f:
            tags, words, tags_to_words = pickle.load(f)
            
        intents_path = os.path.join(path, "intents.pickle")
        with open(intents_path, "rb") as f:
            intents = pickle.load(f)
        
        name = os.path.basename(path)
        chatbot = cls(name)
        chatbot._intents = intents
        chatbot._tags = tags
        chatbot._words = words
        chatbot._tags_to_words = tags_to_words
        chatbot._model = model
        return chatbot