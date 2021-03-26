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
from constants import Constant, Filepath

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


class Breeze():
    
    def __init__(self, intents_file: str) -> None:
        self.intents_file = intents_file
        self.intents = None
        self.tags = set()
        self.words = []
        self.tags_to_words = []
        self.model = None
        
        self._load_intents()
    

    def _load_intents(self) -> None:
        if os.path.exists(self.intents_file):
            with open(self.intents_file, "r") as f:
                self.intents = json.load(f)["intents"]
        else:
            raise FileNotFoundError(f"Could not find '{self.intents_file}'")
            
    def _save_words_and_tags(self) -> None:
        if not os.path.exists(Filepath.TAGS_AND_WORDS):
            path = pathlib.Path(os.path.dirname(Filepath.TAGS_AND_WORDS))
            path.mkdir(parents=True, exist_ok=True)
            
        with open(Filepath.TAGS_AND_WORDS, "wb") as f:
            pickle.dump((self.tags, self.words, self.tags_to_words), f)
            
    def _load_words_and_tags(self) -> None:
        with open(Filepath.TAGS_AND_WORDS, "rb") as f:
            self.tags, self.words, self.tags_to_words = pickle.load(f)
    
    
    def load_model(self) -> None:
        if os.path.exists(Filepath.MODEL):
            self.model = load_model(Filepath.MODEL)
        else:
            print(f"Error: Could not load model '{Filepath.MODEL}'")
            self.model = None
            
    
    def train(self) -> None:
        self._get_words_and_tags()
        self._save_words_and_tags()
        X_train, y_train = self._create_training_data()
        self._train_model(X_train, y_train)
    
    def _get_words_and_tags(self) -> None:
        lemmatizer = WordNetLemmatizer()
        for intent in self.intents:
            self.tags.add(intent["tag"])
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list
                            if word not in Constant.IGNORE_LETTERS]
                self.words.extend(word_list)
                self.tags_to_words.append((intent["tag"], word_list))
                
        self.words = sorted(set(self.words))
        self.tags = sorted(self.tags)
            
    def _create_training_data(self) -> tuple:
        train = []
        for tag, word_list in self.tags_to_words:
            bag_for_word_list = [1 if word in word_list else 0 for word in self.words]
            bag_for_tag = [0] * len(self.tags)
            bag_for_tag[self.tags.index(tag)] = 1
            train.append([bag_for_tag, bag_for_word_list])
            
        random.shuffle(train)
        train = np.array(train, dtype=object)
        X_train = np.array(list(train[:, 1])) # word_lists
        y_train = np.array(list(train[:, 0])) # tags
        return (X_train, y_train)
    
    def _train_model(self, X_train: np.array, y_train: np.array,
                     optimizer: Optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                     loss: str = "categorical_crossentropy", epochs: int = 200, batch_size: int = 5):
        model = Sequential()
        model.add(Dense(128, input_shape=(len(self.words),), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.tags), activation="softmax"))

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        self.model = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        model.save(Filepath.MODEL, self.model)
        
        print("\n==============================================================================")
        print("==================================== DONE ====================================")
        print("==============================================================================")