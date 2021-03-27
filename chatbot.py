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
    
    
    def __init__(self, name: str = "Chatbot", error_threshold: float = 0.7) -> None:
        self.name = name
        self._intents = None
        # self._intents_func_mapping = {}
        self._tags = set()
        self._words = []
        self._tags_to_words = []
        self._model = None
        self._error_threshold = error_threshold
        
        self.no_understanding_messages = [
            "Sorry, I could not understand you",
            "Could you say that again please?"
        ]
        
    @property
    def model(self) -> Sequential:
        return self._model
    
    def get_tags(self) -> list:
        return self._tags.copy()
    
    def add_no_understanding_message(self, message: str) -> None:
        self.no_understanding_messages.append(message)
        
    def set_error_threshold(self, value: float):
        if 0 <= value <= 1:
            self._error_threshold = value
        else:
            raise ValueError("The threshold value must lie between 0 and 1, inclusive")
        
    def get_error_threshold(self) -> float:
        return self._error_threshold
    
    
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
            raise NoModelTrainedError(
                "You must train the chatbot using an intents file before saving it"
            )
        
        if os.path.exists(path) and os.path.isdir(path):
            name = self.name if name is None else name
            save_path = os.path.join(path, name)
            
            model_path = os.path.join(save_path, "model.h5")
            self._model.save(model_path)
            
            # tf model cannot be pickled, therefore we temporarily remove it from the objects state
            model = self._model
            self._model = None
            chatbot_path = os.path.join(save_path, "chatbot.pickle")
            with open(chatbot_path, "wb") as f:
                pickle.dump(self, f)
            self._model = model
        else:
            raise NotADirectoryError("Could not save chatbot model")
                
    
    def load(self, path: str) -> None:
        chatbot_path = os.path.join(path, "chatbot.pickle")
        with open(chatbot_path, "rb") as f:
            chatbot = pickle.load(f)
            
        self.__dict__.update(chatbot.__dict__)
        
        model_path = os.path.join(path, "model.h5")
        self._model = load_model(model_path)
        
    @classmethod
    def load_chatbot(cls, path: str):
        chatbot_path = os.path.join(path, "chatbot.pickle")
        with open(chatbot_path, "rb") as f:
            chatbot = pickle.load(f)
        
        model_path = os.path.join(path, "model.h5")
        model = load_model(model_path)
        
        chatbot._model = model
        return chatbot
    
    
    def greet(self) -> None:
        print(f"Hi! I am {self.name}!")
        
    def prompt_and_respond(self) -> None:
        message = input("")
        self.respond(message)
    
    def respond(self, message: str) -> None:
        tag = self._predict_tag(message)
        reply = self._select_reply(tag)
        print(reply)
    
    def _predict_tag(self, message: str) -> str:
        bag = self._bag_of_words(message)
        result = self.model.predict(bag.reshape(1,-1))[0]
        if result.max() > self._error_threshold:
            return self._tags[np.argmax(result)]
        return None
    
    def _bag_of_words(self, message: str) -> np.array:
        word_list = self._clean_message(message)
        bag_for_word_list = [1 if word in word_list else 0 for word in self._words]
        return np.array(bag_for_word_list)
    
    def _clean_message(self, message: str) -> list:
        lemmatizer = WordNetLemmatizer()
        word_list = nltk.word_tokenize(message)
        word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list 
                    if word not in Chatbot.IGNORE_LETTERS]
        return word_list
    
    def _select_reply(self, tag: str) -> str:
        if tag is None:
            return random.choice(self.no_understanding_messages)
        else:
            for intent in self._intents:
                if intent["tag"] == tag:
                    return random.choice(intent["responses"])