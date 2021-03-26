import string
import os

class Constant:
    NAME = "Breeze"
    IGNORE_LETTERS = [char for char in string.punctuation] + ["'s"]
    
class Filepath:
    TAGS_AND_WORDS = os.path.join("data", "tags_and_words.pickle")
    MODEL = os.path.join("models", f"{Constant.NAME}.h5")
    INTENTS = os.path.join("data", "intents.json")