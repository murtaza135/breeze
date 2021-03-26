import string
import os

class Constants:
    IGNORE_LETTERS = [char for char in string.punctuation] + ["'s"]
    
class Filepaths:
    TAGS_AND_WORDS = os.join("data", "tags_and_words.pickle")