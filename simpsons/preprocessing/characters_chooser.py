import numpy as np

__all__ = ['CharactersChooser']

class CharactersChooser:

    def __init__(self, characters=None):
        self.characters = characters

    def transform(self, X, y):
        if not self.characters:
            # take all characters
            return X, y

        else:
            # filter only requested chars
            return X, y[:,self.characters]
