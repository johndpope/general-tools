import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

__all__ = ['final_report']

CHARACTERS = ['bart', 'homer', 'lisa', 'marge']

def final_report(y_true, y_pred):
    for char_id in range(y_true.shape[1]):
        print("Char #{} - {}".format(char_id, CHARACTERS[char_id]))

        char_true = y_true[:,char_id]
        char_pred = y_pred[:,char_id]

        acc = accuracy_score(char_true, char_pred)
        print("ACCURACY - {:.3f}".format(acc))
        print(confusion_matrix(char_true, char_pred))
        print(classification_report(char_true, char_pred))

        print("\n\n")
