# ==============================================================================
# Aleksandar Cenic 2023.
# ==============================================================================
# /\     | |        | |                              | |
# /  \    | |   ___  | | __  ___    __ _   _ __     __| |   __ _   _ __
# / /\ \   | |  / _ \ | |/ / / __|  / _` | | '_ \   / _` |  / _` | | '__|
# / ____ \  | | |  __/ |   <  \__ \ | (_| | | | | | | (_| | | (_| | | |
# /_/    \_\ |_|  \___| |_|\_\ |___/  \__,_| |_| |_|  \__,_|  \__,_| |_|
# _____                  _
# / ____|                (_)
# | |        ___   _ __    _    ___
# | |       / _ \ | '_ \  | |  / __|
# | |____  |  __/ | | | | | | | (__
# \_____|  \___| |_| |_| |_|  \___|

# ==============================================================================
# Import section.
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import neattext.functions as nfx
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from reldi_tokeniser.tokeniser import ReldiTokeniser
import serbianStopWords as sb
from nltk import ngrams
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')


# ==============================================================================

# ==============================================================================


class SerbianTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.m_reldi = ReldiTokeniser('sr', conllu=True, nonstandard=False, tag=False)
        self.m_list_of_lines = None

    def __call__(self, doc):
        self.m_list_of_lines = [el + '\n' for el in doc.split('\n')]
        self.m_text = self.m_reldi.run(self.m_list_of_lines, mode='object')
        return [t['text'] for t in self.m_text[0][0]['sentence'] if t['text'] not in self.ignore_tokens]


# ==============================================================================

# ==============================================================================


class RougeMetrics:

    # Constructor of class.
    # @param self > Object of class.
    # @param n > Numebr of n-grams.

    def __init__(self, n):
        self.m_n = n  # Nuber of ngrams.
        self.m_serbianTokenizer = SerbianTokenizer()  # Serbian tokenizer.
        self.m_precision = None  # Precision.
        self.m_recall = None  # Recall.
        self.m_fScore = None  # FScore.
        self.m_ngrams_reference = []  # List of ngrams for reference text.
        self.m_ngrams_generated = []  # List of ngrams for generated text.

    # @brief Call default method.
    # @param self > Object of class.
    # @param generatedText > Generated text.
    # @param referenceText > Reference text.

    def __call__(self, generatedText, referenceText):
        reference = self.m_serbianTokenizer(referenceText)
        generated = self.m_serbianTokenizer(generatedText)

        ngrams_reference = list(ngrams(reference, self.m_n))
        ngrams_generated = list(ngrams(generated, self.m_n))

        self.m_ngrams_reference = np.empty(len(ngrams_reference), dtype=object)
        self.m_ngrams_generated = np.empty(len(ngrams_generated), dtype=object)

        self.m_ngrams_reference[:] = ngrams_reference
        self.m_ngrams_generated[:] = ngrams_generated

        count_ref_ngrams_in_gen = np.count_nonzero(np.isin(self.m_ngrams_reference, self.m_ngrams_generated))
        if len(self.m_ngrams_generated) == 0:
            self.m_recall = 0
            self.m_fScore = 0

        else:
            self.m_recall = count_ref_ngrams_in_gen / len(self.m_ngrams_generated)

        if len(self.m_ngrams_reference) == 0:
            self.m_precision = 0
            self.m_fScore = 0

        else:
            self.m_precision = count_ref_ngrams_in_gen / len(self.m_ngrams_reference)
        if self.m_recall > 0 and self.m_precision > 0:
            self.m_fScore = 2 * (self.m_precision * self.m_recall) / (self.m_precision + self.m_recall)
        else:
            self.m_fScore = 0

        return self.m_recall, self.m_precision, self.m_fScore

    # @brief Method for calculation LCS

    def lcs(self, X, Y):

        m = len(X)
        n = len(Y)

        L = [[None] * (n + 1) for i in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

                    #
        return L[m][n]

    # @brief Method for calculate RougeL.
    # @param self > Object of class.
    # @param generatedText > Generated text.
    # @param referenceText > Reference text.

    def RougeL(self, generatedText, referenceText):
        reference = self.m_serbianTokenizer(referenceText)
        generated = self.m_serbianTokenizer(generatedText)

        ngrams_reference = list(ngrams(reference, 1))
        ngrams_generated = list(ngrams(generated, 1))

        self.m_ngrams_reference = np.empty(len(ngrams_reference), dtype=object)
        self.m_ngrams_generated = np.empty(len(ngrams_generated), dtype=object)

        self.m_ngrams_reference[:] = ngrams_reference
        self.m_ngrams_generated[:] = ngrams_generated

        count_ref_ngrams_in_gen = self.lcs(self.m_ngrams_reference, self.m_ngrams_generated)
        if len(self.m_ngrams_generated) == 0:
            self.m_recall = 0
            self.m_fScore = 0

        else:
            self.m_recall = count_ref_ngrams_in_gen / len(self.m_ngrams_generated)

        if len(self.m_ngrams_reference) == 0:
            self.m_precision = 0
            self.m_fScore = 0

        else:
            self.m_precision = count_ref_ngrams_in_gen / len(self.m_ngrams_reference)

        if self.m_recall > 0 and self.m_precision > 0:
            self.m_fScore = 2 * (self.m_precision * self.m_recall) / (self.m_precision + self.m_recall)
        else:
            self.m_fScore = 0

        return self.m_recall, self.m_precision, self.m_fScore


# metric = RougeMetrics(1)
# precision, recall, fsocre = metric("pajton je objektno-orjentisano programski jezik.",
#                                    "pajton je objektno-orjentisani progrmaski jezik u obliku intertetator")
# print(precision, recall, fsocre)


# ==============================================================================

# ==============================================================================


# English Rouge Metrics
# ==============================================================================

# ==============================================================================
#
# import nltk
# import numpy as np
#
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk import ngrams
# from reldi_tokeniser.tokeniser import ReldiTokeniser
#
# nltk.download('punkt_tab')
# # ==============================================================================
#
# # ==============================================================================
#
# class EnglishTokenizer:
#     ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
#
#     def __init__(self):
#         pass
#
#     def __call__(self, doc):
#         tokens = word_tokenize(doc.lower())
#         return [t for t in tokens if t.isalpha() and t not in self.ignore_tokens]
#
#
# # ==============================================================================
#
# # ==============================================================================
#
# class RougeMetricsEnglish:
#     def __init__(self, n):
#         self.m_n = n
#         self.m_englishTokenizer = EnglishTokenizer()
#         self.m_precision = None
#         self.m_recall = None
#         self.m_fScore = None
#         self.m_ngrams_reference = []
#         self.m_ngrams_generated = []
#
#     def __call__(self, generatedText, referenceText):
#         reference = self.m_englishTokenizer(referenceText)
#         generated = self.m_englishTokenizer(generatedText)
#
#         ngrams_reference = list(ngrams(reference, self.m_n))
#         ngrams_generated = list(ngrams(generated, self.m_n))
#
#         self.m_ngrams_reference = np.empty(len(ngrams_reference), dtype=object)
#         self.m_ngrams_generated = np.empty(len(ngrams_generated), dtype=object)
#
#         self.m_ngrams_reference[:] = ngrams_reference
#         self.m_ngrams_generated[:] = ngrams_generated
#
#         count_ref_ngrams_in_gen = np.count_nonzero(np.isin(self.m_ngrams_reference, self.m_ngrams_generated))
#
#         if len(self.m_ngrams_generated) == 0:
#             self.m_recall = 0
#             self.m_fScore = 0
#         else:
#             self.m_recall = count_ref_ngrams_in_gen / len(self.m_ngrams_generated)
#
#         if len(self.m_ngrams_reference) == 0:
#             self.m_precision = 0
#             self.m_fScore = 0
#         else:
#             self.m_precision = count_ref_ngrams_in_gen / len(self.m_ngrams_reference)
#
#         if self.m_recall > 0 and self.m_precision > 0:
#             self.m_fScore = 2 * (self.m_precision * self.m_recall) / (self.m_precision + self.m_recall)
#         else:
#             self.m_fScore = 0
#
#         return self.m_recall, self.m_precision, self.m_fScore
#
#     def lcs(self, X, Y):
#         m = len(X)
#         n = len(Y)
#         L = [[None] * (n + 1) for i in range(m + 1)]
#         for i in range(m + 1):
#             for j in range(n + 1):
#                 if i == 0 or j == 0:
#                     L[i][j] = 0
#                 elif X[i - 1] == Y[j - 1]:
#                     L[i][j] = L[i - 1][j - 1] + 1
#                 else:
#                     L[i][j] = max(L[i - 1][j], L[i][j - 1])
#         return L[m][n]
#
#     def RougeL(self, generatedText, referenceText):
#         reference = self.m_englishTokenizer(referenceText)
#         generated = self.m_englishTokenizer(generatedText)
#
#         ngrams_reference = list(ngrams(reference, 1))
#         ngrams_generated = list(ngrams(generated, 1))
#
#         self.m_ngrams_reference = np.empty(len(ngrams_reference), dtype=object)
#         self.m_ngrams_generated = np.empty(len(ngrams_generated), dtype=object)
#
#         self.m_ngrams_reference[:] = ngrams_reference
#         self.m_ngrams_generated[:] = ngrams_generated
#
#         count_ref_ngrams_in_gen = self.lcs(self.m_ngrams_reference, self.m_ngrams_generated)
#
#         if len(self.m_ngrams_generated) == 0:
#             self.m_recall = 0
#             self.m_fScore = 0
#         else:
#             self.m_recall = count_ref_ngrams_in_gen / len(self.m_ngrams_generated)
#
#         if len(self.m_ngrams_reference) == 0:
#             self.m_precision = 0
#             self.m_fScore = 0
#         else:
#             self.m_precision = count_ref_ngrams_in_gen / len(self.m_ngrams_reference)
#
#         if self.m_recall > 0 and self.m_precision > 0:
#             self.m_fScore = 2 * (self.m_precision * self.m_recall) / (self.m_precision + self.m_recall)
#         else:
#             self.m_fScore = 0
#
#         return self.m_recall, self.m_precision, self.m_fScore
#
#
# # metric = RougeMetricsEnglish(1)
# # precision, recall, fscore = metric("Python is an object-oriented programming language.",
# #                                    "Python is an object-oriented interpreted programming language.")
# # print(precision, recall, fscore)