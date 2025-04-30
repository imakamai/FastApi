# English Rouge Metrics
# ==============================================================================

# ==============================================================================

import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from reldi_tokeniser.tokeniser import ReldiTokeniser

nltk.download('punkt_tab')
# ==============================================================================

# ==============================================================================

class EnglishTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        pass

    def __call__(self, doc):
        tokens = word_tokenize(doc.lower())
        return [t for t in tokens if t.isalpha() and t not in self.ignore_tokens]


# ==============================================================================

# ==============================================================================

class RougeMetricsEnglish:
    def __init__(self, n):
        self.m_n = n
        self.m_englishTokenizer = EnglishTokenizer()
        self.m_precision = None
        self.m_recall = None
        self.m_fScore = None
        self.m_ngrams_reference = []
        self.m_ngrams_generated = []

    def __call__(self, generatedText, referenceText):
        reference = self.m_englishTokenizer(referenceText)
        generated = self.m_englishTokenizer(generatedText)

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
        return L[m][n]

    def RougeL(self, generatedText, referenceText):
        reference = self.m_englishTokenizer(referenceText)
        generated = self.m_englishTokenizer(generatedText)

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


# metric = RougeMetricsEnglish(1)
# precision, recall, fscore = metric("Python is an object-oriented programming language.",
#                                    "Python is an object-oriented interpreted programming language.")
# print(precision, recall, fscore)
