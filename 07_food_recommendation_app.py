import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel

form_window = uic.loadUiType('./movie_recommendation.ui')[0]


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('./models/Tfidf_naver_review.mtx').tocsr()
        with open('./models/tfidf_naver.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_naver_review.model')
        self.df_reviews = pd.read_csv('./cleaned_naver_reviews.csv')
        self.names = list(self.df_reviews['names'])
        self.names.sort()
        for title in self.names:
            self.comboBox.addItem(title)

        model = QStringListModel()
        model.setStringList(self.names)
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)

        self.comboBox.currentIndexChanged.connect(self.combobox_slot)
        self.btn_recommendation.clicked.connect(self.btn_slot)

    def btn_slot(self):
        keyword = self.le_keyword.text()
        if keyword in self.names:
            recommendation = self.recommendation_by_movie_title(keyword)
        else:
            recommendation = self.keyword_recommendation(keyword)

        if recommendation:
            self.lbl_recommendation.setText(recommendation)

    def combobox_slot(self):
        title = self.comboBox.currentText()
        print(title)
        recommendation = self.recommendation_by_movie_title(title)
        print('debug01')
        self.lbl_recommendation.setText(recommendation)
        print('debug02')

    def recommendation_by_keyword(self, key_word):
        try:
            sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)
        except:
            self.lbl_recommendation.setText('제가 모르는 단어에요 ㅠㅠ')
            return 0

        words = [key_word]
        for word, _ in sim_word:
            words.append(word)
        setence = []
        count = 10
        for word in words:
            setence = setence + [word] * count
            count -= 1
        setence = ' '.join(setence)
        print(setence)
        setence_vec = self.Tfidf.transform([setence])
        cosine_sim = linear_kernel(setence_vec, self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        recommendation = '\n'.join(list(recommendation))
        return recommendation

    def recommendation_by_movie_title(self, title):

        movie_idx = self.df_reviews[self.df_reviews['names'] == title].index[0]
        print(movie_idx)
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        recommendation = '\n'.join(list(recommendation))
        return recommendation

    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:50]
        movieIdx = [i[0] for i in simScore]
        recmovieList = self.df_reviews.iloc[movieIdx, 0]
        return recmovieList[1:50]

    def keyword_recommendation(self, keyword):
        words = keyword.split(' ')
        list = []
        score = []
        for idx, word in enumerate(words):
            print(word)
            recommendation = self.recommendation_by_keyword(word).split('\n')
            for rest in recommendation:
                if rest in list:
                    score[list.index(rest)] = score[list.index(rest)] + (100 - recommendation.index(rest))
                else:
                    list.append(rest)
                    if idx == 0:
                        score.append(200 - recommendation.index(rest) * 2)
                    else:
                        score.append(100 - recommendation.index(rest))
        print(list)
        print(score)
        num = len(score)

        for i in range(0, num):
            for j in range(i + 1, num):
                if score[i] <= score[j]:
                    score[i], score[j] = score[j], score[i]
                    list[i], list[j] = list[j], list[i]

        print(list)
        print(score)
        list_str = ''
        for item in list[0:11]:
            list_str += item + '\n'
        return list_str


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())