import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel

form_window = uic.loadUiType('./movie_app.ui')[0]


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('../models/tfidf_restaurant_review.mtx').tocsr()
        with open('../../models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)

        self.embedding_model = Word2Vec.load('../models/word2vec_restaurant_review.model')
        self.df_reviews = pd.read_csv('../../data_naver/cleaned_data.csv')
        self.titles = list(self.df_reviews['names'])
        self.titles.sort()
        for title in self.titles:
            self.comboBox.addItem(title)

        model = QStringListModel()
        model.setStringList(self.titles)
        completer = QCompleter()
        completer.setModel(model)
        self.lineEdit.setCompleter(completer)

        self.comboBox.currentIndexChanged.connect(self.combobox_slot)
        self.pushButton.clicked.connect(self.btn_slot)



    def btn_slot(self):
        keyword = self.lineEdit.text()
        if keyword in self.titles:
            recommendation = self.recommendation_by_movie_title(keyword)
        else:
            recommendation = self.recommendation_by_keyword(keyword)

        if recommendation:
            self.lbl_recommendation.setText(recommendation)


    def recommendation_by_keyword(self, keyword):
        try:
            sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)
        except:
            self.lbl_recommendation.setText('I DONT KNOW, HUMAN\nDo you want a die?')
            for i in range(10, 50):
                self.lbl_recommendation.font().setPointSize(i)
            return 0
        words = [keyword]
        for word, _ in sim_word:
            words.append(word)
        sentence = []
        count = 10
        for word in words:
            sentence = sentence + [word] * count
            count -= 1
        sentence = ' '.join(sentence)
        print(sentence)
        sentence_vec = self.Tfidf.transform([sentence])
        cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        recommendation = '\n'.join(list(recommendation))
        return recommendation

    def combobox_slot(self):
        title = self.comboBox.currentText()
        recommendation = self.recommendation_by_movie_title(title)
        self.lbl_recommendation.setText(recommendation)

    def recommendation_by_movie_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['names'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        recommendation = '\n'.join(list(recommendation))
        return recommendation

    def getRecommendation(self, consine_sim):
        simScore = list(enumerate(consine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]
        movieIdx = [i[0] for i in simScore]
        recmovieList = self.df_reviews.iloc[movieIdx, 0]
        return recmovieList[1:11]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())