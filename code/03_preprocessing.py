import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./reviews_naver.csv')
df.info()

df_stopwords = pd.read_csv('./stopwords.csv')
stopwords = list(df_stopwords['stopword'])
stopwords = stopwords + [ '찍다', '양도', '사장', '다녀오다',
'잇다', '볶다', '모시', '오름', '포장', '웨이팅','넘다',
 '진짜', '절대', '먹다',  '니당다', '다시다','아깝다','번창','직원','찌다','매다',
'집다','기고','이담','질도','기질','우리동네','동네','숨다','발견','이집','여기다',
 '나용','나다','추하다','자다','오다','가요','싶다','이예','일산','굳다']

okt = Okt()
cleaned_sentences = []
for review in df.reviews:
    review = re.sub('[^가-힣]', ' ', review)
    tokened_review = okt.pos(review, stem=True)
    print(tokened_review)
    df_token = pd.DataFrame(tokened_review, columns=['word', 'class'])
    df_token = df_token[(df_token['class']=='Noun') |
                        (df_token['class']=='Adjective') |
                        (df_token['class']=='Verb')]
    words = []
    for word in df_token.word:
        if 1 < len(word):
            if word not in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
df['reviews'] = cleaned_sentences
df.dropna(inplace=True)
df.to_csv('./cleaned_naver_reviews.csv', index=False)

print(df.head())
df.info()

df = pd.read_csv('./cleaned_naver_reviews.csv')
df.dropna(inplace=True)
df.info()
df.to_csv('./cleaned_naver_reviews.csv', index=False)
