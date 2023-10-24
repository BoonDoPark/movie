'''
출처 : https://wikidocs.net/120791
출처 : https://dev-ryuon.tistory.com/
'''

import random
import sys

import numpy as np
import pandas as pd
import requests
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QMainWindow
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from process_visualize import MovieVisualize
from ui_utils_table import QTableFormat, QTableWidgetUtils

df = pd.read_csv('movie_data.csv')
df.drop(df.loc[df['평점'] == 0.00].index, inplace=True)
# df.drop(df.loc[df['평점'] == 10.0].index, inplace=True)
df = df.reset_index()
df['줄거리'] = df['줄거리'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
df = df.reset_index()
x_train = df['장르']
y_train = df['평점']
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x_train)
similarity = cosine_similarity(x, x)
similarity = similarity.argsort()
similarity = similarity[:, ::-1]


def get_recommend_movie(title):
    search_genre = df[df['제목'] == title]
    search_genre_index = search_genre.index.values
    similarity_index = similarity[search_genre_index, :30].reshape(-1)
    similarity_index = similarity_index[similarity_index != search_genre_index]
    result = df.iloc[similarity_index].sort_values('평점', ascending=False)[:20]
    return result


shuffle = list(df['제목'].values)
movie = random.choice(shuffle)
genre_df = get_recommend_movie(movie)
genre_df['평점'] = genre_df['평점'].astype(object)

genre_list = []
for i in range(len(genre_df)):
    genre_list.append(list(genre_df.iloc[i].values))

print(genre_list)

df = pd.read_table('ratings_train.txt')
df.to_csv('emotion_analysis.csv', index=False,encoding='utf-8-sig')
df = pd.read_csv('emotion_analysis.csv')
df = df.rename(columns={'document':'review', 'label':'positive'})

df['review'].nunique(), df['positive'].nunique()
df.drop_duplicates(subset=['review'], inplace=True)

df = df.dropna(how='any')
df['review'] = df['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
df = df.reset_index()

from konlpy.tag import Okt

okt = Okt()
temp_word = []
stop_words = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와',
              '한', '하다', '적', '인', '으로도', '끼리', '을', '까지', '에게도', '이니', '만으로', '이다']

for sentence in df['review']:
    temp = okt.morphs(sentence)
    temp = [word for word in temp if not word in stop_words]
    temp_word.append(temp)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer()
tokenizer.fit_on_texts(temp_word)

threshold = 2
total_count = len(tokenizer.word_index)
print(total_count)
low_count = 0

for key, value in tokenizer.word_counts.items():
    if value < threshold:
        low_count += 1

vocab_size = total_count - low_count + 1
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(temp_word)
temp_word = tokenizer.texts_to_sequences(temp_word)
temp_positive = np.array(df['positive'])

drop_word = [idx for idx, sentence in enumerate(temp_word) if len(sentence) < 1]
temp_word = np.delete(temp_word, drop_word, axis=0)
temp_positive = np.delete(temp_positive, drop_word, axis=0)


def per_sentence(max_len: int, word: list):
    count = 0
    for s in word:
        if len(s) < max_len:
            count += 1
    print((count / len(word)) * 100)

max_len = 28
per_sentence(max_len, temp_word)

temp_word = pad_sequences(temp_word, maxlen=max_len)


from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_len, mask_zero=True))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(temp_word, temp_positive, epochs=60, callbacks=[es, mc], batch_size=60, validation_split=0.2)

loaded_model = load_model('model.h5')
print(loaded_model.evaluate(temp_word, temp_positive))


def get_review(code):
    reviews=[]
    grades=[]
    for p in range(1,5000):
        url='https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code={}&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page={}'.format(code ,p)
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'lxml')
        review_lists=soup.find_all('li')[6:]

        for idx, j in enumerate(review_lists):
            review=j.find('span', id="_filtered_ment_{}".format(idx)).get_text().strip()
            grade=int(j.find('div',class_="star_score").get_text())
            if '' != review and review not in reviews:
                reviews.append(review)
                grades.append(grade)
    zip_grades_reviews = list(zip(reviews, grades))
    return zip_grades_reviews


def sentiment_predict(new_sentence):
    new_sentence = okt.morphs(new_sentence)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stop_words]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(pad_new))  # 예측
    if (score > 0.5):
        return str(round(score * 100, 2)) + ' 긍정'
    else:
        return str(round((1 - score) * 100, 2)) + ' 부정'


positive = []
negative = []
ranks = []


def get_emotion(code):
    movie = get_review(code)
    for review, rank in movie:
        emotion = sentiment_predict(review)
        if '긍정' in emotion:
            emotion = emotion.replace('긍정', '')
            positive.append(emotion)
            ranks.append(rank)

        else:
            emotion = emotion.replace('부정', '')
            negative.append(emotion)
            ranks.append(rank)

    return positive, negative, ranks


FORM_CLASS = uic.loadUiType('movie_data_visualize.ui')[0]


class WindowMovie(QMainWindow, FORM_CLASS):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.refresh_table()

        self.init_signal()
        self.show()

    def init_signal(self):
        self.tablewidget_movie_list.itemClicked.connect(self.on_clicked_refresh_chart)

    def refresh_table(self):
        self.tablewidget_movie_list: QTableWidget
        self.tablewidget_movie_list.setRowCount(0)

        form_display = QTableFormat()
        form_user = QTableFormat()

        for genre in genre_list:
            display_row = tuple(genre[3:6])
            user_row = (genre[2],) * len(display_row)

            form_display.append_by_row(display_row)
            form_user.append_by_row(user_row)

        QTableWidgetUtils.refresh_by_items(self.tablewidget_movie_list, form_display, form_user)
        QTableWidgetUtils.resize_table_widget(self.tablewidget_movie_list)

    def on_clicked_refresh_chart(self, item: QTableWidgetItem):
        selected_movie= item.data(Qt.UserRole)

        get_emotion(selected_movie)

        x = ['긍정', '부정']
        y = [len(positive), len(negative)]

        emotion_review = positive + negative

        data = pd.DataFrame({'댓글': emotion_review, '평점': ranks})
        rate = data['평점'].value_counts()

        MovieVisualize.run(rate, x, y, f'img/{selected_movie}.png')

        pixmap = QPixmap()
        pixmap.load(f'img/{selected_movie}.png')
        self.movie_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = WindowMovie()
    app.exec()
