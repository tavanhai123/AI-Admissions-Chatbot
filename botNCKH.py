# -*- coding: utf-8 -*-

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

# Load intents từ tệp intents.json
with open('e:/code python/intents.json',  encoding='utf-8') as json_data:

    intents = json.load(json_data)

# Khởi tạo các danh sách và cấu trúc dữ liệu
words = []
classes = []
documents = []
stop_words = ['?', 'ạ', 'hả', 'the']

# Khởi tạo stemmer
stemmer = LancasterStemmer()

# Xử lý intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize từng từ trong pattern
        w = nltk.word_tokenize(pattern)
        # Thêm từ vào danh sách từ
        words.extend(w)
        # Thêm cặp (pattern, tag) vào danh sách documents
        documents.append((w, intent['tag']))
        # Thêm tag vào danh sách classes nếu chưa có trong đó
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem và loại bỏ stop words từ danh sách từ
words = [stemmer.stem(w.lower()) for w in words if w not in stop_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Tạo tập huấn luyện
training = []
output = []
output_empty = [0] * len(classes)

# Tạo bag of words cho mỗi câu
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Xáo trộn dữ liệu huấn luyện
random.shuffle(training)
training = np.array(training)

# Tạo tập train và tập test
train_x = list(training[:,0])
train_y = list(training[:,1])

# Khởi tạo lại đồ thị mặc định của TensorFlow
tf.compat.v1.reset_default_graph()


# Xây dựng mạng nơ-ron
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# Định nghĩa và cài đặt model
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Bắt đầu huấn luyện
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)

# Lưu model
model.save('model.tflearn')

# Lưu dữ liệu huấn luyện
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data', 'wb'))

# Khôi phục cấu trúc dữ liệu từ dữ liệu đã lưu
data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# Định nghĩa hàm clean_up_sentence và bow
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

# Kiểm tra hàm bow với câu 'I want to special food'
bow('I want to special food', words)

# Cấu trúc dữ liệu để lưu ngữ cảnh người dùng
context = {}

# Ngưỡng lỗi
ERROR_THRESHOLD = 0.25

# Hàm phân loại
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list
# Hàm phản hồi
import random

# Định nghĩa hàm response
def response(sentence):
    results = classify(sentence)
    if results:
        for intent in intents['intents']:
            if intent['tag'] == results[0][0]:
                responses = intent['responses']
                if len(responses)== 1 :
                    return responses[0]
                else :
                    return random.choice(responses)
    else:
        return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể giải thích thêm không?"

# Vòng lặp để nhận câu hỏi từ người dùng và hiển thị câu trả lời của bot
while True:
    # Nhận câu hỏi từ người dùng
    user_input = input("Bạn: ")

    # Gọi hàm response để nhận câu trả lời
    bot_response = response(user_input)

    # Hiển thị câu trả lời của bot
    print("Bot:", bot_response)
