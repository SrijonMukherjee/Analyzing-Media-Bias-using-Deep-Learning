#To Run:
# Download all three csv files from https://www.kaggle.com/snapcrack/all-the-news
# Put all files in a folder and name it data
# Put the data folder in the same directory as this script

#Code inspirations:
#https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
#https://github.com/amir-jafari/Deep-Learning/blob/master/Exam_MiniProjects/7-Keras_Exam1_Sample_Codes_S20/2-train_ajafari.py
#https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

import random
import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout, Activation, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Text Vectors
MAX_FEATURES = 30000
EMBED_DIM = 300


#Reading in the three datasets
articles1 = pd.DataFrame(pd.read_csv('data/kaggle/articles1.csv'))
articles2 = pd.DataFrame(pd.read_csv('data/kaggle/articles2.csv'))
articles3 = pd.DataFrame(pd.read_csv('data/kaggle/articles3.csv'))


#removing some of the liberal publications to make data balanced
articles3 = articles3[~articles3['publication'].isin(['Washington Post','Reuters','Vox','Guardian'])]

articles = pd.DataFrame(np.concatenate([articles1,articles2,articles3]), columns = articles1.columns)

articles= articles.drop(columns=['Unnamed: 0', 'url'])
articles= articles[~articles['author'].isna()]

articles['year'].value_counts()

#Preprocessing Steps
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))


articles['title'] = articles['title'].apply(lambda x: x.lower())

articles['title'] = articles['title'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop_words)]))

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

articles['title'] = articles['title'].apply(stem_sentences)

#Adding labels
leanings = []
for row in articles['publication']:
    if row in ['New York Times', 'Business Insider', 'Atlantic','CNN','Atlantic','Buzzfeed News','Guardian', 'Talking Points Memo','NPR']:
        leanings.append('liberal')
    else:
        leanings.append('conservative')

articles['leaning'] = leanings

articles['leaning'].value_counts()
train, test = train_test_split(articles, test_size=0.2, random_state=42)


x1_train = np.array(train['title'])
x1_test = np.array(test['title'])


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
targettrain, targettest = train['leaning'].values.tolist(), test['leaning'].values.tolist()
y1_train = np.array(labelencoder.fit_transform(targettrain))
y1_test = np.array(labelencoder.fit_transform(targettest))

tokenizer = Tokenizer(lower=True, split=" ", num_words=MAX_FEATURES)
tokenizer.fit_on_texts(x1_train)

x1_train_vec = tokenizer.texts_to_sequences(x1_train)
x1_test_vec = tokenizer.texts_to_sequences(x1_test)
MAXLEN = max([len(x) for x in x1_train_vec])
print(f"Max vector length: {MAXLEN}")

# pad with zeros for same vector length
x1_train_vec = sequence.pad_sequences(x1_train_vec, maxlen=MAXLEN, padding="post")
x1_test_vec = sequence.pad_sequences(x1_test_vec, maxlen=MAXLEN, padding="post")

from gensim.models import KeyedVectors

word_vec_model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')


# Creating an embedding matrix
print("preparing embedding matrix...")
words_not_found = []

word_index = tokenizer.word_index
# max unique words to keep
nb_words = min(MAX_FEATURES, len(word_index))
# defining the required matrix dimensions
embedding_matrix = np.zeros((nb_words, EMBED_DIM))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    try:
        embedding_vector = word_vec_model.get_vector(word)
    except KeyError:
        embedding_vector = None
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # the words that are not found in embedding index will be made zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print(f"Null word embeddings: {np.sum(np.sum(embedding_matrix, axis=1) == 0)}")
print(
    f"Some of the words not found:\n"
    f"{' '.join([random.choice(words_not_found) for x in range(0,10)])}"
)



zero_vector = np.zeros((1, 300))

embedding_matrix = np.vstack([embedding_matrix,zero_vector])

from keras.utils import to_categorical
y_cat = to_categorical(y1_train)
ycattest = to_categorical(y1_test)

#creating model
nn = Sequential()
nn.add(Embedding(input_dim= nb_words + 1,
                 output_dim= EMBED_DIM,
                 input_length=MAXLEN,
                 weights=[embedding_matrix],
                 trainable=False,
                 ))
nn.add(Dropout(0.4))
nn.add(Bidirectional(LSTM(128, recurrent_dropout= 0.3, return_sequences=True)))
nn.add(LSTM(128, recurrent_dropout=0.2))
nn.add(Dense(2, activation="softmax"))
opt = tf.keras.optimizers.Adam(learning_rate=1e-03)
nn.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"],
)



hist = nn.fit(
    x1_train_vec,
    y_cat,
    batch_size=32,
    epochs=10,
    validation_split = 0.33
)


#Diagnostic Plots

import matplotlib.pyplot as plt
plt.title('Accuracy')
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='validation')
plt.legend()
plt.show()

loss = pd.DataFrame(
    {"train loss": hist.history["loss"], "test loss": hist.history["val_loss"]}
).melt()
loss["epoch"] = loss.groupby("variable").cumcount() + 1
sns.lineplot(x="epoch", y="value", hue="variable", data=loss).set(
    title="Model loss", ylabel=""
)
plt.show()


from sklearn.metrics import log_loss, brier_score_loss, roc_curve, roc_auc_score, confusion_matrix
pred = nn.predict(x1_test_vec)



# checking how liberal biases were predicted
probs = pred[:, 1]
# generating roc curve
fpr, tpr, thresholds = roc_curve(y1_test, probs)

plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr)
plt.show()


#AUC ROC Curve section
from sklearn.metrics import auc
auc_keras = auc(fpr, tpr)

auc_keras

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Area Under Curve (AUC) =  {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for Predicting Liberal Bias')
plt.legend(loc='best')
plt.savefig('auc_final.jpg', dpi = 720)
plt.show()



#converting probabilities to single label prediction
pred_class = (pred>0.5).astype(int)

cm = confusion_matrix(ycattest.argmax(axis = 1), pred_class.argmax(axis = 1))
labels = ['conservative','liberal']
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.savefig('cm_final.jpg', dpi = 720)
plt.show()


#Creating F1 score, Precision and Recall
report = metrics.classification_report(pred_class.argmax(axis = 1),ycattest.argmax(axis = 1), target_names= ['conservative','liberal'])
print(report)


#Appendix: EDA and Supplementary Information

articles['publication'].value_counts().plot(kind='bar', width=0.7)

plt.xlabel("Publications")
plt.ylabel("Count")
plt.title("Number of publications")
plt.savefig("Numberpublications_final.jpg",dpi=720, bbox_inches='tight')
plt.show()



articles['leaning'].value_counts().plot(kind='bar', width=0.7)

plt.xlabel("Leanings")
plt.ylabel("Count")
plt.title("Number of data points")
plt.savefig("Numberlabels_final.jpg",dpi=720, bbox_inches='tight')
plt.show()



year = articles[articles['year'].isin([2015,2016,2017])]
year['year'].astype(int).value_counts().plot(kind='bar', width=0.7)
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Number of Articles Per Year")
plt.savefig("Numberyears_final.jpg",dpi=720, bbox_inches='tight')
plt.show()
