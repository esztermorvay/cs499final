# beforehand
# huggingface
# pip install transformers
# SastrawiStemmer/StopwordsRemover
# pip install PySastrawi
# pip install -q pyyaml h5py

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import Counter
from numpy import interp
import numpy as np

#from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from transformers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tqdm
import time
#from torch.utils.data import TensorDataset, random_split

from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM, Bidirectional, Input, Dropout, GlobalAveragePooling1D, Flatten, Conv1D, GlobalMaxPooling1D
#from tensorflow.keras import Sequential
from tensorflow.keras.models import Model

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.preprocessing import sequence

import torch

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


#LOAD DATASET
# FIXME we can change this if we aren't loading a csv from github
# FIXME but also which data set do we want to use?
#training=pd.read_csv('https://raw.githubusercontent.com/ruzcmc/ClickbaitIndo-textclassifier/master/primary-dataset.csv')
#training=pd.read_csv('https://raw.githubusercontent.com/ruzcmc/ClickbaitIndo-textclassifier/master/clickid-main.csv')
trainingagree=pd.read_csv('https://raw.githubusercontent.com/ruzcmc/ClickbaitIndo-textclassifier/master/all_agree.csv')
pd.options.display.max_colwidth=2000
# label_cols = ['text','label'] FIXME I'm not sure if this is supposed to be commented or not


# FIXME wait is this all on training?
# UNDERSAMPLE IF NEEDED
clickbait = trainingagree.loc[trainingagree['label_score'] == 1]
# print(len(clickbait))
nclickbait = trainingagree.loc[trainingagree['label_score'] == 0].sample(n=3316,random_state=22)
nclickbaitasli = trainingagree.loc[trainingagree['label_score'] == 0]
# print(len(nclickbaitasli))

# FIXME I think this was left over from when they used the other training dataset, hopefully we won't need it
# Put all the fraud class in a separate dataset. ini yang kappa 0.4
#clickbait_df = training.loc[training['label_score'] == 1]
#print(len(clickbait_df))
#nclickbait_df = training.loc[training['label_score'] == 0].sample(n=6290,random_state=13)
#print(len(nclickbait_df))
#Randomly select 492 observations from the non-fraud (majority class)
#non_clickbait_df = training.loc[training['label'] == 0].sample(n=1393,random_state=42)

# Concatenate both dataframes again
normalized_df = pd.concat([clickbait, nclickbait])

# This was to make sure there was roughly the same amount of clickbait as nonclickbait
#plot the dataset after the undersampling
# plt.figure(figsize=(8, 8))
# sns.countplot('label', data=normalized_df)
# plt.title('Balanced Classes')
# plt.show()

#print(normalized_df)
# FIXME are these needed?
tags = normalized_df.label_score
texts = normalized_df.title
#print(tags)


# PREPROCESSING
#stemming INDO - no need with BERT TOkenizeR
#factory = StemmerFactory()
#stemmer= factory.create_stemmer()

#textstemss = []
textstem = []
textstemclick = []
textstemnon =[]
#for line in texts:
#  outstem = stemmer.stem(line)
#  textstemss.append(outstem)

#stopword remove
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
for line in clickbait['title']:
  stop = stopword.remove(line)
  textstemclick.append(stop)
for line in nclickbait['title']:
  stopa = stopword.remove(line)
  textstemnon.append(stopa)


for line in texts:
  stop = stopword.remove(line)
  textstem.append(stop)

# Fixme, this is for if we want to remove digits from titles
#from string import digits
#for line in textstems:
#    remove_digits=str.maketrans(' ', ' ', digits)
#    result=line.translate(remove_digits)
#    textstem.append(result)


#print(textstem)
#print(texts)

X_train, X_test, y_train, y_test = train_test_split(textstem, tags, test_size=0.1, random_state=42)
# FIXME here is a hyperparam
#print(len(X_train))


# GRAPH OF MOST COMMON WORDS
# #descriptive
# # Create a list of words by converting to lowercase and splitting
# kata = [s.lower().split() for s in textstemclick if s]
# noline_ = [sublist for l in kata for sublist in l]
# counts1 = dict(Counter(noline_).most_common(10))
# labels1, values1 = zip(*counts1.items())
# # sort your values in descending order
# indSort1 = np.argsort(values1)[::-1]
# # rearrange your data
# labels1 = np.array(labels1)[indSort1]
# values1 = np.array(values1)[indSort1]
# indexes1 = np.arange(len(labels1))
# bar_width = 1
# mybar=plt.bar(indexes1, values1)
# # get rid of the frame
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
# # remove all the ticks and directly label each bar with respective value
# plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
# # plt.show()
# # direct label each bar with Y axis values
# for bari in mybar:
#     height = bari.get_height()
#     plt.gca().text(bari.get_x() + bari.get_width()/2, bari.get_height()-0.2, str(int(height)),
#                  ha='center', color='black', fontsize=15)
# # add labels
# plt.xticks(indexes1, labels1)
# #plt.savefig('/content/drive/My Drive/clickbait-textclassifier/top10words-NONclickbait.png')
# plt.show()



# ENCODING
# TF x huggingface transformers
MAX_LEN = 22  # FIXME hyperparam
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-lite-base-p1')  # FIXME wait do we wanna do pretrained?
# FIXME the above line needs to be changed depending on the language though

def encodetext(sentence):
    input_ids = []
    attention_masks = []
    for sent in sentence:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,  # '[CLS]' and '[SEP]'
            max_length=MAX_LEN,
            pad_to_max_length=True,  # Pad / truncate
            return_attention_mask=True,  # Construct attn. masks.

            return_token_type_ids=False
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks



#tags = np.array(tags)
input_ids, attention_masks = np.array(encodetext(textstem))
# print('Original: ', textstem[-5])
# print('Token IDs:', input_ids[-5])
#print(normalized_df)
# print(tags)


# model definition
# custom model #keras Functional model
def createmodel():
    token_inputs = Input((MAX_LEN,), dtype=tf.int32, name='input_word_ids')
    mask_inputs = Input((MAX_LEN,), dtype=tf.int32, name='input_masks')

    bert_model = TFBertModel.from_pretrained("indobenchmark/indobert-lite-base-p1")  # FIXME change depending
    seq_output = bert_model.bert(input_ids=token_inputs, attention_mask=mask_inputs)[0]

    X = GlobalAveragePooling1D()(seq_output)
    X = Flatten()(X)
    X = Dense(100, activation='relu')(X)  # fixme hyperparam
    output_ = Dense(1, activation='sigmoid', name='output')(X)

    bert_model2 = Model([token_inputs, mask_inputs], output_)
    return bert_model2


# HF model for Classification
def createhugmodel():
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
    return bert_model


# RUNNING THE MODEL
# Kfold CrossVal
from sklearn.metrics import classification_report

n_folds = 5  # FIXME something we can change
kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=13)
kfold = kfold.split(input_ids, y=tags)
cvscores = []
tprs = []
aucs = []

mean_fpr = np.linspace(0, 1, 100)

i = 0
opt = keras.optimizers.Adam(learning_rate=1e-05)  # FIXME also something we canh change
for i, (train, test) in enumerate(kfold):
    traindata = [input_ids[train], attention_masks[train]]
    testdata = [input_ids[test], attention_masks[test]]
    ytrain = tags.iloc[train]
    ytest = tags.iloc[test]

    print("Running Fold", i + 1, "/", n_folds)

    bert_model2 = createmodel()

    bert_model2.compile(optimizer=opt,
                        loss="binary_crossentropy",
                        metrics=['accuracy'])
    bert_model2.fit(traindata, ytrain, epochs=4, verbose=1, batch_size=16)
    logits = bert_model2.predict(testdata)
    # print(logits) #kalo pake bert layer, ini ga usah tuple langsung aja
    # "if use BERT layer, this doesn't need to be tuple, just use as is"
    predicts = (logits > 0.5).astype("int32")
    scores = bert_model2.evaluate(testdata, ytest, verbose=0)

    cm = confusion_matrix(ytest, predicts)
    creport = classification_report(ytest, predicts)
    print('Confusion matrix')
    print(cm)
    print(creport)
    print("%s: %.2f%%" % (bert_model2.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(ytest, predicts)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    print('AUC')
    print(roc_auc)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    keras.backend.clear_session()

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.savefig('/content/drive/My Drive/clickbait-textclassifier/rocauc-BERT.png')
plt.show()

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

bert_model2.summary()

# print(seq_output)
# model.summary()  # FIXME they commented this out but we may want it?
# Tell pytorch to run this model on the GPU. i gave up on pytorch this time. sorry
# model.cuda()






#save Model for Production ^^
xtraining = [input_ids,attention_masks]
ytraining = tags
opt = keras.optimizers.Adam(learning_rate=1e-05)
bert_model2 = createmodel()

bert_model2.compile(optimizer=opt,
              loss="binary_crossentropy",
              metrics=['accuracy'])

bert_model2.fit(xtraining, ytraining, epochs=4, verbose=1,batch_size=16)


# FIXME these are just floating around, not sure where they should go?
bert_model2.save("Model.h5", save_format="h5")
keras.utils.plot_model(bert_model2)


# MANUAL TESTING -- FIXME we may not need this either
manualtest = pd.read_csv('https://raw.githubusercontent.com/ruzcmc/ClickbaitIndo-textclassifier/master/primary-dataset.csv')

#print(manualtest)
#manualtest.head()
manualtests = manualtest.text
tagtest = manualtest.label
#manualtests

hublah = []
for line in manualtests:
  stop = stopword.remove(line)
  hublah.append(stop)

input_ids2, attention_masks2 = np.array(encodetext(hublah))
# print('Original: ', manualtests[0])
# print('Token IDs:', input_ids2[0])
# print('Attn mask:', attention_masks2[0])
testdatamanual = [input_ids2, attention_masks2]

xtraining = [input_ids,attention_masks]
ytraining = tags
opt = keras.optimizers.Adam(learning_rate=1e-05)
bert_test = createmodel()

bert_test.compile(optimizer=opt,
              loss="binary_crossentropy",
              metrics=['accuracy'])
bert_test.fit(xtraining, ytraining, epochs=4,verbose=1,batch_size=16)

from sklearn.metrics import classification_report

#eval additional test
hasil = bert_model2.predict(testdatamanual)
#print(logits) #kalo pake bert layer, ini ga usah tuple langsung aja
pred=(hasil > 0.5).astype("int32")
scores = bert_model2.evaluate(testdatamanual, tagtest, verbose=0)

cm = confusion_matrix(tagtest, pred)
creport = classification_report(tagtest, pred)
print('Confusion matrix')
print(cm)
print(creport)
print("%s: %.2f%%" % (bert_model2.metrics_names[1], scores[1]*100))
#cvscores.append(scores[1] * 100)
    # Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(tagtest, pred)
tprs.append(interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
print('AUC')
print(roc_auc)
aucs.append(roc_auc)
plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC Additional Eval. (AUC = {roc_auc})')


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

#mean_tpr = np.mean(tprs, axis=0)
#mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
#std_auc = np.std(aucs)
#plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

#std_tpr = np.std(tprs, axis=0)
#tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#plt.savefig('/content/drive/My Drive/clickbait-textclassifier/rocauc-BERT.png')
plt.show()


bert_test.save("model_name",save_format='tf')



# FIXME I have no idea what this is?
#COBA XGBOOST
def createxg():
  clf = xgb.XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7, verbosity=1, n_estimators=250)
  return clf

# tfidf vector

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import cross_val_score
cv = CountVectorizer()

# this steps generates word counts for the words in your docs
wcvec = cv.fit_transform(textstem)
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(wcvec)

countvec = cv.transform(textstem)
tf_idf_vector = tfidf_transformer.transform(countvec)

# print(tf_idf_vector)

xbost = createxg()
# xbost.fit(tf_idf_vector,tags)


scores = cross_val_score(xbost, tf_idf_vector, tags, cv=10, verbose=1)
print('Accuracy for Tf-Idf & XGBoost Classifier : ', scores.mean())

#berttest.save('/content/drive/My Drive/clickbait-textclassifier/BERT-IndoClickbaitClassifierMODEL.h5')

tf.keras.models.save_model(
    bert_model2,
    "/content/model",
    overwrite=False,
    include_optimizer=True
)

model2 = tf.keras.models.load_model("/content/drive/My Drive/clickbait-textclassifier/BERT-IndoClickbaitClassifier


