"""GLOBAL VARIABLES"""

sentence_length = 64 # sentence length in words
indonesia_csv_dir = "./indonesia_csvs"
train_perc = 0.7
batch_size = 64
learning_rate = 0.1
num_epochs = 5
data_using = 2 # 0 for indonesian, 1 for english, 2 for telugu
english_csv_name = "train1.csv"
telugu_csv_name = "clickbait_train.csv"
model_name = 'bert-base-multilingual-uncased'
# model_name  = 'bert-base-uncased'
metric_names = ["loss", "accuracy", "f1", "precision", "recall"]