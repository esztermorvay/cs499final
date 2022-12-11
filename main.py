import time

import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split
# from transformers.utils.notebook import format_time
from torch.utils.throughput_benchmark import format_time

import global_vars
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
import torch
from util import flat_accuracy, plot_metrics

# TODO get the new metrics and confusion matrix working
# TODO parse the Telugu dataset and run it
# TODO get the multilingual model to work, and run all the datasets through it

# reading in CSV: clickbait = 1, non-clickbait = 0

def parse_indonesian_df():
    # iterate through the CSVS in the folder. append them all to one DF
    directory_name = global_vars.indonesia_csv_dir
    df = pd.DataFrame()
    for file in os.listdir(directory_name):
        csv = os.path.join(directory_name, file)
        temp = pd.read_csv(csv)
        # df = df.append(temp)
        df = pd.concat([df, temp])
    # print(df)
    df = df.dropna()
    return df

def parse_english_df():
    df = pd.read_csv(global_vars.english_csv_name)
    df.dropna()
    # split df into clickbait and non
    df_clickbait = df[df['clickbait'] == 1]
    df_not = df[df['clickbait'] == 0]
    # print(df_not)
    df_clickbait = df_clickbait.truncate(after=6290-1)
    df_not = df_not.truncate(after=15999+8710-1)
    df_main = pd.concat((df_clickbait, df_not))
    # print(df)
    return df_main

def parse_telugu_df():
    df = pd.read_csv(global_vars.telugu_csv_name)
    df.dropna()
    # split df into clickbait and non
    df_clickbait = df[df['label'] == 1]
    df_not = df[df['label'] == 0]
    # print(df_not)
    df_clickbait = df_clickbait.truncate(after=6290-1)
    df_not = df_not.truncate(after=15999+8710-1)
    df_main = pd.concat((df_clickbait, df_not))
    # print(df)
    return df_main
    df.dropna()
    # split df into clickbait and non
    df_clickbait = df[df['clickbait'] == 1]
    df_not = df[df['clickbait'] == 0]
    # print(df_not)
    df_clickbait = df_clickbait.truncate(after=6290-1)
    df_not = df_not.truncate(after=15999+8710-1)
    df_main = pd.concat((df_clickbait, df_not))
    # print(df)
    return df_main


def tokenize():
    if global_vars.data_using == 0:
        df = parse_indonesian_df()
        title = "title"
        label_ = "label_score"
    elif global_vars.data_using == 1:
        df = parse_english_df()
        title = "headline"
        label_ = "clickbait"
    elif global_vars.data_using == 2:
        df = parse_telugu_df()
        title = "text"
        label_ = "label"

    print(df)
    # print(len(df)) # 15000 datapoints, of which 6290 are clickbait
    # df_clickbait = df[df['label_score'] == 1]
    # print(len(df_clickbait))
    labels = []
    # getting labels
    for label in df[label_]:
        labels.append(label)
    inputs_all = []
    attention_masks_all = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    for sentence in df[title]:
        # print(sentence)
        inputs = tokenizer.encode(sentence, add_special_tokens=True)
        # print(inputs)
        encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=global_vars.sentence_length,
                                             pad_to_max_length=True, truncation=True,
                                             return_attention_mask=True, return_tensors='pt', )
        inputs = []
        attention_masks = []
        # inputs.append(encoded_dict['input_ids'])
        inputs.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])
        inputs = torch.cat(inputs, dim=0)
        inputs_all.append(inputs)
        # inputs_all = torch.cat(inputs_all, dim=0)
        # inputs_all = inputs_all + inputs
        # inputs_all = torch.cat((inputs_all, inputs), dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        attention_masks_all.append(attention_masks)
    labels = torch.tensor(labels)
    inputs_all = torch.cat(inputs_all, dim=0)
    attention_masks_all = torch.cat(attention_masks_all, dim=0)

    return TensorDataset(inputs_all, attention_masks_all, labels)
    # print("done")


def setup_dataloader(dataset):
    num_train = int(global_vars.train_perc * len(dataset))
    num_test = len(dataset) - num_train
    train_set, val_set = random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=global_vars.batch_size)
    val_loader = DataLoader(val_set, sampler=RandomSampler(val_set), batch_size=global_vars.batch_size)
    return train_loader, val_loader

def setup_model():
    # TODO: change the model to be the multilingual one here. i'm just using this from the tutorial to test it for now
    # model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased",
    #     num_labels=2,
    #     output_attentions=False,
    #     output_hidden_states=False,
    # )
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    
    )
    return model

def setup_optimizer(model):
    optimizer = AdamW(model.parameters(),
                      # lr=global_vars.learning_rate,
                      eps=1e-8
                      )
    return optimizer
def train(model, optimizer, device, train_dataloader, val_dataloader):
    train_losses =[]
    train_accs = []
    train_precisions = []
    train_recalls = []
    train_f1s = []
    val_losses = []
    val_accs = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    for epoch in range(0,global_vars.num_epochs):
        t0 = time.time()
        total_train_loss = 0
        total_train_acc = 0
        total_precision = 0
        total_f1 = 0
        total_recall = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels,
                                 return_dict=False)
            # print({'train_batch_loss': loss.item()})
            total_train_loss += loss.item()
            label_ids = b_labels.to('cpu').numpy()
            total_train_acc += flat_accuracy(logits.detach().numpy(), label_ids)
            metrics = classification_report(label_ids, np.argmax(logits.detach().numpy(), axis=1), digits=4, output_dict=True)
            total_precision += metrics["weighted avg"]["precision"]
            total_f1 += metrics["weighted avg"]["f1-score"]
            total_recall += metrics["weighted avg"]["recall"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(train_dataloader)
        avg_recall = total_recall / len(train_dataloader)
        avg_precision = total_precision / len(train_dataloader)
        avg_f1 = total_f1 / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        train_precisions.append(avg_precision)
        train_recalls.append(avg_recall)
        train_f1s.append(avg_f1)
        training_time = format_time(time.time() - t0)
        # Log the Avg. train loss
        print({'avg_train_loss': avg_train_loss})
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Average training accuracy: {0:.2f}".format(avg_train_acc))

        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        total_eval_precision = 0
        total_eval_f1 = 0
        total_eval_recall = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in val_dataloader:
            # b_input_ids = batch[0].cuda()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                (loss, logits) = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels,
                                       return_dict=False)

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_eval_precision += metrics["weighted avg"]["precision"]
            total_eval_f1 += metrics["weighted avg"]["f1-score"]
            total_eval_recall += metrics["weighted avg"]["recall"]
            # get the recall, f1, precision scores
            print(classification_report(label_ids, np.argmax(logits, axis=1), digits=4))

        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(val_dataloader)
        avg_eval_recall = total_recall / len(train_dataloader)
        avg_eval_precision = total_precision / len(train_dataloader)
        avg_eval_f1 = total_f1 / len(train_dataloader)
        val_accs.append(avg_val_accuracy)
        val_losses.append(avg_val_loss)
        val_precisions.append(avg_eval_precision)
        val_recalls.append(avg_eval_recall)
        val_f1s.append(avg_eval_f1)
        validation_time = format_time(time.time() - t0)
        # Log the Avg. validation accuracy
        print(({'val_accuracy': avg_val_accuracy, 'avg_val_loss': avg_val_loss}))
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    train_metrics = [train_losses, train_accs, train_f1s, train_precisions, train_recalls]
    val_metrics = [val_losses, val_accs, val_f1s, val_precisions, val_recalls]
    plot_metrics(train_metrics, val_metrics)

def main():
    dataset = tokenize()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = setup_model()
    model.to(device)
    train_dataloader, validation_dataloader = setup_dataloader(dataset)
    optimizer = setup_optimizer(model)
    train(model, optimizer, device, train_dataloader, validation_dataloader)

if __name__ == "__main__":
    main()
