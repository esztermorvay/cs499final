import numpy as np
import matplotlib.pyplot as plt

import global_vars


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# lists of values, save_dir should correspond to a context window size
# train metrics is a list of training metrics: loss, accuracy, f1, precision, recall
def plot_metrics(train_metrics, val_metrics):
    if global_vars.data_using == 0:
        language_name = "Indonesian"
    elif global_vars.data_using == 1:
        language_name = "English"
    save_name_pref = language_name+str(global_vars.learning_rate)

    for i in range(0,len(train_metrics)):
        plt.plot(train_metrics[i], label="train")
        plt.plot(val_metrics[i], label="validation")
        plt.title("Training and Validation "+global_vars.metric_names[i] + " for " + language_name + " Clickbait Detection")
        plt.xlabel("Epoch")
        plt.ylabel(global_vars.metric_names[i])
        plt.legend()
        save_name = save_name_pref+global_vars.metric_names[i]+".png"
        plt.savefig(save_name)
        plt.clf()
        plt.cla()
        plt.close()
    # # plot train_loss and val_loss
    # plt.figure()
    # plt.title("Train And Val Loss for " + language_name + " Clickbait Detection" )
    # plt.plot(train_loss, label="train")
    # plt.plot(val_loss, label="val")
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.legend()


    # # save_name = "multilingual" + save_name_pref + "loss.png"
    # plt.savefig("graphs/" + save_name)
    # plt.clf()
    # # plot train_acc and val_acc
    # plt.figure()
    # plt.title("Train And Val Accuracy for " + language_name + " Clickbait Detection" )
    # plt.plot(train_acc, label="train")
    # plt.plot(val_acc, label="val")
    # plt.ylabel("accuracy")
    # plt.xlabel("epoch")
    # plt.legend()
    # save_name = save_name_pref + "accuracy.png"
    # plt.savefig("graphs/" + save_name)
    # plt.clf()