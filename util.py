import numpy as np
import matplotlib.pyplot as plt

import global_vars


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# lists of values, save_dir should correspond to a context window size
def plot_metrics(train_loss, train_acc, val_loss, val_acc):
    if global_vars.data_using == 0:
        language_name = "Indonesian"
    elif global_vars.data_using == 1:
        language_name = "English"
    # plot train_loss and val_loss
    plt.figure()
    plt.title("Train And Val Loss for " + language_name + " Clickbait Detection" )
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()

    save_name_pref = language_name+str(global_vars.learning_rate)

    save_name = save_name_pref + "loss.png"
    plt.savefig("graphs/" + save_name)
    plt.clf()
    # plot train_acc and val_acc
    plt.figure()
    plt.title("Train And Val Accuracy for " + language_name + " Clickbait Detection" )
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="val")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend()
    save_name = save_name_pref + "accuracy.png"
    plt.savefig("graphs/" + save_name)
    plt.clf()