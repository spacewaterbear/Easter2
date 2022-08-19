import sys

import cv2
import pandas as pd

import config
import tensorflow
import tensorflow as tf
import itertools
import numpy as np
from editdistance import eval as edit_distance
from tqdm import tqdm
from data_loader import data_loader
import tensorflow.keras.backend as K

from src.easter_model import Easter2


def ctc_custom(args):
    y_pred, labels, input_length, label_length = args

    ctc_loss = K.ctc_batch_cost(
        labels,
        y_pred,
        input_length,
        label_length
    )
    p = tensorflow.exp(-ctc_loss)
    gamma = 0.5
    alpha = 0.25
    return alpha * (K.pow((1 - p), gamma)) * ctc_loss


def load_easter_model(checkpoint_path):
    if checkpoint_path == "Empty":
        checkpoint_path = config.BEST_MODEL_PATH
    try:
        checkpoint = Easter2()
        checkpoint.load_weights(checkpoint_path)
        # checkpoint = tensorflow.keras.models.load_model(
        #     checkpoint_path,
        #     custom_objects={'<lambda>': lambda x, y: y,
        #                     'tf': tf}
        # )

        EASTER = tensorflow.keras.models.Model(
            checkpoint.get_layer('the_input').input,
            checkpoint.get_layer('Final').output
        )
    except Exception as e:
        print("Unable to Load Checkpoint.")
        print(e)
        return None
    return EASTER


def decoder(output, letters):
    ret = []
    for j in range(output.shape[0]):
        out_best = list(np.argmax(output[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

def predict_one_image(img, model, charlist):
    img = read_and_process_image(img)
    output = model.predict(img)
    prediction = decoder(output, charlist)
    output = (prediction[0].strip(" ").replace("  ", " "))
    return output

def preprocess(img, augment=True):
    if augment:
        img = self.apply_taco_augmentations(img)

    # scaling image [0, 1]
    img = img / 255
    img = img.swapaxes(-2, -1)[..., ::-1]
    target = np.ones((config.INPUT_WIDTH, config.INPUT_HEIGHT))
    new_x = config.INPUT_WIDTH / img.shape[0]
    new_y = config.INPUT_HEIGHT / img.shape[1]
    min_xy = min(new_x, new_y)
    new_x = int(img.shape[0] * min_xy)
    new_y = int(img.shape[1] * min_xy)
    img2 = cv2.resize(img, (new_y, new_x))
    target[:new_x, :new_y] = img2
    return 1 - (target)

def read_and_process_image(img_path: str):
    img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img1, augment=False)
    img = np.expand_dims(img, 0)
    return img

def test_on_image(img_path: str):
    with open("charlist.txt", "r") as f:
        charlist = f.read().splitlines()
    model = load_easter_model(config.BEST_MODEL_PATH)
    prediction = predict_one_image(img_path, model, charlist)
    return prediction

def test_on_iam(show=True, partition='test', uncased=False, checkpoint="Empty"):

    print("loading metdata...")
    training_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    validation_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    test_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)

    training_data.trainSet()
    validation_data.validationSet()
    test_data.testSet()
    charlist = training_data.charList
    charlist_str = '\n'.join(charlist)
    with open("charlist.txt", "w") as f:
        f.write(charlist_str)

    with open("charlist.txt", "r") as f:
        charlist = f.read().splitlines()

    print("loading checkpoint...")
    print("calculating results...")
    model = load_easter_model(checkpoint)
    char_error = 0
    total_chars = 0
    data = []
    batches = 1
    while batches > 0:
        batches = batches - 1
        if partition == 'validation':
            print("Using Validation Partition")
            imgs, truths, _ = validation_data.getValidationImage()
        else:
            print("Using Test Partition")
            imgs, truths, _ = test_data.getTestImage()

        print("Number of Samples : ", len(imgs))
        for i in tqdm(range(0, len(imgs))):
            img = imgs[i]
            truth = truths[i].strip(" ").replace("  ", " ")
            output = model.predict(img)
            prediction = decoder(output, charlist)
            output = (prediction[0].strip(" ").replace("  ", " "))
            if uncased:
                char_error += edit_distance(output.lower(), truth.lower())
            else:
                char_error += edit_distance(output, truth)

            total_chars += len(truth)
            if show:

                error = edit_distance(output, truth)
                print("Ground Truth :", truth)
                print("Prediction [", error, "]  : ", output)
                print("*" * 50)
                data.append({"ground_truth": truth, "prediction": output, "error": error})

    print("Character error rate is : ", (char_error / total_chars) * 100)
    if show:
        df = pd.DataFrame(data)
        df.to_csv('results.csv', index=False)