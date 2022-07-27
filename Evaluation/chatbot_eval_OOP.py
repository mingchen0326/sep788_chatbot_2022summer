import pandas as pd
import numpy as np
import string
import spacy
import os


# Define function to calculate accuracy
def eval(answer, predict):
    nlp = spacy.load("en_core_web_lg")
    result = []
    for row in answer.index:
        ans = nlp(answer[row])
        pred = nlp(predict[row])
        result.append(ans.similarity(pred))
    return result

def eval_dir(path):
    files = os.listdir(path)
    files.remove('.gitkeep')
    for f in files:
        vail_df = pd.read_csv(path + '/' + f)
        answer = vail_df['Answer']
        prediction = vail_df['Predictive Answer']
        accuracy = eval(answer, prediction)
        print(f)
        # print(accuracy)
        print(np.mean(accuracy))

def eval_file(f):
    vail_df = pd.read_csv(f)
    answer = vail_df['Answer']
    prediction = vail_df['Predictive Answer']
    accuracy = eval(answer, prediction)
    print(f)
    # print(accuracy)
    print(np.mean(accuracy))

eval_dir("Results")



