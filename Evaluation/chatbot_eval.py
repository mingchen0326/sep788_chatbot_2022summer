import pandas as pd
import numpy as np
import string
import spacy

# import data
vail_df = pd.read_csv(r"Evaluation\SampleOutput.csv")

# Extract standard answer and predictive answer
answer = vail_df['Answer']
prediction = vail_df['Predictive Answer']


# Define function to calculate accuracy
def eval(answer, predict):
    nlp = spacy.load("en_core_web_lg")
    accuracy = []
    for row in answer.index:
        ans = nlp(answer[row])
        pred = nlp(predict[row])
        accuracy.append(ans.similarity(pred))
    return accuracy


accuracy = eval(answer, prediction)
print(accuracy)
print(np.mean(accuracy))




