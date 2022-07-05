# -*- coding: UTF-8 -*-
# Fei
# Exploratory Data Analysis (EDC)

# require
import numpy
import pandas as pd
from pathlib import Path
from os import listdir
import codecs
import unicodedata
import re

# main class
class Eda:
    def __init__(self, raw_path="Raw_data") -> None:
        self._dirs = listdir(raw_path)
        self._raw_path = raw_path
        self.df = self.fixdata()

    def fixdata(self):
        df_list = []
        for f in self._dirs:
            if "pairs" in f:
                try:
                    lines = open('%s\%s' % (self._raw_path, f),
                                 encoding='utf-8').read().strip().split('\n')
                except:
                    lines = open('%s\%s' % (self._raw_path, f),
                                 encoding='ISO-8859-1').read().strip().split('\n')
                col_names = self.normalizeString(lines[0]).split("\t")
                col_len = len(col_names)
                df = pd.DataFrame()
                lines = lines[1:]
                for line in lines:
                    new_line = self.normalizeString(line)
                    line_split = new_line.split("\t")
                    if len(line_split) != col_len:
                        continue
                    new_line_df = pd.DataFrame([line_split], columns=col_names)
                    df = pd.concat([df, new_line_df])
                df_list.append(df)
        combined_df = pd.concat(df_list)
        return combined_df

    def info(self):
        # dataset info
        self.df.info()

    def dataclean(self):
        # local var
        d1 = self.df

        # dataset info
        ori_rows = len(d1)

        # clean null
        d1 = d1.dropna()
        n_droped = ori_rows - len(d1)
        print("%d rows has been droped" % (n_droped))

        # save dataset
        file_name = "cleaned.cbcsv"
        file_path = Path("Dataset\%s" % (file_name))
        try:
            d1.to_csv(file_path, index=False)
        except Exception as e:
            print("data save failed duo to: ", e)
        else:
            print("data save succeed")

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)           # Split .!? with words
        # Remove useless characters
        s = re.sub(r"[^a-zA-Z.!?\t]+", r" ", s)
        if s[0] == " ":
            s = s[1:]
        return s

# main function entry
def run():
    task = Eda()
    task.info()
    task.dataclean()

# OOP
if __name__ == '__main__':
    run()

pass
