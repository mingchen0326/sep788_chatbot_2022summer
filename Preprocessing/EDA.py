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
                lines = open('%s\%s' % (self._raw_path,f), encoding='utf-8').read().strip().split('\n')
                col_names = self.normalizeString(lines[0]).split("\t")
                df = pd.DataFrame()
                lines = lines[1:]
                for line in lines:
                    new_line = self.normalizeString(line)
                    line_split = new_line.split("\t")
                    new_line_df = pd.DataFrame([line_split], columns=col_names)
                    df = pd.concat(df, new_line_df)
                    

    def info(self):
        # dataset info
        self.d1.info()

    def dataclean(self):
        # local var
        d1 = self.d1

        # dataset info
        ori_rows = len(d1)

        # clean null
        d1 = d1.dropna()
        n_droped = ori_rows - len(d1)
        print("{n_droped} rows has been droped")

        # save dataset
        file_name = "cleaned.cbcsv"
        file_path = Path("Dataset\{filename}")
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
        s = re.sub(r"[^a-zA-Z.!?\t]+", r" ", s)       # Remove useless characters
        if s[0] == " ":
            s = s[1:]
        return s

# main function
def run():
    # # load files
    # raw_path = "Raw_data"
    # dirs = listdir(raw_path)

    # # iter txt files
    # df_list = []

    # for f in dirs:
    #     if ".txt" in f:
    #         df = pd.read_csv("%s\%s" % (raw_path, f), delimiter="\t",
    #                          on_bad_lines="skip", encoding="utf-8-sig")
    #         df_list.append(df)

    # # concat dataframes
    # d1 = pd.concat(df_list)

    # start task
    task = Eda()
    task.eda()
    task.dataclean()




run()

pass
