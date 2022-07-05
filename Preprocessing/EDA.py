# -*- coding: UTF-8 -*-
# Fei
# Exploratory Data Analysis (EDC)

# require
import numpy
import pandas as pd
from pathlib import Path
from os import listdir

# load files
raw_path = "Raw_data"
dirs = listdir(raw_path)

# main class
class Eda:
    def __init__(self, df, name) -> None:
        self.d1 = df
        self.name = name
        pass

    def eda(self):
        # dataset info
        self.d1.info()

    def dataclean(self):
        # local var
        d1 = self.d1

        # dataset info
        ori_rows = len(d1)

        # clean null
        d1 = d1.dropna()
        d1.info()
        n_droped = ori_rows - len(d1)
        print("%d rows has been droped" % (n_droped))

        # save dataset
        file_name = "%s.cbcsv" % (self.name)
        file_path = Path("Dataset\%s" % (file_name))
        try:
            d1.to_csv(file_path, index=False)
        except:
            print("data save failed")
        else:
            print("data save succeed")


# main function
for f in dirs:
    if ".txt" in f:
        d1 = pd.read_csv("%s\%s" % (raw_path, f), delimiter="\t",
                         error_bad_lines=False, encoding_errors="surrogateescape")
        task = Eda(d1, f)
        task.eda()
        task.dataclean()

pass
