import os
import pandas

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

trainset_df = pandas.read_csv(os.path.join(DIR_PATH, "jester-v1-train.csv"),
        sep=';')
valset_df = pandas.read_csv(os.path.join(DIR_PATH, "jester-v1-validation.csv"),
        sep=';')
testset_df = pandas.read_csv(os.path.join(DIR_PATH, "jester-v1-test.csv"),
        sep=';')

if __name__ == "__main__":
    print(trainset_df)
    for idx, row in trainset_df.iterrows():
        print(row['video'])