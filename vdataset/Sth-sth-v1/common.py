import os
import pandas

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

trainset_df = pandas.read_csv(
        os.path.join(DIR_PATH, "something-something-v1-train.csv"), sep=';')
valset_df = pandas.read_csv(
        os.path.join(DIR_PATH, "something-something-v1-validation.csv"), 
        sep=';')
testset_df = pandas.read_csv(os.path.join(
        DIR_PATH, "something-something-v1-test.csv"), sep=';')

if __name__ == "__main__":
    print("[Training Set Dataframe]")
    print(trainset_df)
    print("[Validation Set Dataframe]")
    print(valset_df)
    print("[Testing Set Dataframe]")
    print(testset_df)