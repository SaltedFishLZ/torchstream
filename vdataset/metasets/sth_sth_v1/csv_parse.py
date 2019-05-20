"""
CSV File Parsing Module
"""
import os
import pandas

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TRAINSET_DF = pandas.read_csv(
    os.path.join(DIR_PATH, "something-something-v1-train.csv"), sep=';')
VALSET_DF = pandas.read_csv(
    os.path.join(DIR_PATH, "something-something-v1-validation.csv"), sep=';')
TESTSET_DF = pandas.read_csv(
    os.path.join(DIR_PATH, "something-something-v1-test.csv"), sep=';')

## Self Test Function
#  
#  Details
def test():
    """
    Self-test function
    """
    print("[Training Set Dataframe]")
    print(TRAINSET_DF)
    print("[Validation Set Dataframe]")
    print(VALSET_DF)
    print("[Testing Set Dataframe]")
