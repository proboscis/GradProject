__author__ = 'kentomasui'

from util import fileLines
def purchaseData():
    return (line.split(",") for line in fileLines("purchase.csv"))

def bookInfo():
    return (line.split(",") for line in fileLines("boookinfo.csv"))
