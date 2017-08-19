import sys
import os

import random

#Setting paths
pathFileToBeShuffled = "inputVertebra.csv"
pathFileShuffled = "shuffledInputVertebra.csv"

#Read file into line buffer
fid = open(pathFileToBeShuffled, "r")
print("Input file successfully opened for reading")
li = fid.readlines()
fid.close()
print("Input file successfully closed")

#Invoke shuffle() from the random module
random.shuffle(li)

#Write output to new file
fid = open(pathFileShuffled, "w")
print("Output file successfully opened for writing")
fid.writelines(li)
fid.close()
print("Output file successfully closed")
