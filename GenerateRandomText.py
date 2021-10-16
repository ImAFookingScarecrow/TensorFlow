import random

numbers = 100
digits = 20

open('randomtext.txt', 'w').close() # clears the file

filee = open("randomtext.txt", "a")

for i in range(0, numbers):
    string = ''
    for i in range(0, digits):
        string = string + str(random.randint(0, 1))
    filee.write(string + '\n')

filee.close()