import os
import csv

file = csv.reader(open('data/Student/student_mat_Cortez.csv'))
lines = list(file)

for line in lines:
    sex = line[1]

    if sex == 'M':
        line[1] = 0
    elif sex == 'F':
        line[1] = 1
    else:
        print("NO SEX", line[1])

writer = csv.writer(open('data/Student/student_mat_Cortez_sexmod.csv', 'w', newline=''))
writer.writerows(lines)