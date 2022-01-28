import csv

"""
THIS FILE IS NOT IN USE!
USE THE convert_data_columns function in 'alter_data.py'!
"""

"""
Converts the 'M' and 'F' values of the sex column in the Student Dataset to a numerical value such that it can be
run on the baseline of Bera et al., and stores the result in a 'modded' .csv-file
"""
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