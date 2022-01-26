import csv

file = csv.reader(open('data/German_Credit/german_credit_data.csv'))
lines = list(file)
# new_lines = [['A', 'B']]
new_lines = [lines[0]]

for line in lines[1:]:
    if line[2] == 'male':
        line[2] = 0
    elif line[2] == 'female':
        line[2] = 1
    else:
        print('UNKOWN')
        continue

    new_lines.append(line[1:])

print(new_lines)
writer = csv.writer(open('data/German_Credit/german_Credit_mod.csv', 'w', newline=''))
writer.writerows(new_lines)