import csv

file = csv.reader(open('data/Synthetic/synthetic.csv'))
lines = list(file)
new_lines = [['A', 'B']]

for line in lines[1:]:
    new_lines.append(line[1:])

print(new_lines)
writer = csv.writer(open('data/Synthetic/synthetic_mod.csv', 'w', newline=''))
writer.writerows(new_lines)