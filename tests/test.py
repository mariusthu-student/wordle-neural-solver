mot = 'aaaaa'
colors = [1, 2, 1, 1, 1]

for i in range(5):
    print([int(l == mot[i]) for l in 'abcdefghijklmnopqrstuvwxyz'] + [int(k == colors[i]) for k in range(3)])
