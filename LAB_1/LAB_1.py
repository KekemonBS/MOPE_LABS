import numpy as np

limit = 20
rows = 8
columns = 3
a0, a1, a2, a3 = 2, 3, 4, 5
#while True:
matrix = np.random.randint(1,limit+1,(rows,columns))                                       

x0 = []
dx = []
for i in range(columns):
	column = matrix[:,i]
	col_min = min(column)
	col_max = max(column)
	col_x0 = (col_max+col_min)/2.
	col_dx = col_x0-col_min
	x0.append(col_x0)
	dx.append(col_dx)
matrix = np.append(matrix, [x0], axis=0)
matrix = np.append(matrix, [dx], axis=0)

Y = [[a0+a1*matrix[X][0]+a1*matrix[X][1]+a1*matrix[X][2]] for X in range(len(matrix))][:-1]
Y.extend([[0]])	
matrix = np.append(matrix, Y , axis=1)

for i in range(columns):
	column = list(matrix[:,i])
	norm_column = list(map(lambda i : [(i-column[-2])/column[-1]], column))[:-2]
	norm_column.extend([[0],[0]])
	matrix = np.append(matrix, norm_column , axis=1)
print("    Матриця планування\n")
print("  x1    x2     x3     y      x1n    x2n    x3n  "   )
for i in range(matrix.shape[0]):
	for j in range(matrix.shape[1]):
		print("{:^5.1f}".format(matrix[i][j]), end = "  ")
	print("\t")
print("Критерій оптимальності : ->Ует")
Yet = list(matrix[:,columns])[-2]

Y = 0
for i in list(matrix[:,columns]):
	if (i < Yet) and (i > Y):
		Y=i
#		print(Y)
print(str(Y)+" -> "+str(Yet))

chosen_row = list(matrix[list(matrix[:,columns]).index(Y),:])[0:columns+1]
print(str(chosen_row)+"\n")
print("{Y} = a0 + a1*{} + a2*{} + a3*{}".format(chosen_row[0],chosen_row[1],chosen_row[2],Y=chosen_row[3]))
