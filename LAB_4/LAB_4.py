from prettytable import PrettyTable as prtt
from scipy.stats import f, t 
import numpy as np
import functools as ft
import itertools as itr


# ~ Дано----------------------------
x1_min = 10
x1_max = 40

x2_min = 25
x2_max = 45

x3_min = 40
x3_max = 45

x_max_list = [x1_max, x2_max, x3_max]
x_min_list = [x1_min, x2_min, x3_min]

k = len(x_max_list)
m = 3
p = 0.95

y_max = 200 + sum(x_max_list)/len(x_max_list)
y_min = 200 + sum(x_min_list)/len(x_min_list)
# ~ --------------------------------  

def fisher_critical(prob, f1, f2):
    return f.ppf(prob, f1, f2)

def student_critical(prob, df):
    return t.ppf(prob, df)

def cochran_critical(q, f1, f2):
    return 1 / (1 + (f2 - 1) / f.ppf(1 - q/f2, f1, (f2 - 1)*f1) )

def print_result(x_max_list, k, x_matr, y_matr, y_avg, y_disp, beta_list):
	adequate_table  = prtt()
	group = ["x"+str(n+1) for n in range(len(x_max_list))]
	group_quantity = k+1
	header = []
	for j in range(group_quantity):
		header.extend(list(itr.combinations(group,j+1)))
	for i in range(len(y_matr[0])):
		header.append('y'+str(i+1))
	header.append('y_avg')
	header.append('s2{y}') # ~ variance or disp
	# ~ add squares
	# ~ print(len(norm_matrix),len(y_matr),len(y_avg),len(y_disp))
	final_matrix = np.hstack((x_matr,y_matr,[[round(avg,2)] for avg in y_avg],[[round(disp,2)] for disp in y_disp]))
	header = list([ft.reduce(lambda x,y : x+y, i) for i in header])
	adequate_table.field_names = ['x0'] + header
	for row_number in range(len(final_matrix)):
		adequate_table.add_row(final_matrix[row_number])
	print(adequate_table)
	# ~ form = 
	# ~ print(form.format(*beta_list))

# ~ Normalized matrix
# ~ factors
items = [(-1,1) for i in range(k)]
f_matrix = list(itr.product(*items))
x0_vector = [[1] for i in range(len(f_matrix))]
# ~ add zor t

# ~ print(np.matrix(f_matrix))
# ~ with interractions---------------------------		
interractions_matrix1 = []
group_quantity = k - 1
for i in f_matrix:
	comb_list = []
	for j in range(group_quantity):
		comb_list.extend(list(itr.combinations(i,j+2)))
	comb_values = [np.prod(k) for k in comb_list]
	interractions_matrix1.append(comb_values)
# ~ ---------------------------------------------
	# ~ print(comb_list)
# ~ print(np.matrix(interractions_matrix1))
norm_matrix = np.hstack((x0_vector, f_matrix, interractions_matrix1))
# ~ print(norm_matrix)

# ~ Naturalized matrix
col_list = []
for i in range(len(f_matrix[0])):
	col = [row[i] for row in f_matrix]
	col = list(map(lambda x : x_max_list[i] if x==1 else x_min_list[i], col))
	col_list.append(col)
nf_matrix = list(zip(*col_list)) # ~ naturalized factors
# ~ add zor t

x0_vector = [[1] for i in range(len(nf_matrix))]
# ~ print(np.matrix(nf_matrix))
# ~ with interractions---------------------------	
interractions_matrix2 = []
group_quantity = k - 1
for i in nf_matrix:
	comb_list = []
	for j in range(group_quantity):
		comb_list.extend(list(itr.combinations(i,j+2)))
	comb_values = [np.prod(k) for k in comb_list]
	interractions_matrix2.append(comb_values)
# ~ ---------------------------------------------

	# ~ print(comb_list)
# ~ print(np.matrix(interractions_matrix2))
natur_matrix = np.hstack((x0_vector, nf_matrix, interractions_matrix2))
# ~ print(natur_matrix)


escape = False
while escape == False:
	y_matr = np.random.randint(y_min, y_max,(len(natur_matrix),m))
	y_avg = [sum(i)/len(i) for i in y_matr] 
	y_disp = [] # ~ i.e. variance
	for i in range(len(natur_matrix)):
		tmp_disp = 0
		for j in range(m):
			tmp_disp += ((y_matr[i][j] - y_avg[i]) ** 2) / m
		y_disp.append(tmp_disp)
	f1 = m - 1
	f2 = len(natur_matrix)
	q = 1 - p
# ~ Cochran test
	Gp = max(y_disp) / sum(y_disp)
	Gt = cochran_critical(q, f1, f2)
	# ~ print(Gt, Gp)
	if Gt > Gp:
		escape = True
	else:
		m += 1
	# ~ print(y_matr)
	# ~ for avg in y_avg:
		# ~ print('{:.2f}'.format(avg), end=', ')
	# ~ print('\n ')

beta_list_norm = list(np.linalg.solve(norm_matrix, y_avg))	
beta_list_natur = list(np.linalg.solve(natur_matrix, y_avg))

print_result(x_max_list, k, norm_matrix, y_matr, y_avg, y_disp, beta_list_norm)
print_result(x_max_list, k, natur_matrix, y_matr, y_avg, y_disp, beta_list_natur)
