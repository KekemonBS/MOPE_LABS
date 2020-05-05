from prettytable import PrettyTable as prtt
from scipy.stats import f, t 
import numpy as np
import functools as ft
import itertools as itr
from math import sqrt
import random

# ~ Дано---------------------------- 
x1_min = -5
x1_max = 15

x2_min = -15
x2_max = 35

x3_min = 15
x3_max = 30

x_max_list = [x1_max, x2_max, x3_max]
x_min_list = [x1_min, x2_min, x3_min]

k = len(x_max_list)
m = 2
p = 0.95
with_interractions_and_squares = False

#y_max = 200 + sum(x_max_list)/len(x_max_list)
#y_min = 200 + sum(x_min_list)/len(x_min_list)
# ~ --------------------------------  

def fisher_critical(prob, f4, f3):
    return f.ppf(p, f4, f3)

def student_critical(q, f3):
    return t.ppf((1 + (1 - q)) / 2, f3)

def cochran_critical(q, f1, f2):
    return 1 / (1 + (f2 - 1) / f.ppf(1 - q/f2, f1, (f2 - 1)*f1) )

def print_matrix(x_max_list, k, x_matr, y_matr, y_avg, y_disp):
	adequate_table  = prtt()
	group = ['x'+str(n+1) for n in range(len(x_max_list))]
	if with_interractions_and_squares :
		group_quantity = k+1
	else:
		group_quantity = 1
	header = []
	for j in range(group_quantity):
		header.extend(list(itr.combinations(group,j+1)))
	if with_interractions_and_squares:
		header.extend([('x'+str(n+1)+'^2',) for n in range(len(x_max_list))])
	for i in range(len(y_matr[0])):
		header.append('y'+str(i+1))
	header.append('y_avg')
	header.append('s2{y}') # ~ variance or disp
	# ~ add squares
	final_matrix = np.hstack((x_matr,y_matr,[[round(avg,2)] for avg in y_avg],[[round(disp,2)] for disp in y_disp]))
	header = list([ft.reduce(lambda x,y : x+y, i) for i in header])
	adequate_table.field_names = header
	for row_number in range(len(final_matrix)):
		adequate_table.add_row(final_matrix[row_number])
	print(adequate_table)

def print_equation(x_max_list, k, beta_list, ):
	group = ['x'+str(n+1) for n in range(len(x_max_list))]
	if with_interractions_and_squares:
		group_quantity = k+1
		header = []
		for j in range(group_quantity):
			header.extend(list(itr.combinations(group,j+1)))
		header.extend([('x'+str(n+1)+'^2',) for n in range(len(x_max_list))])
		x_names = list([ft.reduce(lambda x,y : x+y, i) for i in header])
		form = '*{:.4f} + '.join(x_names)+'*{:.4f} =  y'
	else:
		group_quantity = 1
		header = []
		for j in range(group_quantity):
			header.extend(list(itr.combinations(group,j+1)))
		x_names = list([ft.reduce(lambda x,y : x+y, i) for i in header])
		form = '*{:.4f} + '.join(x_names)+'*{:.4f} =  y'
	print(form.format(*beta_list))

def generate_y_matr(natur_matrix,m):
	y_matr = []
	natur_list = list(map(lambda x: list(x),natur_matrix))
	for i in natur_list:
		y_row = []
		for j in range(m):
			x1 = i[0]
			x2 = i[1]
			x3 = i[2]
			y = 3.9+5.6*x1+7.9*x2+7.3*x3+2.0*x1*x1+0.5*x2*x2+4.2*x3*x3+1.5*x1*x2+0.1*x1*x3+9.9*x2*x3+5.3*x1*x2*x3 + random.random()*10 - 5
			y_row.append(round(y,3))
		y_matr.append(y_row)
	return y_matr

adequacy = False
while adequacy == False :	
	# ~ Normalized matrix
	# ~ factors
	items = [(-1,1) for i in range(k)]
	f_matrix = list(itr.product(*items))
	# ~ add zor t
	initial_f_matrix = f_matrix.copy()
	l = round(sqrt(len(x_max_list)),2)
	zor_t_list = []
	for i in range(k):
		zor_t_row_pos = tuple([ l if i==j else 0 for j in range(k)])
		zor_t_row_neg = tuple([ -l if i==j else 0 for j in range(k)])
		zor_t_list.append(zor_t_row_pos)
		zor_t_list.append(zor_t_row_neg)
#	zero_t_row = tuple([0 for j in range(k)])
#	zor_t_list.append(zero_t_row)
	f_matrix.extend(zor_t_list)
	# ~ with interractions---------------------------		
	if with_interractions_and_squares :
		interractions_matrix1 = []
		group_quantity = k - 1
		for i in f_matrix:
			comb_list = []
			for j in range(group_quantity):
				comb_list.extend(list(itr.combinations(i,j+2)))
			comb_values = [round(np.prod(k),2) for k in comb_list]
			squared_values = [round(x**2,2) for x in i]
			comb_and_squared_values = []
			comb_and_squared_values.extend(comb_values)
			comb_and_squared_values.extend(squared_values)
			interractions_matrix1.append(comb_and_squared_values)
	# ~ ---------------------------------------------
	if with_interractions_and_squares :
		norm_matrix = np.hstack((f_matrix, interractions_matrix1)) #x0_vector, 
	else:
		norm_matrix = f_matrix #x0_vector, 
	# ~ Naturalized matrix
	col_list = []
	for i in range(len(f_matrix[0])):
		col1 = [row[i] for row in initial_f_matrix]
		col1 = list(map(lambda x : x_max_list[i] if x==1 else x_min_list[i], col1))

		col2 = [row[i] for row in zor_t_list]
		x0i =(x_max_list[i]+x_min_list[i])/2
		deltaxi = x_max_list[i]-x0i
		col2 = list(map(lambda x : round(x0i,2) if x==0 else round(x*deltaxi+x0i,2), col2))
		
		col12 = col1+col2
		col_list.append(col12)
	nf_matrix = list(zip(*col_list)) # ~ naturalized factors
	
	# ~ with interractions---------------------------	
	if with_interractions_and_squares :
		interractions_matrix2 = []
		group_quantity = k - 1
		for i in nf_matrix:
			comb_list = []
			for j in range(group_quantity):
				comb_list.extend(list(itr.combinations(i,j+2)))
			comb_values = [round(np.prod(k),2) for k in comb_list]
			squared_values = [round(x**2,2) for x in i]
			comb_and_squared_values = []
			comb_and_squared_values.extend(comb_values)
			comb_and_squared_values.extend(squared_values)
			interractions_matrix2.append(comb_and_squared_values)
	# ~ ---------------------------------------------
	if with_interractions_and_squares :
		natur_matrix = np.hstack((nf_matrix, interractions_matrix2)) #x0_vector, 
	else:
		natur_matrix = nf_matrix #x0_vector, 
	# ~ --------------------------------------------------------------------------------
	escape = False
	while escape == False:
		y_matr = generate_y_matr(natur_matrix, m) #np.random.randint(y_min, y_max,(len(natur_matrix),m))#####################################
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
			form = 'Cochran`s test passed with significance level {:.4f} : Gt > Gp'
		else:
			m += 1
			form = 'Cochran`s test failed with significance level {:.4f} : Gt < Gp'
		print(form.format(q))
		print('Gt = {}\nGp = {}'.format(Gt, Gp))


	# ~ --------------------------------------------------------------------------------	
	if with_interractions_and_squares :
		beta_list_norm1 = list(np.linalg.solve(norm_matrix[3:13], y_avg[3:13]))	
		beta_list_natur1 = list(np.linalg.solve(natur_matrix[3:13], y_avg[3:13]))
		beta_list_norm2 = list(np.linalg.solve(norm_matrix[1:11], y_avg[1:11]))	
		beta_list_natur2 = list(np.linalg.solve(natur_matrix[1:11], y_avg[1:11]))
		beta_list_norm = list(zip(beta_list_norm1,beta_list_norm2))
		beta_list_norm =list(map(lambda x : sum(x)/2, beta_list_norm))
		beta_list_natur = list(zip(beta_list_natur1,beta_list_natur2))
		beta_list_natur =list(map(lambda x : sum(x)/2, beta_list_natur))
	else:
		beta_list_norm = list(np.linalg.solve(norm_matrix[1:4], y_avg[1:4]))	
		beta_list_natur = list(np.linalg.solve(natur_matrix[1:4], y_avg[1:4]))
	print_matrix(x_max_list, k, norm_matrix, y_matr, y_avg, y_disp)
	print('Equation with normalized coefficients : ')
	print_equation(x_max_list, k, beta_list_norm)
	print_matrix(x_max_list, k, natur_matrix, y_matr, y_avg, y_disp)
	print('Equation with naturalized coefficients : ')
	print_equation(x_max_list, k, beta_list_natur)
	# ~ --------------------------------------------------------------------------------


	print('\nStudent`s test')
	N = len(norm_matrix)
	f3 = f1 * f2
	S2b = sum(y_disp)**2 / (N * N * m)
	Sb = S2b
	betast_list = []
	for i in range(len(norm_matrix[0])):
		x_list_tmp = np.array((norm_matrix))[:,i]
		beta_tmp = ((sum([np.prod(i) for i in list(zip(x_list_tmp, y_avg))])) / N)
		betast_list.append(beta_tmp)

	T_list = [abs(beta)/Sb for beta in betast_list]
	T = student_critical(q, f3)
	print('T = '+str(T)+ '\nT_list = '+str(list(map(lambda x : round(x, 2), T_list))))
	for i in range(len(T_list)):
		if T_list[i] < T :
			T_list[i] = 0
			beta_list_natur[i] = 0
	print('Fixed beta_list = '+ str(list(map(lambda x :round(x, 2), beta_list_natur))))
	print('Equation without insignificant coefficients : ')
	print_equation(x_max_list, k, beta_list_natur)

	print("\nFisher test")
	equation_y_list = []
	for i in range(len(natur_matrix)):
		x_list_tmp = natur_matrix[i]
		y_tmp = sum([np.prod(i) for i in list(zip(x_list_tmp, beta_list_natur))])
		equation_y_list.append(y_tmp)

	beta_list_natur = list(filter(lambda i : (i != 0), beta_list_natur))
	d = len(beta_list_natur)
	f4 = N - d 
	S2ad = m * (sum([(i[0]-i[1])**2 for i in list(zip(equation_y_list, y_avg))])) /f4
	Fp = sqrt(S2ad) / S2b
	Ft = fisher_critical(p, f4, f3)
	print('Fp = '+ str(Fp)+"\nFt = "+str(Ft))
	if Fp > Ft:
		print("	The regression equation is inadequate at the significance level {:.2f}".format(q))
		with_interractions_and_squares = True
	else:
		print("	The regression equation is adequate at the significance level {:.2f}".format(q))
		adequacy = True	
		
	print('\nResult check')
	counter = 1
	for i in list(zip(equation_y_list, y_avg)):
		print('y{counter:2} = {:13.4f} ;  y_avg{counter:2} = {:13.4f} ;'.format(i[0], i[1], counter = counter))
		counter += 1
