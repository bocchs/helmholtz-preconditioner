
import numpy as np
import scipy.linalg
import sys
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import scipy.sparse
np.set_printoptions(threshold=sys.maxsize,precision=3,linewidth=np.inf)

def sigma1(x,const,eta):
	if x <= eta:
		return const / eta * ((x - eta)/eta)**2
	elif x >= 1 - eta:
		return const / eta * ((x - 1 + eta)/eta)**2
	else:
		return 0 


def sigma2(x,const,eta):
	if x <= eta:
		return const / eta * ((x - eta)/eta)**2
	else:
		return 0

def s1(x,const,eta):
	return (1 + 1j*sigma1(x,const,eta)/omega)**-1


def s2(x,const,eta):
	return (1 + 1j*sigma2(x,const,eta)/omega)**-1

def s2m(x,m,b,const,eta):
	return (1 + 1j*sigma2(x-(m-b)*h,const,eta)/omega)**-1

# velocity field 1 described in paper
def init_c1_mat(r1,r2):
	x_i = np.linspace(0,1,n+2)
	xx, yy = np.meshgrid(x_i,x_i)
	c_mat = 4/3 * (1-.5*np.exp(-32*((xx-r1)**2 + (yy-r2)**2)))
	return c_mat

# external force 1 described in paper
def init_f1_mat(r1,r2):
	x_i = np.linspace(0,1,n+2) # n+2 points in each dimension, including boundary
	xx, yy = np.meshgrid(x_i[1:-1],x_i[1:-1]) # n interior points
	f_mat = np.exp(-(4*omega/np.pi)**2*((xx-r1)**2 + (yy-r2)**2))
	return f_mat



# computes desired n x n block of A_a,a that corresponds to a row a in the grid
# a = 1..n
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_diag_block(a, choose_s2, m, b, const, eta, h):
	c1_vec = np.zeros((n-1,), dtype=np.cdouble)
	c2_vec = np.zeros((n-1,), dtype=np.cdouble)
	c5_vec = np.zeros((n,), dtype=np.cdouble)

	c1_idx = 0
	c2_idx = 0
	c5_idx = 0
	row_i = 1 # 1..n

	j = a
	for i in range(1, n+1):
		x1 = (i-.5)*h
		x2 = j*h
		if choose_s2:
			c1 = 1/h**2 * (s1(x1,const,eta) / s2(x2,const,eta))
		else:
			c1 = 1/h**2 * (s1(x1,const,eta) / s2m(x2,m,b,const,eta))
		if row_i >= 2:
			c1_vec[c1_idx] = c1
			c1_idx += 1

		x1 = (i+.5)*h
		x2 = j*h
		if choose_s2:
			c2 = 1/h**2 * (s1(x1,const,eta) / s2(x2,const,eta))
		else:
			c2 = 1/h**2 * (s1(x1,const,eta) / s2m(x2,m,b,const,eta))			
		if row_i <= n - 1:
			c2_vec[c2_idx] = c2
			c2_idx += 1

		x1 = i*h
		x2 = (j-.5)*h
		if choose_s2:
			c3 = 1/h**2 * (s2(x2,const,eta) / s1(x1,const,eta))
		else:
			c3 = 1/h**2 * (s2m(x2,m,b,const,eta) / s1(x1,const,eta))

		x1 = i*h
		x2 = (j+.5)*h
		if choose_s2:
			c4 = 1/h**2 * (s2(x2,const,eta) / s1(x1,const,eta))
		else:
			c4 = 1/h**2 * (s2m(x2,m,b,const,eta) / s1(x1,const,eta))

		x1 = i*h
		x2 = j*h
		if choose_s2:
			c5 = omega**2 / (s1(x1,const,eta)*s2(x2,const,eta)*c_mat[i-1,j-1]**2) - (c1 + c2 + c3 + c4)
		else:
			c5 = omega**2 / (s1(x1,const,eta)*s2m(x2,m,b,const,eta)*c_mat[i-1,j-1]**2) - (c1 + c2 + c3 + c4)
		c5_vec[c5_idx] = c5
		c5_idx += 1

		row_i += 1

	A_block = scipy.sparse.diags(c5_vec) + scipy.sparse.diags(c1_vec,-1) \
				+ scipy.sparse.diags(c2_vec,1)
	return A_block


# computes desired n x n block of A_row,col that corresponds to "col'th" row of the grid
# row,col = 1..n (indexes the block matrix)
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_block(row, col, choose_s2, m, b, const, eta, h):
	assert(row >= 1 and row <= n and row >= 1 and row <= n)
	if row == col:
		return get_A_diag_block(row,choose_s2,m,b,const,eta,h)
	if abs(row-col) > 1:
		return sparse.csc_matrix((n, n), dtype=np.cdouble) # zeros

	c4_vec = np.zeros((n,), dtype=np.cdouble)

	j = col
	x2 = (j+.5)*h
	for i in range(1, n+1):
		x1 = i*h
		if choose_s2:
			c4 = 1/h**2 * (s2(x2,const,eta) / s1(x1,const,eta))
		else:
			c4 = 1/h**2 * (s2m(x2,m,b,const,eta) / s1(x1,const,eta))
		c4_vec[i-1] = c4

	A_block = scipy.sparse.diags(c4_vec)
	return A_block


# computes bn x bn block of A_F,F that corresponds to first b rows of grid
# uses s2m for aux prob on subgrid
def get_A_FF_block(b, const, eta, h):
	# A_block_diags = b*[scipy.sparse.csc_matrix((n, n), dtype=np.cdouble)]
	# A_block_off_diags = (b-1)*[scipy.sparse.csc_matrix((n, n), dtype=np.cdouble)]
	# for i in range(1,b+1):
	# 	A_block_diags[i-1] = get_A_block(i,i,s2)
	# for i in range(1,b):
		# A_block_off_diags[i-1] = get_A_block(i,i+1,s2)
	# # diag_blocks = scipy.linalg.block_diag(*A_block_diags)
	# diag_blocks = scipy.sparse.block_diag(A_block_diags)
	# diag_elems = diag_blocks.diagonal()
	# off_diag_blocks = scipy.sparse.block_diag(A_block_off_diags)
	# off_diag_elems = off_diag_blocks.diagonal()
	# upper = scipy.sparse.diags(off_diag_elems,n)
	# lower = scipy.sparse.diags(off_diag_elems,-n)
	# middle = scipy.sparse.diags(diag_elems)
	# return upper + middle + lower
	diag_block_ra = []
	A_block_off_diags_ra = []
	for i in range(1,b):
		A_block_off_diags_ra.append(get_A_block(i,i+1,False,i,b,const,eta,h))
	off_diag_blocks = scipy.sparse.block_diag(A_block_off_diags_ra)
	off_diag_elems = off_diag_blocks.diagonal()
	upper = scipy.sparse.diags(off_diag_elems,n)
	lower = scipy.sparse.diags(off_diag_elems,-n)
	for i in range(1,b+1):
		diag_block_ra.append(get_A_block(i,i,False,i,b,const,eta, h))
	middle = scipy.sparse.block_diag(diag_block_ra)
	A_FF = middle# + upper + lower
	return A_FF



# computes bn x n block of A_F,b+1 that corresponds to first b rows of grid
# take transpose of this result to get n x bn block of A_b+1,F
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_Fb1_block(choose_s2, m, b, const, eta, h):
	A_block = get_A_block(b,b+1,choose_s2,m,b,const,eta, h)
	return scipy.sparse.vstack((scipy.sparse.csc_matrix(((b-1)*n, n), dtype=np.cdouble), A_block))


# computes permutation matrix from row major to column major ordering for one PML
# subgrid of b layers
def get_P_mat():
	Pm = np.zeros((b*n,b*n))
	# for each column
	for i in range(n):
		# compute submatrix that corresponds to one column of grid
		for j in range(b):
			Pm[i*b+j, i+j*n] = 1
	Pm = scipy.sparse.csc_matrix(Pm)
	return Pm


def build_A_matrix(choose_s2, m, b, const, eta, h):
	block_diags_ra = []
	off_diags_ra = [] # array of diagonals
	for i in range(1,n+1):
		block_diags_ra.append(get_A_block(i,i,choose_s2,m,b,const,eta,h))
	for i in range(1,n):
		off_diags_ra.append(get_A_block(i,i+1,choose_s2,m,b,const,eta,h).diagonal())
	block_diag = scipy.sparse.block_diag(block_diags_ra)
	off_diags = np.concatenate(off_diags_ra)
	upper = scipy.sparse.diags(off_diags,n)
	lower = scipy.sparse.diags(off_diags,-n)
	A = block_diag + upper + lower
	return A




# Computes Hm: bn x bn A matrix for PML's b x n subgrid for layer m
def get_Hm(m, b, const, eta, h):
	c1_vec = np.zeros((b*n-1,), dtype=np.cdouble)
	c2_vec = np.zeros((b*n-1,), dtype=np.cdouble)
	c3_vec = np.zeros((b*n-n,), dtype=np.cdouble)
	c4_vec = np.zeros((b*n-n,), dtype=np.cdouble)
	c5_vec = np.zeros((b*n,), dtype=np.cdouble)

	c1_idx = 0
	c2_idx = 0
	c3_idx = 0
	c4_idx = 0
	c5_idx = 0
	row_i = 1 # row in A matrix 1..bn
	for j in range(m-b+1, m+1):
		for i in range(1, n+1):
			x1 = (i-.5)*h
			x2 = j*h
			c1 = 1/h**2 * (s1(x1,const,eta) / s2m(x2,m,b,const,eta))
			if row_i >= 2:
				c1_vec[c1_idx] = c1
				c1_idx += 1


			x1 = (i+.5)*h
			x2 = j*h
			c2 = 1/h**2 * (s1(x1,const,eta) / s2m(x2,m,b,const,eta))
			if row_i <= b*n - 1:
				c2_vec[c2_idx] = c2
				c2_idx += 1

			x1 = i*h
			x2 = (j-.5)*h
			c3 = 1/h**2 * (s2m(x2,m,b,const,eta) / s1(x1,const,eta))
			if row_i >= n+1:
				c3_vec[c3_idx] = c3
				c3_idx += 1

			x1 = i*h
			x2 = (j+.5)*h
			c4 = 1/h**2 * (s2m(x2,m,b,const,eta) / s1(x1,const,eta))
			if row_i <= b*n - n:
				c4_vec[c4_idx] = c4
				c4_idx += 1

			x1 = i*h
			x2 = j*h
			c5 = omega**2 / (s1(x1,const,eta)*s2m(x2,m,b,const,eta)*c_mat[i-1,j-1]**2) - (c1 + c2 + c3 + c4)
			c5_vec[c5_idx] = c5
			c5_idx += 1

			row_i += 1

	c1_vec[n-1::n] = 0
	c2_vec[n-1::n] = 0

	A = scipy.sparse.diags(c5_vec) + scipy.sparse.diags(c1_vec,-1) \
		+ scipy.sparse.diags(c2_vec,1) + scipy.sparse.diags(c3_vec, -n) \
		+ scipy.sparse.diags(c4_vec, n)
	return A


def algo2_1(const, eta, h):
	S1 = get_A_block(1,1,True,0,0,const,eta,h).A
	T = scipy.linalg.inv(S1)

	S_ra = [S1]
	T_ra = [T]
	for m in range(2,n+1):
		Amm = get_A_block(m,m,True,0,0,const,eta,h).A
		Amm1 = get_A_block(m,m-1,True,0,0,const,eta,h).A
		Am1m = np.copy(Amm1).T
		Sm = Amm - Amm1@T@Am1m
		T = scipy.linalg.inv(Sm)
		S_ra.append(Sm)
		T_ra.append(T)
	# end algo

	# rebuild A matrix from LDL factorization
	L_ra = []
	for k in range(1,n):
		L = np.eye(n**2,dtype=np.cdouble)
		L[k*n:(k+1)*n,(k-1)*n:k*n] = get_A_block(k+1,k,True,m,b,const,eta,h)@T_ra[k-1]
		L_ra.append(L)
	A_rebuilt = np.eye(n**2,dtype=np.cdouble)
	for i in range(1,n):
		A_rebuilt = A_rebuilt@L_ra[i-1]
	A_rebuilt = A_rebuilt@scipy.sparse.block_diag(S_ra)
	for i in range(n-1,0,-1):
		A_rebuilt = A_rebuilt@(L_ra[i-1].T)
	# print(np.real(A.A))
	# print()
	# print(np.real(A_rebuilt))
	# print()
	diff = A_rebuilt - A
	print(np.imag(diff))
	print("real part max diff = " + str(np.max(np.abs(np.real(diff)))))
	print("imag part max diff = " + str(np.max(np.abs(np.imag(diff)))))
	print("max diff magnitude = " + str(np.max(np.abs(diff))))
	# sys.exit()
	return T_ra, S_ra, L_ra, A_rebuilt


def algo2_2(T_ra, S_ra, L_ra, const, eta):
	u = np.copy(f_mat).astype(np.cdouble)
	for m in range(1,n):
		u[m] = u[m] - get_A_block(m+1,m,True,0,0,const,eta,h).A@T_ra[m-1]@u[m-1]
	for m in range(1,n+1):
		u[m-1] = T_ra[m-1]@u[m-1]
	for m in range(n-1,0,-1):
		u[m-1] = u[m-1] - T_ra[m-1]@get_A_block(m,m+1,True,0,0,const,eta,h)@u[m]
	return u



def algo2_3(b, const, eta, h):
	HF = get_A_FF_block(b,const,eta,h)
	PF = get_P_mat()
	# print(PF.A)
	# sys.exit()
	PHP = PF @ HF @ PF.T
	# print(np.real(PHP.A))
	# print(PF.shape)
	# print(HF.shape)
	# sys.exit()
	lu = scipy.sparse.linalg.splu(PHP)
	LF = lu.L
	UF = lu.U

	Hm_ra = []
	Lm_ra = []
	Um_ra = []
	Pm = get_P_mat()
	for m in range(b+1,n+1):
		Hm = get_Hm(m,b,const,eta,h)
		# print(np.real(Hm.A))
		# print()
		Hm_ra.append(Hm)
		PHP = Pm@Hm@Pm.T
		print(np.real(PHP.A))
		sys.exit()
		lu = scipy.sparse.linalg.splu(PHP)
		# print(np.real(lu.L.A))
		# print()
		# print(np.real(lu.U.A))
		# print()
		# print()
		# print()
		Lm_ra.append(lu.L)
		Um_ra.append(lu.U)
	# sys.exit()
	return HF, LF, UF, Hm_ra, Lm_ra, Um_ra


def prec(f_vec, b, n, TF, A_b1F, A_Fb1, A_ra, mat_ra):
	f_mat = f_vec.reshape((n,n))
	uF = np.zeros((b,n), dtype=np.cdouble)
	for i in range(b):
		uF[i] = f_mat[i]
	um_ra = np.zeros((n-b,n), dtype=np.cdouble)
	for i in range(n-b):
		um_ra[i] = f_mat[i+b]
	u = np.vstack((uF, um_ra))
	TFuF = TF@(uF.reshape(-1))
	u[b] = u[b] - A_b1F@TFuF
	for m in range(b+1, n):
		A = A_ra[m-1].T
		mat = mat_ra[m-1-b]
		u_temp = np.zeros((mat.shape[0],), dtype=np.cdouble)
		u_temp[-n:] = u[m-1]
		Tu = mat @ u_temp
		Tu = Tu[-n:]
		u[m] = u[m] - A@Tu
	uF = TF@(uF.reshape(-1))
	for m in range(b+1, n+1):
		mat = mat_ra[m-1-b]
		u_temp = np.zeros((mat.shape[0],), dtype=np.cdouble)
		u_temp[-n:] = u[m-1]
		Tu = mat @ u_temp
		Tu = Tu[-n:]
		u[m-1] = Tu
	for m in range(n-1, b, -1):
		A = A_ra[m-1]
		mat = mat_ra[m-1-b]
		Au_temp = np.zeros((mat.shape[0],), dtype=np.cdouble)
		Au_temp[-n:] = A@u[m]
		TAu = mat @ Au_temp
		TAu = TAu[-n:]
		u[m-1] = u[m-1] - TAu
	uF = uF - TF@A_Fb1@u[b]
	for i in range(b):
		u[i] = uF[i*n:(i+1)*n]
	return u



if __name__ == "__main__":
	alpha = 2
	lam = 4
	omega = 2*np.pi*lam + 1j*alpha # angular frequency
	const = 1 # appropriate positive constant for sigma1, sigma2

	n = 5#127 # interior grid size
	h = 1 / (n + 1) # spatial step size
	b = 2# 12 # width of PML in number of grid points
	eta = b*h # width of PML in spatial dim

	u_mat = np.zeros((n,n))
	r1 = .5
	r2 = .5
	f_mat = init_f1_mat(r1,r2)
	f_vec = f_mat.flatten()
	c_mat = init_c1_mat(r1,r2)

	# plt.imshow(np.real(c_mat))
	# plt.show()
	# sys.exit()

	A = build_A_matrix(True,0,0,const,eta,h)
	"""
	u_true, exit_code = scipy.sparse.linalg.gmres(A, f_vec, tol=1e-3, callback=print, callback_type='pr_norm')
	u_true = u_true.reshape((n,n))
	plt.figure()
	plt.imshow(np.imag(u_true))
	# plt.show()
	# sys.exit()


	# print(np.real(A.A))
	# print()
	

	T_ra, S_ra, L_ra, A_rebuilt = algo2_1(const,eta,h)
	u_solved = algo2_2(T_ra, S_ra, L_ra,const,eta)
	print(np.max(np.abs(u_true - u_solved)))
	plt.figure()
	plt.imshow(np.imag(u_solved))
	u_rebuilt, exit_code = scipy.sparse.linalg.gmres(A_rebuilt, f_vec, tol=1e-3, callback=print, callback_type='pr_norm')
	plt.figure()
	u_rebuilt = u_rebuilt.reshape((n,n))
	plt.imshow(np.imag(u_rebuilt))
	plt.show()
	# sys.exit()
	"""





	
	HF, LF, UF, Hm_ra, Lm_ra, Um_ra = algo2_3(b,const,eta,h)

	# initalize variables used in prec
	PF = get_P_mat()
	Pm = PF
	mat_ra = []
	for i in range(len(Um_ra)):
		fname = 'saved_files/mat_ra'+str(i)+'_n' + str(n) + '_b' + str(b) + '.npz'
		print(str(i))
		Um_inv = scipy.sparse.linalg.inv(Um_ra[i])
		Lm_inv = scipy.sparse.linalg.inv(Lm_ra[i])
		mat = Pm.T@Um_inv@Lm_inv@Pm
		# scipy.sparse.save_npz(fname, mat)
		# mat = scipy.sparse.load_npz(fname)
		mat_ra.append(mat)

	UF_inv = scipy.sparse.linalg.inv(UF)
	LF_inv = scipy.sparse.linalg.inv(LF)
	TF = PF.T@UF_inv@LF_inv@PF
	A_b1F = get_A_Fb1_block(False,1,b,const,eta,h).T ################### CHECK HERE ################
	A_Fb1 = get_A_Fb1_block(False,1,b,const,eta,h)
	A_ra = []
	for i in range(1,n):
		if i <= b:
			A_ra.append(get_A_block(i,i+1,True,0,0,const,eta,h))
		else:
			A_ra.append(get_A_block(i,i+1,False,b,i,const,eta,h))
	

	M = scipy.sparse.linalg.LinearOperator((n**2,n**2), \
					matvec=lambda f_vec: prec(f_vec, b, n, TF, A_b1F, A_Fb1, A_ra, mat_ra))

	# temp = M.matmat(A.A)
	# plt.figure(1)
	# plt.imshow(np.real(temp))
	# plt.title('real(M*A)')
	# plt.colorbar()
	# plt.figure(2)
	# plt.imshow(np.imag(temp))
	# plt.title('imag(M*A)')
	# plt.colorbar()
	# plt.show()
	# print('cond(M*A) = ' + str(np.linalg.cond(temp)))
	# print('cond(A) = ' + str(np.linalg.cond(A.A)))
	# sys.exit()
	# print(np.real(temp)) # should be approx identity matrix
	print("AAA")
	u, exit_code = scipy.sparse.linalg.gmres(A, f_vec, M=M, tol=1e-3, restart=30, maxiter=5, callback=print,callback_type='pr_norm')
	# u, exit_code = scipy.sparse.linalg.gmres(A, f_vec, M=M, tol=1e-3, callback=print, callback_type='pr_norm')
	# u, exit_code = scipy.sparse.linalg.gmres(A, f_vec, tol=1e-3, restart=30, maxiter=1, callback=print,callback_type='pr_norm')
	# u, exit_code = scipy.sparse.linalg.gmres(A, f_vec, tol=1e-3, callback=print, callback_type='pr_norm')
	print("BBB")
	if exit_code > 0:
		print("GMRES: convergence to tolerance not achieved")
	elif exit_code < 0:
		print("GMRES: illegal input or breakdown")
	else:
		print("GMRES: convergence achieved")
	
	u = u.reshape((n,n))


	# plt.imshow(np.abs(u))
	plt.imshow(np.real(u))
	plt.show()

