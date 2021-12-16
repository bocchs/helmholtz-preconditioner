
import numpy as np
import scipy.linalg
import sys
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import scipy.sparse
from numba import jit

@jit(nopython=True)
def sigma1(x,const,eta):
	if x <= eta:
		return const / eta * ((x - eta)/eta)**2
	elif x >= 1 - eta:
		return const / eta * ((x - 1 + eta)/eta)**2
	else:
		return 0 

@jit(nopython=True)
def sigma2(x,const,eta):
	if x <= eta:
		return const / eta * ((x - eta)/eta)**2
	else:
		return 0

@jit(nopython=True)
def s1(x,const,eta,omega):
	return (1 + 1j*sigma1(x,const,eta)/omega)**-1

@jit(nopython=True)
def s2(x,const,eta,omega):
	return (1 + 1j*sigma2(x,const,eta)/omega)**-1

@jit(nopython=True)
def s2m(x,m,b,const,eta,omega,h):
	return (1 + 1j*sigma2(x-(m-b)*h,const,eta)/omega)**-1

# velocity field 1 described in paper
def init_c1_mat(r1,r2,n):
	x_i = np.linspace(0,1,n+2)
	xx, yy = np.meshgrid(x_i,x_i)
	c_mat = 4/3 * (1-.5*np.exp(-32*((xx-r1)**2 + (yy-r2)**2)))
	return c_mat

# external force 1 described in paper
def init_f1_mat(r1,r2,omega,n):
	x_i = np.linspace(0,1,n+2) # n+2 points in each dimension, including boundary
	xx, yy = np.meshgrid(x_i[1:-1],x_i[1:-1]) # n interior points
	f_mat = np.exp(-(4*omega/np.pi)**2*((xx-r1)**2 + (yy-r2)**2))
	return f_mat


# helper function for get_A_diag_block()
@jit(nopython=True)
def get_A_diag_block_coeffs(a, b, const, eta, omega, h, n, c_mat):
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
		c1 = 1/h**2 * (s1(x1,const,eta,omega) / s2(x2,const,eta,omega))
		if row_i >= 2:
			c1_vec[c1_idx] = c1
			c1_idx += 1

		x1 = (i+.5)*h
		x2 = j*h
		c2 = 1/h**2 * (s1(x1,const,eta,omega) / s2(x2,const,eta,omega))		
		if row_i <= n - 1:
			c2_vec[c2_idx] = c2
			c2_idx += 1

		x1 = i*h
		x2 = (j-.5)*h
		c3 = 1/h**2 * (s2(x2,const,eta,omega) / s1(x1,const,eta,omega))

		x1 = i*h
		x2 = (j+.5)*h
		c4 = 1/h**2 * (s2(x2,const,eta,omega) / s1(x1,const,eta,omega))

		x1 = i*h
		x2 = j*h
		c5 = omega**2 / \
			(s1(x1,const,eta,omega)*s2(x2,const,eta,omega)*c_mat[i-1,j-1]**2) \
			 - (c1 + c2 + c3 + c4)
		c5_vec[c5_idx] = c5
		c5_idx += 1

		row_i += 1

	return c1_vec, c2_vec, c5_vec


# computes desired n x n block of A_a,a that corresponds to a row a in the grid
# a: block index in diagonal
def get_A_diag_block(a, b, const, eta, omega, h, n, c_mat):
	c1_vec, c2_vec, c5_vec = \
				get_A_diag_block_coeffs(a, b, const, eta, omega, h, n, c_mat)
	A_block = scipy.sparse.diags(c5_vec) \
			+ scipy.sparse.diags(c1_vec,-1) \
			+ scipy.sparse.diags(c2_vec,1)
	return A_block


# helper function for get_A_block()
@jit(nopython=True)
def get_upper_A_block(row, col, b, const, eta, omega, h, n, c_mat):
	assert col == row + 1
	c4_vec = np.zeros((n,), dtype=np.cdouble)
	j = row
	x2 = (j+.5)*h
	for i in range(1, n+1):
		x1 = i*h
		c4 = 1/h**2 * (s2(x2,const,eta,omega) / s1(x1,const,eta,omega))
		c4_vec[i-1] = c4
	return c4_vec


# helper function for get_A_block()
@jit(nopython=True)
def get_lower_A_block(row, col, b, const, eta, omega, h, n, c_mat):
	assert row == col + 1
	c3_vec = np.zeros((n,), dtype=np.cdouble)
	j = row
	x2 = (j-.5)*h
	for i in range(1, n+1):
		x1 = i*h
		c3 = 1/h**2 * (s2(x2,const,eta,omega) / s1(x1,const,eta,omega))
		c3_vec[i-1] = c3
	return c3_vec


# computes desired n x n block of A_row,col that corresponds to "col'th" row of the grid
# row,col = 1..n (indexes the block matrix)
def get_A_block(row, col, b, const, eta, omega, h, n, c_mat):
	assert(row >= 1 and row <= n and row >= 1 and row <= n)
	if row == col:
		return get_A_diag_block(row,b,const,eta,omega,h,n,c_mat)
	elif col == row + 1:
		c4_vec = \
			get_upper_A_block(row, col, b, const, eta, omega, h, n, c_mat)
		A_block = scipy.sparse.diags(c4_vec)
		return A_block
	elif row == col + 1:
		c3_vec = \
			get_lower_A_block(row, col, b, const, eta, omega, h, n, c_mat)
		A_block = scipy.sparse.diags(c3_vec)
		return A_block
	else:
		return sparse.csc_matrix((n, n), dtype=np.cdouble) # zeros


# computes bn x bn block of A_F,F that corresponds to first b rows of grid
def get_A_FF_block(b, const, eta, omega, h, n, c_mat):
	diag_block_ra = []
	for i in range(1,b+1):
		diag_block_ra.append(get_A_block(i,i,b,const,eta,omega,h,n,c_mat))
	A_FF = scipy.sparse.block_diag(diag_block_ra)
	return A_FF


# computes bn x n block of A_(F,b+1) 
def get_A_Fb1_block(b, const, eta, omega, h, n, c_mat):
	A_block = get_A_block(b,b+1,b,const,eta,omega,h,n,c_mat)
	block = scipy.sparse.vstack((\
			scipy.sparse.csc_matrix(((b-1)*n, n), dtype=np.cdouble), A_block))
	return block


# computes n x bn block of A_(b+1,F)
def get_A_b1F_block(b, const, eta, omega, h, n, c_mat):
	A_block = get_A_block(b+1,b,b,const,eta,omega,h,n,c_mat)
	block = scipy.sparse.hstack((\
			scipy.sparse.csc_matrix((n, (b-1)*n), dtype=np.cdouble), A_block))
	return block


def build_A_matrix(b, const, eta, omega, h, n, c_mat):
	block_diags_ra = []
	up_off_diags_ra = [] # array of diagonals
	lo_off_diags_ra = []
	for i in range(1,n+1):
		block_diags_ra.append(get_A_block(i,i,b,const,eta,omega,h,n,c_mat))
	for i in range(1,n):
		up_off_diags_ra.append(\
				get_A_block(i,i+1,b,const,eta,omega,h,n,c_mat).diagonal())
		lo_off_diags_ra.append(\
				get_A_block(i+1,i,b,const,eta,omega,h,n,c_mat).diagonal())
	block_diag = scipy.sparse.block_diag(block_diags_ra)
	up_off_diags = np.concatenate(up_off_diags_ra)
	lo_off_diags = np.concatenate(lo_off_diags_ra)
	upper = scipy.sparse.diags(up_off_diags,n)
	lower = scipy.sparse.diags(lo_off_diags,-n)
	A = block_diag + upper + lower
	return A


# helper function for get_Hm()
@jit(nopython=True)
def get_Hm_coeffs(m, b, const, eta, omega, h, n, c_mat):
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
			c1 = 1/h**2 * (s1(x1,const,eta,omega) / s2m(x2,m,b,const,eta,omega,h))
			if row_i >= 2:
				c1_vec[c1_idx] = c1
				c1_idx += 1

			x1 = (i+.5)*h
			x2 = j*h
			c2 = 1/h**2 * (s1(x1,const,eta,omega) / s2m(x2,m,b,const,eta,omega,h))
			if row_i <= b*n - 1:
				c2_vec[c2_idx] = c2
				c2_idx += 1

			x1 = i*h
			x2 = (j-.5)*h
			c3 = 1/h**2 * (s2m(x2,m,b,const,eta,omega,h) / s1(x1,const,eta,omega))
			if row_i >= n+1:
				c3_vec[c3_idx] = c3
				c3_idx += 1

			x1 = i*h
			x2 = (j+.5)*h
			c4 = 1/h**2 * (s2m(x2,m,b,const,eta,omega,h) / s1(x1,const,eta,omega))
			if row_i <= b*n - n:
				c4_vec[c4_idx] = c4
				c4_idx += 1

			x1 = i*h
			x2 = j*h
			c5 = omega**2 / \
				(s1(x1,const,eta,omega)*s2m(x2,m,b,const,eta,omega,h)*c_mat[i-1,j-1]**2) \
				- (c1 + c2 + c3 + c4)
			c5_vec[c5_idx] = c5
			c5_idx += 1

			row_i += 1

	c1_vec[n-1::n] = 0
	c2_vec[n-1::n] = 0
	return c1_vec, c2_vec, c3_vec, c4_vec, c5_vec


# Computes Hm: bn x bn A matrix for PML's b x n subgrid for layer m
def get_Hm(m, b, const, eta, omega, h, n, c_mat):
	c1_vec, c2_vec, c3_vec, c4_vec, c5_vec = get_Hm_coeffs(m, b, const, eta, omega, h, n, c_mat)
	A = scipy.sparse.diags(c5_vec) + \
		scipy.sparse.diags(c1_vec,-1) + \
		scipy.sparse.diags(c2_vec,1) + \
		scipy.sparse.diags(c3_vec, -n) + \
		scipy.sparse.diags(c4_vec, n)
	return A


def algo2_1(b, const, eta, omega, h, n, c_mat):
	S1 = get_A_block(1,1,b,const,eta,omega,h,n,c_mat).A
	T = scipy.linalg.inv(S1)

	S_ra = [S1]
	T_ra = [T]
	for m in range(2,n+1):
		Amm = get_A_block(m,m,b,const,eta,omega,h,n,c_mat).A
		Amm1 = get_A_block(m,m-1,b,const,eta,omega,h,n,c_mat).A
		Am1m = get_A_block(m-1,m,b,const,eta,omega,h,n,c_mat).A
		Sm = Amm - Amm1@T@Am1m
		T = scipy.linalg.inv(Sm)
		S_ra.append(Sm)
		T_ra.append(T)
	# end algo

	# rebuild A matrix from LDL factorization exactly
	L_ra = []
	for k in range(1,n):
		L = np.eye(n**2,dtype=np.cdouble)
		L[k*n:(k+1)*n,(k-1)*n:k*n] = get_A_block(k+1,k,b,const,eta,omega,h,n,c_mat)@T_ra[k-1]
		L_ra.append(L)
	A_rebuilt = np.eye(n**2,dtype=np.cdouble)
	for i in range(1,n):
		A_rebuilt = A_rebuilt@L_ra[i-1]
	A_rebuilt = A_rebuilt@scipy.sparse.block_diag(S_ra)
	for i in range(n-1,0,-1):
		A_rebuilt = A_rebuilt@(L_ra[i-1].T)
	diff = A_rebuilt - build_A_matrix(b, const, eta, omega, h, n, c_mat)
	# print('diff:')
	# print(np.real(diff))
	# print("A_rebuilt:")
	# print(np.real(A_rebuilt))
	# print("A:")
	# print(np.real(A))
	print("real part max diff = " + str(np.max(np.abs(np.real(diff)))))
	print("imag part max diff = " + str(np.max(np.abs(np.imag(diff)))))
	print("max diff magnitude = " + str(np.max(np.abs(diff))))
	return T_ra, S_ra, L_ra, A_rebuilt


def algo2_2(T_ra, S_ra, L_ra, const, eta, omega, b, h, n, c_mat, f_mat):
	u = np.copy(f_mat).astype(np.cdouble)
	for m in range(1,n):
		u[m] = u[m] - get_A_block(m+1,m,b,const,eta,omega,h,n,c_mat).A@T_ra[m-1]@u[m-1]
	for m in range(1,n+1):
		u[m-1] = T_ra[m-1]@u[m-1]
	for m in range(n-1,0,-1):
		u[m-1] = u[m-1] - T_ra[m-1]@get_A_block(m,m+1,b,const,eta,omega,h,n,c_mat)@u[m]
	return u


def algo2_3(b, const, eta, omega, h, n, c_mat):
	HF = get_A_FF_block(b,const,eta,omega,h,n,c_mat).tocsc()
	lu_HF = scipy.sparse.linalg.splu(HF)
	lu_Hm_ra = []
	for m in range(b+1,n+1):
		Hm = get_Hm(m,b,const,eta,omega,h,n,c_mat).tocsc()
		lu_Hm = scipy.sparse.linalg.splu(Hm)
		lu_Hm_ra.append(lu_Hm)
	return lu_HF, lu_Hm_ra


def algo2_4(f_vec, b, n, lu_HF, A_b1F, A_Fb1, up_A_ra, lo_A_ra, lu_Hm_ra):
	f_mat = f_vec.reshape((n,n))
	uF = np.zeros((b,n), dtype=np.cdouble)
	for i in range(b):
		uF[i] = f_mat[i]
	um_ra = np.zeros((n-b,n), dtype=np.cdouble)
	for i in range(n-b):
		um_ra[i] = f_mat[i+b]
	u = np.vstack((uF, um_ra))
	TFuF = lu_HF.solve(uF.flatten())
	u[b] = u[b] - A_b1F@TFuF
	for m in range(b+1, n):
		A = lo_A_ra[m-1]
		u_temp = np.zeros((b*n,), dtype=np.cdouble)
		u_temp[-n:] = u[m-1]
		u[m] = u[m] - A@lu_Hm_ra[m-b-1].solve(u_temp)[-n:]
	uF = TFuF
	for m in range(b+1, n+1):
		u_temp = np.zeros((b*n,), dtype=np.cdouble)
		u_temp[-n:] = u[m-1]
		u[m-1] = u[m-1] - lu_Hm_ra[m-b-1].solve(u_temp)[-n:]
	for m in range(n-1, b, -1):
		A = up_A_ra[m-1]
		Au_temp = np.zeros((b*n,), dtype=np.cdouble)
		Au_temp[-n:] = A@u[m]
		u[m-1] = u[m-1] - lu_Hm_ra[m-b-1].solve(Au_temp)[-n:]
	Au = A_Fb1@u[b]
	uF = uF - lu_HF.solve(Au)
	for i in range(b):
		u[i] = uF[i*n:(i+1)*n]
	return u


def run_solver(n,b,wave_num,const,alpha):
	"""
	n: interior grid size
	b: width of PML in number of grid points
	wave_num: omega/2pi (avg wave number)
	const: appropriate positive constant for sigma1, sigma2
	alpha: small positive constant to adjust omega (paper uses alpha=2)
	"""

	omega = 2*np.pi*wave_num + 1j*alpha # angular frequency
	h = 1 / (n + 1) # spatial step size
	eta = b*h # width of PML in spatial dim
	r1 = .5
	r2 = .5
	f_mat = init_f1_mat(r1,r2,omega,n)
	f_vec = f_mat.flatten()
	c_mat = init_c1_mat(r1,1/8,n)

	A = build_A_matrix(b,const,eta,omega,h,n,c_mat)


	#--- test LDL factorization of A and plot the solution using algos 2.1, 2.2 -----
	"""
	T_ra, S_ra, L_ra, A_rebuilt = algo2_1(b,const,eta,omega,h,n,c_mat)
	u_solved = algo2_2(T_ra, S_ra, L_ra,const, eta, omega, b, h, n, c_mat, f_mat)
	fig = plt.figure()
	plt.imshow(np.flipud(np.real(u_solved)), extent=[0,1,0,1])
	t = 'N = ' + str(n) + '$^2$ \n $\omega /(2\pi)$ = ' + str(wave_num) \
		+ ' \n const = ' + str(const) + ' \n Real(u) \n Solution using Algo 2.1 and 2.2'
	plt.title(t)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.colorbar()
	plt.tight_layout()
	fig.subplots_adjust(left=-0.4)
	plt.show()
	# sys.exit()
	"""
	#------------------------------------------------------------------------------

	# use preconditioner:

	lu_HF, lu_Hm_ra = algo2_3(b,const,eta,omega,h,n,c_mat)

	# prepare A blocks to be used in the preconditioner
	A_b1F = get_A_b1F_block(b,const,eta,omega,h,n,c_mat)
	A_Fb1 = get_A_Fb1_block(b,const,eta,omega,h,n,c_mat)
	up_A_ra = []
	lo_A_ra = []
	for i in range(1,n):
		A_up = get_A_block(i,i+1,b,const,eta,omega,h,n,c_mat)
		A_lo = get_A_block(i+1,i,b,const,eta,omega,h,n,c_mat)
		up_A_ra.append(A_up)
		lo_A_ra.append(A_lo)


	M = scipy.sparse.linalg.LinearOperator((n**2,n**2), matvec=lambda x: \
			algo2_4(f_vec, b, n, lu_HF, A_b1F, A_Fb1, up_A_ra, lo_A_ra, lu_Hm_ra))


	u, exit_code = scipy.sparse.linalg.gmres(A, f_vec, M=M, tol=1e-3)

	u = np.flipud(u.reshape((n,n)))
	fig = plt.figure()
	plt.imshow(np.real(u), extent=[0,1,0,1])
	plt.xlabel('x')
	plt.ylabel('y')
	t = 'N = ' + str(n) + '$^2$ \n $\omega /(2\pi)$ = ' \
			+ str(wave_num) + ' \n const = ' + str(const) + ' \n Real(u)'
	plt.title(t)
	plt.colorbar()
	plt.tight_layout()
	fig.subplots_adjust(left=-0.4)
	plt.show()


if __name__ == "__main__":
	# run_solver(31,12,4,80,2)
	# run_solver(127,12,16,80,2)
	run_solver(255,12,32,50,2)
	# run_solver(1023,12,128,90,2) # 80 had large numbers

