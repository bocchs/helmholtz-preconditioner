
# Alex Bocchieri

import numpy as np
from numba import jit
import scipy.linalg
import sys
import numpy
from scipy.linalg import lu
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
numpy.set_printoptions(threshold=sys.maxsize,precision=3,linewidth=np.inf)

@jit(nopython=True)
def sigma1(x):
	if x <= eta:
		return const / eta * ((x - eta)/eta)**2
	elif x >= 1 - eta:
		return const / eta * ((x - 1 + eta)/eta)**2
	else:
		return 0 


@jit(nopython=True)
def sigma2(x):
	if x <= eta:
		return const / eta * ((x - eta)/eta)**2
	else:
		return 0

@jit(nopython=True)
def s1(x):
	return (1 + 1j*sigma1(x)/omega)**-1


@jit(nopython=True)
def s2(x):
	return (1 + 1j*sigma2(x)/omega)**-1

def s2m(x,m):
	return (1 + 1j*sigma2(x-(m-b)*h)/omega)**-1

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
def get_A_diag_block(a,s2):
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
		c1 = 1/h**2 * (s1(x1) / s2(x2))
		if row_i >= 2:
			c1_vec[c1_idx] = c1
			c1_idx += 1

		x1 = (i+.5)*h
		x2 = j*h
		c2 = 1/h**2 * (s1(x1) / s2(x2))
		if row_i <= n - 1:
			c2_vec[c2_idx] = c2
			c2_idx += 1

		x1 = i*h
		x2 = (j-.5)*h
		c3 = 1/h**2 * (s2(x2) / s1(x1))

		x1 = i*h
		x2 = (j+.5)*h
		c4 = 1/h**2 * (s2(x2) / s1(x1))

		x1 = i*h
		x2 = j*h
		c5 = omega**2 / (s1(x1)*s2(x2)*c_mat[i-1,j-1]**2) - (c1 + c2 + c3 + c4)
		c5_vec[c5_idx] = c5
		c5_idx += 1

		row_i += 1

	A_block = scipy.sparse.diags(c5_vec) + scipy.sparse.diags(c1_vec,-1) \
				+ scipy.sparse.diags(c2_vec,1)
	return A_block



# Computes Hm: bn x bn A matrix for PML's b x n subgrid for layer m
def get_Hm(m):
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
	for j in range(1, b+1):
		for i in range(1, n+1):
			x1 = (i-.5)*h
			x2 = j*h
			c1 = 1/h**2 * (s1(x1) / s2m(x2,m))
			if row_i >= 2:
				c1_vec[c1_idx] = c1
				c1_idx += 1


			x1 = (i+.5)*h
			x2 = j*h
			c2 = 1/h**2 * (s1(x1) / s2m(x2,m))
			if row_i <= b*n - 1:
				c2_vec[c2_idx] = c2
				c2_idx += 1

			x1 = i*h
			x2 = (j-.5)*h
			c3 = 1/h**2 * (s2m(x2,m) / s1(x1))
			if row_i >= n+1:
				c3_vec[c3_idx] = c3
				c3_idx += 1

			x1 = i*h
			x2 = (j+.5)*h
			c4 = 1/h**2 * (s2m(x2,m) / s1(x1))
			if row_i <= b*n - n:
				c4_vec[c4_idx] = c4
				c4_idx += 1

			x1 = i*h
			x2 = j*h
			c5 = omega**2 / (s1(x1)*s2m(x2,m)*c_mat[i-1,j-1]**2) - (c1 + c2 + c3 + c4)
			c5_vec[c5_idx] = c5
			c5_idx += 1

			row_i += 1

	c1_vec[n-1::n] = 0
	c2_vec[n-1::n] = 0

	A = scipy.sparse.diags(c5_vec) + scipy.sparse.diags(c1_vec,-1) \
		+ scipy.sparse.diags(c2_vec,1) + scipy.sparse.diags(c3_vec, -n) \
		+ scipy.sparse.diags(c4_vec, n)
	return A



# computes desired n x n block of A_a,b that corresponds to row b of grid
# a,b = 1..n
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_block(a,b,s2):
	assert(a >= 1 and a <= n and b >= 1 and b <= n)
	if a == b:
		return get_A_diag_block(a,s2)
	if abs(a-b) > 1:
		return sparse.csr_matrix((n, n), dtype=np.complex) # zeros

	c4_vec = np.zeros((n,), dtype=np.cdouble)

	j = b
	x2 = (j+.5)*h
	for i in range(1, n+1):
		x1 = i*h
		c4 = 1/h**2 * (s2(x2) / s1(x1))
		c4_vec[i-1] = c4

	A_block = scipy.sparse.diags(c4_vec)
	return A_block


# computes bn x bn block of A_F,F that corresponds to first b rows of grid
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_FF_block(s2):
	# A_block_diags = np.zeros((b,n,n), dtype=np.cdouble)
	A_block_diags = b*[scipy.sparse.csr_matrix((n, n), dtype=np.complex)]
	# A_block_off_diags = np.zeros((b-1,n,n), dtype=np.cdouble)
	A_block_off_diags = (b-1)*[scipy.sparse.csr_matrix((n, n), dtype=np.complex)]
	for i in range(1,b+1):
		A_block_diags[i-1] = get_A_block(i,i,s2)
	for i in range(1,b):
		A_block_off_diags[i-1] = get_A_block(i,i+1,s2)
	# diag_blocks = scipy.linalg.block_diag(*A_block_diags)
	diag_blocks = scipy.sparse.block_diag(A_block_diags)
	diag_elems = diag_blocks.diagonal()
	off_diag_blocks = scipy.sparse.block_diag(A_block_off_diags)
	off_diag_elems = off_diag_blocks.diagonal()
	upper = scipy.sparse.diags(off_diag_elems,n)
	lower = scipy.sparse.diags(off_diag_elems,-n)
	middle = scipy.sparse.diags(diag_elems)
	return upper + middle + lower


# computes bn x n block of A_F,b+1 that corresponds to first b rows of grid
# take transpose of this result to get n x bn block of A_b+1,F
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_Fb1_block(s2):
	A_block = get_A_block(b,b+1,s2)
	return scipy.sparse.vstack((scipy.sparse.csr_matrix(((b-1)*n, n), dtype=np.cdouble), A_block))


# computes permutation matrix from row major to column major ordering for one PML
# subgrid of b layers
def get_P_mat():
	Pm = np.zeros((b*n,b*n))
	# for each column
	for i in range(n):
		# compute submatrix that corresponds to one column of grid
		for j in range(b):
			Pm[i*b+j, i+j*n] = 1
	Pm = scipy.sparse.csr_matrix(Pm)
	return Pm


def algo2_3():
	HF = get_A_FF_block(s2)
	PF = get_P_mat()
	PHP = PF @ HF @ PF.T
	lu = scipy.sparse.linalg.splu(PF@HF@PF.T)
	LF = lu.L
	UF = lu.U

	Hm_ra = []
	Lm_ra = []
	Um_ra = []
	Pm = get_P_mat()
	for i in range(n-b):
		Hm = get_Hm(i+b+1)
		Hm_ra.append(Hm)
		lu = scipy.sparse.linalg.splu(Pm@Hm@Pm.T)
		Lm_ra.append(lu.L)
		Um_ra.append(lu.U)
	return HF, LF, UF, Hm_ra, Lm_ra, Um_ra


def prec(f_vec):
	f_mat = f_vec.reshape((n,n))
	PF = get_P_mat()
	Pm = PF
	uF = np.zeros((b,n), dtype=np.cdouble)
	for i in range(b):
		uF[i] = f_mat[i]
	um_ra = np.zeros((n-b,n), dtype=np.cdouble)
	for i in range(n-b):
		um_ra[i] = f_mat[i+b]
	u = np.vstack((uF, um_ra))
	A_b1F = get_A_Fb1_block(s2).T
	TF = PF.T@scipy.sparse.linalg.inv(UF)@scipy.sparse.linalg.inv(LF)@PF
	TFuF = TF@(uF.reshape(-1))
	u[b] = u[b] - A_b1F@TFuF
	for m in range(b+1, n):
		A = get_A_block(m+1,m,s2)
		mat = Pm.T@scipy.sparse.linalg.inv(Um_ra[m-1-b])@scipy.sparse.linalg.inv(Lm_ra[m-1-b])@Pm
		u_temp = np.zeros((mat.shape[0],), dtype=np.cdouble)
		u_temp[-n:] = u[m-1]
		Tu = mat @ u_temp
		Tu = Tu[-n:]
		u[m] = u[m] - A@Tu
		print("first loop: " + str(m) + " / " + str(n-1))
	uF = TF@(uF.reshape(-1))
	for m in range(b+1, n+1):
		mat = Pm.T@scipy.sparse.linalg.inv(Um_ra[m-1-b])@scipy.sparse.linalg.inv(Lm_ra[m-1-b])@Pm
		u_temp = np.zeros((mat.shape[0],), dtype=np.cdouble)
		u_temp[-n:] = u[m-1]
		Tu = mat @ u_temp
		Tu = Tu[-n:]
		u[m-1] = Tu
		print("second loop: " + str(m) + " / " + str(n))
	for m in range(n-1, b, -1):
		A = get_A_block(m,m+1,s2)
		mat = Pm.T@scipy.sparse.linalg.inv(Um_ra[m-1-b])@scipy.sparse.linalg.inv(Lm_ra[m-1-b])@Pm
		Au_temp = np.zeros((mat.shape[0],), dtype=np.cdouble)
		Au_temp[-n:] = A@u[m]
		TAu = mat @ Au_temp
		TAu = TAu[-n:]
		u[m-1] = u[m-1] - TAu
		print("third loop: " + str(m) + " / " + str(b+1))
	A_Fb1 = get_A_Fb1_block(s2)
	uF = uF - TF@A_Fb1@u[b]
	for i in range(b):
		u[i] = uF[i*n:(i+1)*n]
	return u


def build_A_matrix():
	# A_FF = get_A_FF_block(s2)
	# A_Fb1_diag = get_A_Fb1_block(s2).diagonal(-n)
	# # A_b1F_diag = np.copy(A_Fb1_diag)
	# block_diags_ra = []
	# upper_block_diags_ra = []
	# lower_block_diags_ra = []
	# for i in range(b+1,n+1):
	# 	block_diags_ra.append(get_A_block(i,i,s2))
	# block_diag = scipy.sparse.block_diag(block_diags_ra)
	# for i in range(b+1,n):
	# 	upper_block_diag = get_A_block(i,i+1,s2).diagonal()
	# 	upper_block_diags_ra.append(upper_block_diag)

	# upper_block_diag = scipy.sparse.diags(np.concatenate([A_Fb1_diag,*upper_block_diags_ra]),b*n)
	# lower_block_diag = scipy.sparse.diags(np.concatenate([A_Fb1_diag,*upper_block_diags_ra]),-b*n)
	# diag = scipy.sparse.block_diag((A_FF, block_diag))
	# # mat1 = block_diag + upper_block_diag + lower_block_diag
	# # A = scipy.sparse.block_diag((A_FF, mat1))
	# A = diag + upper_block_diag + lower_block_diag

	# # A[:b*n,b*n:b*n+n] = A_Fb1
	# # A[b*n:b*n+n,:b*n] = A_b1F
	# print(np.real(A.toarray()))
	block_diags_ra = []
	off_diags_ra = [] # array of diagonals
	for i in range(1,n+1):
		block_diags_ra.append(get_A_block(i,i,s2))
	for i in range(1,n):
		off_diags_ra.append(get_A_block(i,i+1,s2).diagonal())
	block_diag = scipy.sparse.block_diag(block_diags_ra)
	off_diags = np.concatenate(off_diags_ra)
	upper = scipy.sparse.diags(off_diags,n)
	lower = scipy.sparse.diags(off_diags,-n)
	A = block_diag + upper + lower
	return A



if __name__ == "__main__":
	omega = 2*np.pi*16 # angular frequency
	const = 1 # appropriate positive constant for sigma1, sigma2

	n = 127 # int(.1*omega) # interior grid size, proportional to omega
	h = 1 / (n + 1) # spatial step size
	lam = 2 * np.pi / omega
	eta = lam # width of PML in spatial dim, typically around 1 wavelength
	b = 12 # int(h / eta) # width of PML in number of grid points

	u_mat = np.zeros((n,n))
	r1 = .5
	r2 = .5
	f_mat = init_f1_mat(r1,r2)
	c_mat = init_c1_mat(r1,r2)

	A = build_A_matrix()

	HF, LF, UF, Hm_ra, Lm_ra, Um_ra = algo2_3()

	f_vec = f_mat.flatten()
	M = LinearOperator((n**2,n**2), matvec=prec)
	u, exit_code = gmres(A, f_vec, M=M, tol=1e-3)
	if exit_code > 0:
		print("GMRES: convergence to tolerance not achieved")
	elif exit_code < 0:
		print("GMRES: illegal input or breakdown")
	
	u = u.reshape((n,n))


	# plt.imshow(np.abs(u))
	plt.imshow(np.real(u))
	# plt.show()

