
# Alex Bocchieri

import numpy as np
from numba import jit
import scipy.linalg
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize,precision=1,linewidth=np.inf)

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


# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
@jit(nopython=True)
def init_A(s2):
	c1_vec = np.zeros((n**2-1,), dtype=np.cdouble)
	c2_vec = np.zeros((n**2-1,), dtype=np.cdouble)
	c3_vec = np.zeros((n**2-n,), dtype=np.cdouble)
	c4_vec = np.zeros((n**2-n,), dtype=np.cdouble)
	c5_vec = np.zeros((n**2,), dtype=np.cdouble)

	c1_idx = 0
	c2_idx = 0
	c3_idx = 0
	c4_idx = 0
	c5_idx = 0
	row_i = 1 # 1..n^2
	for j in range(1, n+1):
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
			if row_i <= n**2 - 1:
				c2_vec[c2_idx] = c2
				c2_idx += 1

			x1 = i*h
			x2 = (j-.5)*h
			c3 = 1/h**2 * (s2(x2) / s1(x1))
			if row_i >= n+1:
				c3_vec[c3_idx] = c3
				c3_idx += 1

			x1 = i*h
			x2 = (j+.5)*h
			c4 = 1/h**2 * (s2(x2) / s1(x1))
			if row_i <= n**2 - n:
				c4_vec[c4_idx] = c4
				c4_idx += 1

			x1 = i*h
			x2 = j*h
			c5 = omega**2 / (s1(x1)*s2(x2)*c_mat[i-1,j-1]**2) - (c1 + c2 + c3 + c4)
			c5_vec[c5_idx] = c5
			c5_idx += 1

			row_i += 1

	A = np.diag(c5_vec) + np.diag(c1_vec,-1) + np.diag(c2_vec,1) + np.diag(c3_vec, -n) \
		+ np.diag(c4_vec, n)
	return A


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

	A_block = np.diag(c5_vec) + np.diag(c1_vec,-1) + np.diag(c2_vec,1)
	return A_block


# computes desired n x n block of A_a,b that corresponds to row b of grid
# a,b = 1..n
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_block(a,b,s2):
	assert(a >= 1 and a <= n and b >= 1 and b <= n)
	if a == b:
		return get_A_diag_block(a,s2)
	if abs(a-b) > 1:
		return np.zeros((n.n))

	c4_vec = np.zeros((n,), dtype=np.cdouble)

	j = b
	x2 = (j+.5)*h
	for i in range(1, n+1):
		x1 = i*h
		c4 = 1/h**2 * (s2(x2) / s1(x1))
		c4_vec[i-1] = c4

	A_block = np.diag(c4_vec)
	return A_block


# computes bn x bn block of A_F,F that corresponds to first b rows of grid
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_FF_block(s2):
	A_block_diags = np.zeros((b,n,n), dtype=np.cdouble)
	A_block_off_diags = np.zeros((b-1,n,n), dtype=np.cdouble)
	for i in range(1,b+1):
		A_block_diags[i-1] = get_A_block(i,i,s2)
	for i in range(1,b):
		A_block_off_diags[i-1] = get_A_block(i,i+1,s2)
	diag_blocks = scipy.linalg.block_diag(*A_block_diags)
	diag_elems = diag_blocks.diagonal()
	off_diag_blocks = scipy.linalg.block_diag(*A_block_off_diags)
	off_diag_elems = off_diag_blocks.diagonal()
	upper = np.diag(off_diag_elems,n)
	lower = np.diag(off_diag_elems,-n)
	middle = np.diag(diag_elems)
	return upper + middle + lower


# computes bn x n block of A_F,b+1 that corresponds to first b rows of grid
# take transpose of this result to get n x bn block of A_b+1,F
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_Fb1_block(s2):
	A_block = get_A_block(b,b+1,s2)
	return np.vstack((np.zeros(((b-1)*n, n), dtype=np.cdouble), A_block))







# ===============================================================
# PML subgrid








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

	A = np.diag(c5_vec) + np.diag(c1_vec,-1) + np.diag(c2_vec,1) + np.diag(c3_vec, -n) \
		+ np.diag(c4_vec, n)
	return A



# computes desired n x n block of A_a,b that corresponds to row b of grid
# a,b = 1..n
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_block(a,b,s2):
	assert(a >= 1 and a <= n and b >= 1 and b <= n)
	if a == b:
		return get_A_diag_block(a,s2)
	if abs(a-b) > 1:
		return np.zeros((n.n))

	c4_vec = np.zeros((n,), dtype=np.cdouble)

	j = b
	x2 = (j+.5)*h
	for i in range(1, n+1):
		x1 = i*h
		c4 = 1/h**2 * (s2(x2) / s1(x1))
		c4_vec[i-1] = c4

	A_block = np.diag(c4_vec)
	return A_block


# computes bn x bn block of A_F,F that corresponds to first b rows of grid
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_FF_block(s2):
	A_block_diags = np.zeros((b,n,n), dtype=np.cdouble)
	A_block_off_diags = np.zeros((b-1,n,n), dtype=np.cdouble)
	for i in range(1,b+1):
		A_block_diags[i-1] = get_A_block(i,i,s2)
	for i in range(1,b):
		A_block_off_diags[i-1] = get_A_block(i,i+1,s2)
	diag_blocks = scipy.linalg.block_diag(*A_block_diags)
	diag_elems = diag_blocks.diagonal()
	off_diag_blocks = scipy.linalg.block_diag(*A_block_off_diags)
	off_diag_elems = off_diag_blocks.diagonal()
	upper = np.diag(off_diag_elems,n)
	lower = np.diag(off_diag_elems,-n)
	middle = np.diag(diag_elems)
	return upper + middle + lower


# computes bn x n block of A_F,b+1 that corresponds to first b rows of grid
# take transpose of this result to get n x bn block of A_b+1,F
# s2: arg should be either s2 for main prob on full grid or s2m for aux prob on subgrid
def get_A_Fb1_block(s2):
	A_block = get_A_block(b,b+1,s2)
	return np.vstack((np.zeros(((b-1)*n, n), dtype=np.cdouble), A_block))


# computes permutation matrix from row major to column major ordering for one PML
# subgrid of b layers
def get_P_mat():
	Pm = np.zeros((b*n,b*n))
	# for each column
	for i in range(n):
		# compute submatrix that corresponds to one column of grid
		for j in range(b):
			Pm[i*b+j, i+j*n] = 1
	return Pm


def algo2_3():
	# Hm_ra = np.zeros((b,n,n), dtype=np.cdouble)
	# for m in range(b+1,n+1):
	# 	for j in range(m-b+1, m+1):
	# 		Hm = get_A_block(j,j)

	# 	Hm_ra = np.zeros((b,n,n), dtype=np.cdouble)
	# 	for i in range(b):
	# 		Hm = get_A_block(m,m,s2)
	# 		Hm_ra[i] = 
	Hm = get_Hm(b+1)
	Pm = get_P_mat()
	print(Hm.shape)
	print(Pm.shape)
	print(Hm)
	PHP = Pm @ Hm @ Pm.T
	return PHP


if __name__ == "__main__":
	omega = 1 # angular frequency
	const = 1 # appropriate positive constant for sigma1, sigma2

	n = 5 # int(.1*omega) # interior grid size, proportional to omega
	h = 1 / (n + 1) # spatial step size
	lam = 2 * np.pi / omega
	eta = lam # width of PML in spatial dim, typically around 1 wavelength
	b = 2 # int(h / eta) # width of PML in number of grid points

	u_mat = np.zeros((n,n))
	f_mat = np.zeros((n,n))
	c_mat = np.ones((n,n)) # velocity field

	# A = init_A()
	# A_21 = get_A_block(2,1)
	# print(A[n,0])
	# print(A[0,n])
	# print(A_21[0,0])
	# print()
	# print(A[n+1,1])
	# print(A[1,n+1])
	# print(A_21[1,1])


	# Aff = get_A_FF_block()
	# print(Aff)
	# print(Aff.shape)

	# Afb = get_A_Fb1_block()
	# print(Afb)
	# print(Afb.shape)

	# Pm = get_P_mat()
	# print(Pm)

	PHP = algo2_3()
	print()
	print(PHP)
	# x_str = np.array_repr(PHP)
	# print(x_str)

