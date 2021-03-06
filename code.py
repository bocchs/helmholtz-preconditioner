
import numpy as np
import scipy.linalg
import sys
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import scipy.sparse
from numba import jit
import time

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

# velocity field 2 described in paper
def init_c2_mat(n):
	x_i = np.linspace(0,1,n+2)
	xx, yy = np.meshgrid(x_i,x_i)
	c_mat = 4/3 * (1-.5*np.exp(-32*((xx-.5)**2)))
	return c_mat

# external force 1 described in paper
def init_f1_mat(r1,r2,omega,n):
	x_i = np.linspace(0,1,n+2) # n+2 points in each dimension, including boundary
	xx, yy = np.meshgrid(x_i[1:-1],x_i[1:-1]) # n interior points
	f_mat = np.exp(-(4*omega/np.pi)**2*((xx-r1)**2 + (yy-r2)**2))
	return f_mat

# external force 2 described in paper
def init_f2_mat(r1,r2,d1,d2,omega,n):
	x_i = np.linspace(0,1,n+2) # n+2 points in each dimension, including boundary
	xx, yy = np.meshgrid(x_i[1:-1],x_i[1:-1]) # n interior points
	f_mat = np.exp(-4*omega*((xx-r1)**2 + (yy-r2)**2)) \
			* np.exp(1j*omega*(xx*d1 + yy*d2))
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


# computes desired n x n block of A_row,col
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
	uF = np.zeros((b,n), dtype=np.cdouble)
	for i in range(b):
		uF[i] = f_vec[i*n:(i+1)*n]
	um_ra = np.zeros((n-b,n), dtype=np.cdouble)
	for i in range(n-b):
		um_ra[i] = f_vec[(i+b)*n:(i+b+1)*n]
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


# initialize c_mat and f_mat used in paper
# default arg values are those used in the paper
def init_c1_f1(omega, n, cr1=.5, cr2=.5, fr1=.5, fr2=.125):
	c_mat = init_c1_mat(cr1, cr2, n)
	f_mat = init_f1_mat(fr1, fr2, omega, n)
	return c_mat, f_mat

def init_c1_f2(omega, n, cr1=.5, cr2=.5, fr1=.125, fr2=.125, d1=1/2**.5, d2=1/2**.5):
	c_mat = init_c1_mat(cr1, cr2, n)
	f_mat = init_f2_mat(fr1, fr2, d1, d2, omega, n)
	return c_mat, f_mat

def init_c2_f1(omega, n, r1=.5, r2=.5):
	c_mat = init_c2_mat(n)
	f_mat = init_f1_mat(r1, r2, omega, n)
	return c_mat, f_mat

def init_c2_f2(omega, n, r1=.5, r2=.5, d1=1/2**.5, d2=1/2**.5):
	c_mat = init_c2_mat(n)
	f_mat = init_f2_mat(r1, r2, d1, d2, omega, n)
	return c_mat, f_mat


# https://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method
# counts number of gmres iterations
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))



def run_solver(n, b, wave_num, const, alpha, init_func=init_c1_f1, plot_solution=True):
	"""
	n: interior grid size
	b: width of PML in number of grid points
	wave_num: omega/2pi (avg wave number)
	const: appropriate positive constant for sigma1, sigma2
	alpha: small positive constant to adjust omega (paper uses alpha=2)
	init_func: function to initialize c_mat and f_mat described in paper
	"""

	# for counting number of gmres iterations with preconditioner
	counter_prec = gmres_counter(False)

	# for counting number of gmres iterations without preconditioner
	counter = gmres_counter(False)

	init_time_start = time.time()

	omega = 2*np.pi*wave_num + 1j*alpha # angular frequency
	h = 1 / (n + 1) # spatial step size
	eta = b*h # width of PML in spatial dim

	# initialize velocity field and external force
	c_mat, f_mat = init_func(omega, n)
	f_vec = f_mat.flatten()

	A = build_A_matrix(b,const,eta,omega,h,n,c_mat)

	# ----------- Test solution without using the preconditioner -------------
	"""
	u, exit_code = scipy.sparse.linalg.gmres(A, f_vec, tol=1e-3, callback=counter)
	print("GMRES iterations without preconditioner: " + str(counter.niter))
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
	# sys.exit()
	"""
	# ------------------------------------------------------------------------


	#--- Test LDL factorization of A and plot the solution using algos 2.1, 2.2 -----
	#--- Not used in preconditioner with moving PML ---
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


	#-------- Use preconditioner with moving PML (algos 2.3, 2.4) ----------

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


	init_time_end = time.time()

	u, exit_code = scipy.sparse.linalg.gmres(A, f_vec, M=M, tol=1e-3, callback=counter_prec)

	solve_time_end = time.time()

	print("GMRES iterations with preconditioner: " + str(counter_prec.niter))

	init_time_length = init_time_end - init_time_start
	solve_time_length = solve_time_end - init_time_end
	print("Initialization time = " + str(init_time_length))
	print("GMRES solve time = " + str(solve_time_length))

	if plot_solution:
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

	return init_time_length, solve_time_length


def plot_time(init_time_ra, solve_time_ra, n_ra):
	total_time_ra = init_time_ra + solve_time_ra
	N_ra = n_ra ** 2
	plt.plot(N_ra, solve_time_ra, 'b-o')
	plt.plot(N_ra, init_time_ra, 'g-o')
	plt.plot(N_ra, total_time_ra, 'r-o')
	# plt.plot(N_ra, [N for N in N_ra], 'k--' )
	plt.xlabel("N")
	plt.ylabel("Time (s)")
	plt.legend(["Solve Time", "Init Time", "Total Time"])
	plt.title("Runtime c2f2")
	plt.show()


if __name__ == "__main__":
	"""
	args order:
	n, b, omega/2pi, const, alpha, init_func, plot_solution

	note: N = n^2
	"""

	plot_soln = False

	# --- Run this without preconditioner to see how many gmres iters required vs with preconditioner ---
	# --- Have to uncomment code in run_solver() as well ---
	# run_solver(63, 12, 4, 61, 2, init_c1_f1, plot_soln)
	# sys.exit()


	init_time_127_c1f1, solve_time_127_c1f1 = run_solver(127, 12, 16, 81, 2, init_c1_f1, plot_soln)
	# init_time_127_c1f2, solve_time_127_c1f2 = run_solver(127, 12, 16, 61, 2, init_c1_f2, plot_soln)
	# init_time_127_c2f1, solve_time_127_c2f1 = run_solver(127, 12, 16, 80, 2, init_c2_f1, plot_soln)
	# init_time_127_c2f2, solve_time_127_c2f2 = run_solver(127, 12, 16, 84.2, 2, init_c2_f2, plot_soln)

	init_time_255_c1f1, solve_time_255_c1f1 = run_solver(255, 12, 32, 62, 2, init_c1_f1, plot_soln)
	# init_time_255_c1f2, solve_time_255_c1f2 = run_solver(255, 12, 32, 61, 2, init_c1_f2, plot_soln)
	# init_time_255_c2f1, solve_time_255_c2f1 = run_solver(255, 12, 32, 62, 2, init_c2_f1, plot_soln)
	# init_time_255_c2f2, solve_time_255_c2f2 = run_solver(255, 12, 32, 100, 2, init_c2_f2, plot_soln) # solution has large values

	init_time_511_c1f1, solve_time_511_c1f1 = run_solver(511, 12, 64, 81, 2, init_c1_f1, plot_soln)
	# init_time_511_c1f2, solve_time_511_c1f2 = run_solver(511, 12, 64, 62, 2, init_c1_f2, plot_soln)
	# init_time_511_c2f1, solve_time_511_c2f1 = run_solver(511, 12, 64, 63.5, 2, init_c2_f1, plot_soln)
	# init_time_511_c2f2, solve_time_511_c2f2 = run_solver(511, 12, 64, 85, 2, init_c2_f2, plot_soln) # couldn't get this one to work

	init_time_1023_c1f1, solve_time_1023_c1f1 = run_solver(1023, 12, 128, 100, 2, init_c1_f1, plot_soln)
	# init_time_1023_c1f2, solve_time_1023_c1f2 = run_solver(1023, 12, 128, 100.6, 2, init_c1_f2, plot_soln) # this gets killed on my Linux desktop
	# init_time_1023_c2f1, solve_time_1023_c2f1 = run_solver(1023, 12, 128, 100, 2, init_c2_f1, plot_soln)  # solution has large values
	# init_time_1023_c2f2, solve_time_1023_c2f2 = run_solver(1023, 12, 128, 100, 2, init_c2_f2, plot_soln) # couldn't get this one to work


	# --- plot runtimes ---
	n_ra = np.array([127, 255, 511, 1023])
	solve_time_c1f1_ra = np.array([solve_time_127_c1f1, solve_time_255_c1f1, solve_time_511_c1f1, solve_time_1023_c1f1])
	init_time_c1f1_ra = np.array([init_time_127_c1f1, init_time_255_c1f1, init_time_511_c1f1, init_time_1023_c1f1])
	plot_time(init_time_c1f1_ra, solve_time_c1f1_ra, n_ra)

	# solve_time_c1f2_ra = np.array([solve_time_127_c1f2, solve_time_255_c1f2, solve_time_511_c1f2])#, solve_time_1023_c1f2])
	# init_time_c1f2_ra = np.array([init_time_127_c1f2, init_time_255_c1f2, init_time_511_c1f2])#, init_time_1023_c1f2])
	# plot_time(init_time_c1f2_ra, solve_time_c1f2_ra, n_ra)

	# solve_time_c2f1_ra = np.array([solve_time_127_c2f1, solve_time_255_c2f1, solve_time_511_c2f1, solve_time_1023_c2f1])
	# init_time_c2f1_ra = np.array([init_time_127_c2f1, init_time_255_c2f1, init_time_511_c2f1, init_time_1023_c2f1])
	# plot_time(init_time_c2f1_ra, solve_time_c2f1_ra, n_ra)

	# solve_time_c2f2_ra = np.array([solve_time_127_c2f2, solve_time_255_c2f2, solve_time_511_c2f2, solve_time_1023_c2f2])
	# init_time_c2f2_ra = np.array([init_time_127_c2f2, init_time_255_c2f2, init_time_511_c2f2, init_time_1023_c2f2])
	# plot_time(init_time_c2f2_ra, solve_time_c2f2_ra, n_ra)

