import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)


# --------------------------------------------------------------------
# Part 0: definitions
# --------------------------------------------------------------------

def fourier(f, t, N):
    return np.exp(1j * 2* np.pi * (f/N) * t)

def fourierBaseVec(f, N):
    return [fourier(f, t, N) for t in range(N)]

def fourierBase(N):
    return [fourierBaseVec(f, N) for f in range(N)]

def fourierBaseNp(N):
    return np.array(fourierBase(N)).transpose()


# --------------------------------------------------------------------
# Part 1: test if v_F = F^-1 v
# --------------------------------------------------------------------

N = 100
fourBase = fourierBaseNp(N)

v = np.array([1.2 * np.sin(0.1 * t) + 0.1*t for t in range(N)])

fourBaseInv = np.linalg.inv(fourBase)
v_F = fourBaseInv.dot(v)

v_F_correct = np.fft.fft(v) / N

v_reconstr = fourBase.dot(v_F)

# fig = plt.figure()
# ax0 = fig.add_subplot(141)
# ax0.set_title("v")
# ax0.plot(v)
# ax1 = fig.add_subplot(142)
# ax1.set_title("v_F")
# ax1.plot(np.real(v_F))
# ax2 = fig.add_subplot(143)
# ax2.set_title("v_F correct")
# ax2.plot(np.real(v_F_correct))
# ax3 = fig.add_subplot(144)
# ax3.set_title("v reconstructed")
# ax3.plot(v_reconstr)
# plt.show()



# -------------------------------------------------------------------------------
# Part 2: test if p conv q == ConvMatrix_q * p
# -------------------------------------------------------------------------------

# r(c) = p(a) conv q(b)
#      = ConvMatrix_q * p



def convMatrixNp(q):
    N = len(q)
    startoff = int(N/2)
    mtrx = np.zeros((N, N))
    for row in range(N):
        for col in range(N):
            indx = startoff + row - col
            if 0 <= indx < N:
                mtrx[row,col] = q[startoff + row - col]
    return mtrx




def p(a):
    return 3.1 * np.cos(0.2 * a) + 0.2

def q(b): 
    return 1.1 * np.sin(0.1 * b) + 0.4

pData = [p(a) for a in range(N)]
qData = [q(b) for b in range(N)]

convMtrx_q = convMatrixNp(qData)
r = convMtrx_q.dot(pData)

r_correct = np.convolve(pData, qData, 'same')


# fig = plt.figure()
# ax0 = fig.add_subplot(131)
# ax0.plot(r)
# ax1 = fig.add_subplot(132)
# ax1.plot(r_correct)
# ax2 = fig.add_subplot(133)
# ax2.plot(r - r_correct)
# plt.show()

# -------------------------------------------------------------------------------
# Part 3: test if ConvMatrix_q * p == p * FourBase^-1 * q
# -------------------------------------------------------------------------------

# r(c)  = p(a) conv q(b)
#       = ConvMatrix_q * p
#       = fourInv( four(p) * four(q) )
#       = FourBase * ( FourBase^-1 * p * FourBase^-1 * q )
#       = p * FourBase^-1 * q


r_convMtrx  = convMtrx_q.dot(pData)
r_fourInv   = fourBase.dot( fourBaseInv.dot(pData) * fourBaseInv.dot(qData) )
r_fourInv_correct = N * np.fft.ifft( (1/N) * np.fft.fft(pData) * (1/N) * np.fft.fft(qData) )

fig = plt.figure()
ax0 = fig.add_subplot(141)
ax0.set_title("p * q correct")
ax0.plot(r_correct)
ax1 = fig.add_subplot(142)
ax1.set_title("p * q conv with ConvMatrix")
ax1.plot(r_convMtrx)
ax2 = fig.add_subplot(143)
ax2.set_title("p * q conv with FourierMatrix")
ax2.plot(r_fourInv)
ax3 = fig.add_subplot(144)
ax3.set_title("p * q conv with np.fft")
ax3.plot(r_fourInv_correct)
plt.show()