import numpy as np

N = 16
K = 3

# make random data and model
y =     np.random.normal(size=(N,))
M =     np.random.normal(size=(N, K))
theta = np.random.normal(size=(K,))
mu =    np.random.normal(size=(K,))
print(y.shape, M.shape, theta.shape, mu.shape)

# cleverly make Hermitian covariances
ntrials = 32
Cvecs = np.random.normal(size=(N, ntrials))
C = Cvecs.dot(Cvecs.T)
Vvecs = np.random.normal(size=(K, ntrials))
V = Vvecs.dot(Vvecs.T)
Cinv = np.linalg.inv(C)
Vinv = np.linalg.inv(V)
print(C.shape, V.shape)

# make variance tensors for the product
Ainv = Vinv + M.T.dot(Cinv).dot(M)
A = np.linalg.inv(Ainv)
B = C + M.dot(V).dot(M.T)
Binv = np.linalg.inv(B)
print(A.shape, B.shape)

# determinant test:
LHS = np.linalg.slogdet(C)[1] + np.linalg.slogdet(V)[1]
RHS = np.linalg.slogdet(A)[1] + np.linalg.slogdet(B)[1]
print("logdets:", LHS, RHS, LHS - RHS)

# make means for the product
a = A.dot(M.T.dot(Cinv).dot(y) + Vinv.dot(mu))
b = M.dot(mu)
print(a.shape, b.shape)

# quadratic test:
LHS = (y - M.dot(theta)).T.dot(Cinv).dot(y - M.dot(theta)) \
    + (theta - mu).T.dot(Vinv).dot(theta - mu)
RHS = (theta - a).T.dot(Ainv).dot(theta - a) \
    + (y - b).T.dot(Binv).dot(y - b)
print("quadratics:", LHS, LHS, LHS - RHS)
