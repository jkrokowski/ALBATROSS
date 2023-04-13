from ufl import as_tensor,indices,Identity

#geometric dimension
d = 3
#indices
i,j,k,l=indices(4)
m,n=indices(2)
#d x d indentity matrix
delta = Identity(d)

#4th order tensor (shape 3x3x3x3)
C = as_tensor(delta[i,k]*delta[j,l] + delta[j,k]*delta[i,l],(i,j,k,l))
print(C.ufl_shape)

#2nd order tensor (shape 3x3)
Ci1k1 = as_tensor(C[i,0,k,0],(i,k))
print(Ci1k1.ufl_shape)

# #3rd order tensor desired (shape 3x3x2)
# Ci1kB = as_tensor(C[i,0,k,1:2],(i,k))
# print(Ci1kB)

D = as_tensor([[[C[i, 0, k, l] for l in range(1, 3)]
              for k in range(d)] for i in range(d)])
print(D.ufl_shape)

# Ci1k2 = as_tensor(C[i,0,k,1],(i,k))
# Ci1k3 = as_tensor(C[i,0,k,2],(i,k))
# Ci1kB = as_tensor([Ci1k2,Ci1k3])
# print(Ci1kB.ufl_shape)

# Ci2k1 = as_tensor(C[i,1,k,0],(i,k))
# Ci3k1 = as_tensor(C[i,2,k,0],(i,k))
# Ciak1 = as_tensor([Ci2k1,Ci3k1])
# print(Ciak1.ufl_shape)
# print(as_tensor(Ciak1[i,j,k],(j,i,k)).ufl_shape)

# C_test = Ciak1[i,j,k]*Ci1kB[l,m,n]
# print(C_test.ufl_shape)