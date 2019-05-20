include("Schollwock.jl")
using LinearAlgebra
using Arpack
using TensorOperations

W = Dict{Int,AbstractArray}()
function W_constructor(l::Int;j=-1.0,λ = 2)
    I = [1. 0.
        0. 1.]
    Z = [1. 0.
       0. -1.]
   for k in 2:l-1
       W[k] = zeros(Float64,(2,2,3,3))
       W[k][:,:,1,1] = I
       W[k][:,:,3,3] = I
       W[k][:,:,2,1] = Z
       W[k][:,:,3,1] = λ*Z
       W[k][:,:,3,2] = j*Z
   end
   W[1] = zeros(Float64,(2,2,1,3))
   W[1][:,:,1,1] = λ*Z
   W[1][:,:,1,2] = j*Z
   W[1][:,:,1,3] = I
   W[l] = zeros(Float64,(2,2,3,1))
   W[l][:,:,1,1] = I
   W[l][:,:,2,1] = Z
   W[l][:,:,3,1] = λ*Z
end
Le = 12
W_constructor(Le)

function W_constructor2(l::Int;j=-1.0,λ=2.0)
    Z = [1. 0.
       0. -1.]
    Hss = λ*Z
    Hnn = j*kron(Z,Z)
    return NNHandSSHtoMPO(Hnn,Hss,l)
end

function W_constructor3(l::Int;j=-1.0,λ=2.0)
    II = [1. 0.
        0. 1.]
    Z = [1. 0.
       0. -1.]
    Hss = λ*Z
    Hnn = j*kron(Z,Z)
    return NNHtoMPO(Hnn+kron(Hss,II)+kron(II,Hss),l)
end

# display(W[1])
# display(W_constructor2(Le)[1])
# println(norm(W[4][:,:,3,2]- W_constructor2(Le)[4][:,:,3,2]))
# display(W[4][:,:,3,2]*W[4][:,:,2,1])
# display(W_constructor2(Le)[4][:,:,3,2]*W_constructor2(Le)[4][:,:,2,1])
# gr1 = DMRG(W,Le,maxiteration=20,D=5)#,ψ0 = ψ₀)
# res = deepcopy((gr.tensors[1])[1,:,:])
# for i in 2:Le
#     global res
#     res = res*(gr.tensors[i])[1,:,:]
# end
# println(res*res'/inner(gr,gr))
# res = deepcopy((gr.tensors[1])[2,:,:])
# for i in 2:Le
#     global res
#     res = res*(gr.tensors[i])[2,:,:]
# end
# println(res*res'/inner(gr,gr))

gr2 = DMRG(W_constructor2(Le,j=1),Le,ϵ=1e-20,maxiteration=100,D=4)
gr2p = DMRG(W_constructor2(Le,j=1),Le,ϵ=1e-20,maxiteration=100,D=4)
println("First:")
mydump(gr2)
println("Second")
mydump(gr2p)
println(abs2(inner(gr2,gr2p)))
# gr3 = DMRG(W_constructor3(Le),Le,maxiteration=20,D=5)
# println("correct answer")
# mydump(gr1)
# println("NNHandSSHtoMPO")
# mydump(gr2)
# println("This should do nothing:")
# gr2p = CutEntanglement(gr2,Le-3,after=false)
# mydump(gr2p)
# println("NNHtoMPO")
# mydump(gr3)


#GHZ state MPS:
# function GHZ_constructor(l::Int)::MPS2
#     A₀ = [1.0 0.0; 0.0 0.0]
#     A₁ = [0.0 0.0; 0.0 1.0]
#     t = Dict{Int, Array{Float64,3}}()
#     for i in 2:l-1
#         t[i] = zeros(Float64,(2,2,2))
#         t[i][1,:,:] = A₀
#         t[i][2,:,:] = A₁
#     end
#     t[1] = zeros(Float64,(2,1,2))
#     t[1][1,:,:] = [1.0 0.0]
#     t[1][2,:,:] = [0.0 1.0]
#     t[l] = zeros(Float64,(2,2,1))
#     t[l][1,:,:] = [1.0; 0.0]
#     t[l][2,:,:] = [0.0; 1.0]
#     return MPS2(t,l,2,l,2)
# end
#
# println("GHZ before cut")
# mydump(GHZ_constructor(13))
# cut = CutEntanglement(GHZ_constructor(13),10,after=true)
# println("GHZ after cut")
# mydump(cut)
#
# println(inner(GHZ_constructor(13),cut))
