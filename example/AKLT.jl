# AKLT ground state using DMRG

include("Schollwock.jl")
# include("AKLT-graph.jl")
using Plots

# function W_constructor(l::Int)
#     W = Dict{Int,Array{ComplexF64,4}}()
#     Z = Diagonal([1.0,0.0,-1.0])
#     Sm = diagm(-1=>[sqrt(2.0),sqrt(2.0)])
#     Sp = diagm(1=>[sqrt(2.0),sqrt(2.0)])
#     X = (Sm+Sp)/2.0
#     Y = (Sp-Sm)/2im
#     for k in 2:l-1
#         W[k] = zeros(Float64,(3,3,14,14))
#     end
# end
δ = 0.001
H = zeros(Float64,(9,9))
H[1,1] = 4.0/3.0 + δ^2
H[2,2] = 1.0/3.0
H[2,4] = 1.0
H[4,2] = 1.0
H[3,3] = -1.00/3.0 - δ^2
H[3,5] = 2.0/3.0
H[5,3] = 2.0/3.0
H[3,7] = 1.00/3.0
H[7,3] = 1.00/3.0
H[4,4] = 1.00/3.0
H[5,5] = 2.0/3.0
H[5,7] = 2.0/3.0
H[7,5] = 2.0/3.0
H[6,6] = 1.00/3.0 - δ^2
H[6,8] = 1.0
H[8,6] = 1.0
H[7,7] = -1.00/3.0
H[8,8] = 1.00/3.0
H[9,9] = 4.0/3.0 + δ^2

function InnerProductOfAdjoiningMPS(Le::Int,n::Int,BD::Int)
    mps_long = DMRG(NNHtoMPO(H,Le+n),Le+n,D=BD,σ=3,ϵ=1.0e-15,maxiteration=30)
    mps_short_temp = DMRG(NNHtoMPO(H,Le),Le,D=BD,σ=3,ϵ=1.0e-15,maxiteration=30)
    mps_cut = CutEntanglement(mps_long,Le,after=true)
    tensors = mps_short_temp.tensors
    for i in Le+1:Le+n
        σ,_,_ = size(mps_cut.tensors[i])
        tensors[i] = ones(typeof(mps_cut.tensors[i][1]),(σ,1,1))
    end
    mps_short = MPS2(tensors, Le+n,mps_short_temp.BD,Le,3)
    norm_of_MPS = sqrt(abs(inner(mps_short,mps_short)))
    mps_short.tensors[Le] = mps_short.tensors[Le]/norm_of_MPS
    return (mps_long.BD,abs2(inner(mps_short,mps_long)))
end

function main()
    l = 50
    D = 2
    local inners = Array{Float64,1}()
    for Le in 3:1:l
        try
            D,temp_inner = InnerProductOfAdjoiningMPS(Le,1,D)
            push!(inners,temp_inner)
        catch er
            write(stderr,"$(er)\n")
        end
    end
    plot(4:1:l+1,inners)
end
# M = DMRG(NNHtoMPO(H,21),21,D=2,σ=3)
# println(abs(inner(M,CutEntanglement(M,20,after=true))))
# println(abs(inner(M,CutEntanglement(M,21,after = false))))
# main()

GS1 = DMRG(NNHtoMPO(H,30),30,D=5,σ=3,ϵ=1.0e-15,maxiteration=100)
GS2 = DMRG(NNHtoMPO(H,30),30,D=5,σ=3,ϵ=1.0e-15,maxiteration=100)
println(inner(GS1,GS2))
println(abs2(inner(GS1,GS2)))
println(abs2(inner(GS1,GS1)))
