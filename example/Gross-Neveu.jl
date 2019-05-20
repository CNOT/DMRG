# Gross-Nevue ground state using DMRG

include("Schollwock.jl")
using DelimitedFiles
using Plots
using LsqFit
using DataFrames
using CSV

function Hss(m₀::Float64,g₀::Float64,r::Float64,n::Int;ϵ=0.0)
    a = 1.0/(n+1)
    a⁺ = [0.0 0.0;
          1.0 0.0]
    a⁻ = [0.0 1.0;
          0.0 0.0]
    Z = [1.0 0.0;
         0. -1.0]
    # MagneticField = ϵ*randn(Float64)*Diagonal([1.5, 0.5, -0.5, -1.5])
    # return im*(m₀-r/a)*(kron(a⁺,a⁻)-kron(a⁻,a⁺))+g₀^2/(2*a)*(kron(a⁺,a⁻)-kron(a⁻,a⁺))^2 + m₀*MagneticField
    return im*(-m₀-r/a)*(kron(a⁺,a⁻)-kron(a⁻,a⁺))+g₀^2/(2*a)*(kron(a⁺,a⁻)-kron(a⁻,a⁺))^2
end

function Hnn(r::Float64,n::Int)
    a = 1.0/(n+1)
    a⁺ = [0.0 0.0;
          1.0 0.0]
    a⁻ = [0.0 1.0;
          0.0 0.0]
    Z = [1.0 0.0;
         0. -1.0]
    II = [1. 0.;
          0. 1.]
    # return im/(2*a)*(-kron(II,a⁻,Z,a⁻)+kron(a⁻,Z,a⁻,II)+kron(II,a⁺,Z,a⁺)-kron(a⁺,Z,a⁺,II)+r*(kron(II,a⁻,a⁺,II)-kron(II,a⁺,a⁻,II)+kron(a⁺,Z,Z,a⁻)-kron(a⁻,Z,Z,a⁺)))
    return im/(2*a)*(kron(II,a⁻,Z,a⁺)-kron(a⁻,Z,a⁺,II)-kron(II,a⁺,Z,a⁻)+kron(a⁺,Z,a⁻,II)+r*(kron(II,a⁻,a⁺,II)-kron(II,a⁺,a⁻,II)+kron(a⁺,Z,Z,a⁻)-kron(a⁻,Z,Z,a⁺)))
end


#Making a Plot#

# function InnerOfAdjoiningMPS(Le::Int,n::Int,BD::Int,r::Float64,m0::Float64,g0::Float64) #Le: length of the smaller system; n: Number of differing sites; BD: Starting Bond dimension
#     mps_long = DMRG(NNHandSSHtoMPO(Hnn(r,Le+n),Hss(m0,g0,r,Le+n),Le+n),Le+n,D=BD,σ=4,ϵ=1.0e-8,maxiteration=30)
#     mps_short_temp = DMRG(NNHandSSHtoMPO(Hnn(r,Le),Hss(m0,g0,r,Le),Le),Le,D=BD,σ=4,ϵ=1.0e-8,maxiteration=30)
#     mps_cut = CutEntanglement(mps_long,Le+1,after=false)
#     tensors = mps_short_temp.tensors
#     for i in Le+1:Le+n
#         tensors[i] = mps_cut.tensors[i]
#     end
#     mps_short = MPS2(tensors, Le+n,mps_short_temp.BD,Le,4)
#     norm_of_MPS = sqrt(abs(inner(mps_short,mps_short)))
#     mps_short.tensors[Le] = mps_short.tensors[Le]/norm_of_MPS
#     return (mps_long.BD,abs(inner(mps_short,mps_long)))
# end

function AbsInnerProductOfMismatchingMPS(M1::MPS2,M2::MPS2)
    M1.length == M2.length && return inner(M1,M2)
    M1.length < M2.length ? (SM,BM) = (M1,M2) : (SM,BM) = (M2,M1)
    l1 = SM.length
    l2 = BM.length
    M_cut = CutEntanglement(BM,l1+1,after=false)
    tensors = SM.tensors
    for l in l1+1:l2
        tensors[l] = M_cut.tensors[l]
    end
    temp_MPS = MPS2(tensors)
    norm_of_MPS = sqrt(abs(inner(temp_MPS,temp_MPS)))
    temp_MPS.tensors[l2] = temp_MPS.tensors[l2]/norm_of_MPS
    return abs(inner(temp_MPS,BM))
end

function UniformInnerProductOfMPS(M1::MPS2,M2::MPS2)
    M1.length == M2.length && return inner(M1,M2)
    M1.length < M2.length ? (SM,BM) = (M1,M2) : (SM,BM) = (M2,M1)
    l1 = SM.length
    l2 = BM.length
    σ = SM.σ
    tensors = SM.tensors
    for l in l1+1:l2
        tensors[l] = ones(ComplexF64,σ,1,1)
    end
    temp_MPS = normalize(MPS2(tensors))
    return abs(inner(temp_MPS,BM))
end

function AddRandomSite(M::MPS2)
    σ = M.σ
    l = M.length
    tensors = M.tensors
    T = typeof(tensors[1][1])
    temp_tensor = rand(T,(σ,1,1))
    tensors[l+1] = temp_tensor/sum(abs2.(temp_tensor))
    Mt = MPS2(tensors,l+1,M.BD,M.CP,M.σ)
    norm_of_MPS = sqrt(abs(inner(Mt,Mt)))
    Mt.tensors[l+1] = Mt.tensors[l+1]/norm_of_MPS
    return Mt
end


function main1(args)
    l0 = parse(Int,args[1+1])
    l = parse(Int,args[2+1])
    r = parse(Float64,args[3+1])
    m0 = parse(Float64,args[4+1])
    g0 = parse(Float64,args[5+1])
    n = parse(Int,args[6+1])
    ϵ0 = parse(Float64,args[7+1])
    D = parse(Int,args[9])
    DMAX = parse(Int,args[10])
    local inners = Array{Float64,1}()
    t = time()
    local SmallGroundStates = Dict{Int,MPS2}()
    local ϵ_Small = zeros(Float64,n)
    local ϵ_Large = zeros(Float64,n)
    local BigGroundStates = Dict{Int,MPS2}()
    local DF_inners = DataFrame(Length = Int[],
                                ϵSmall = Float64[],
                                BD_Small = Int[],
                                ϵLarge = Float64[],
                                BD_Large = Int[],
                                InnerProduct = Float64[],
                                time = Float64[])
    for k in 1:n
        SmallGroundStates[k],ϵ_Small[k] = DMRG(NNHandSSHtoMPO(Hnn(r,l0),Hss(m0,g0,r,l0),l0),D=D,ϵ=ϵ0,maxiteration=3,DMax=DMAX)
    end
    open("Gross-Neveu_r=$(r)_m0=$(m0)_g0=$(g0)_l=$(l).txt","w") do io
        open("MAX_Gross-Neveu_r=$(r)_m0=$(m0)_g0=$(g0)_l=$(l).txt","w") do ioMAX
            for Le in l0+1:1:l
                t1 = time()
                for k in 1:n
                    BigGroundStates[k],ϵ_Large[k] = DMRG(NNHandSSHtoMPO(Hnn(r,Le),Hss(m0,g0,r,Le),Le),D=SmallGroundStates[k].BD,ϵ=ϵ0,maxiteration=3,DMax=DMAX)
                end
                push!(inners,UniformInnerProductOfMPS(SmallGroundStates[1],BigGroundStates[1]))
                for j in 1:n, k in 1:n
                    tmp = UniformInnerProductOfMPS(SmallGroundStates[j],BigGroundStates[k])
                    inners[end] = max(tmp,inners[end])
                    write(io,"$(Le) \t $(round(tmp,digits=5)) \t $(round(time()-t))\n")
                end
                write(ioMAX,"$(Le) \t $(round(inners[end],digits=5)) \t $(round(time()-t))\n")
                push!(DF_inners,(Le,ϵ_Small[1],SmallGroundStates[1].BD,ϵ_Large[1],BigGroundStates[1].BD,inners[end],round(time()-t1,digits=2)))
                SmallGroundStates = deepcopy(BigGroundStates)
                ϵ_Small = deepcopy(ϵ_Large)
            end
            write(ioMAX,"finished computation in $(time()-t).\n")
            CSV.write("Gross-Neveu_r=$(r)_m0=$(m0)_g0=$(g0)_l=$(l).csv",DF_inners)
        end
        write(io,"finished computation in $(time()-t).\n")
    end
    gr()
    p = plot(l0+1:l,inners)
    png("Gross-Neveu_r=$(r)_m0=$(m0)_g0=$(g0)_l=$(l).png")
end

function main2()
    l = parse(Int,ARGS[1+1])
    r = parse(Float64,ARGS[2+1])
    m0 = parse(Float64,ARGS[3+1])
    mf = parse(Float64,ARGS[4+1])
    dm = parse(Float64,ARGS[5+1])
    g0 = parse(Float64,ARGS[6+1])
    D = 25
    local inners = Array{Float64,1}()
    t = time()
    open("Gross-Neveu_Varying-Mass_r=$(r)_l=$(l)_g0=$(g0).txt","w") do io
        for m in m0:dm:mf
            local SmallGroundStates , _ = DMRG(NNHandSSHtoMPO(Hnn(r,l),Hss(m,g0,r,l),l),D=D,ϵ=1.0e-8,maxiteration=5)
            local BigGroundStates , _ = DMRG(NNHandSSHtoMPO(Hnn(r,l+1),Hss(m,g0,r,l+1),l+1),D=SmallGroundStates.BD,ϵ=1.0e-8,maxiteration=5)
            push!(inners,UniformInnerProductOfMPS(SmallGroundStates,BigGroundStates))
            write(io,"$(m) \t $(round(inners[end],digits=5)) \t $(round(time()-t))\n")
        end
        write(io,"finished computation in $(time()-t).\n")
    end
    gr()
    p = plot(m0:dm:mf,inners)
    png("Gross-Neveu_Varying-Mass_r=$(r)_l=$(l)_g0=$(g0).png")
end

function main3(args)
    l = parse(Int,args[1+1])
    r = parse(Float64,args[2+1])
    m0 = parse(Float64,args[3+1])
    rm = parse(Float64,args[4+1])
    n = parse(Int,args[5+1])
    g0 = parse(Float64,args[6+1])
    ϵ0 = parse(Float64,args[7+1])
    D = parse(Int,args[9])
    DMAX = parse(Int,args[10])
    local inners = Array{Float64,1}()
    local DF_inners = DataFrame(m = Float64[],
                                ϵ_Small = Float64[],
                                BD_Small = Int[],
                                ϵ_Large = Float64[],
                                BD_Large = Int[],
                                Inner = Float64[],
                                time = Int[])
    t = time()
    open("Gross-Neveu_Power-Mass2_r=$(r)_l=$(l)_g0=$(g0).txt","w") do io
        local m = m0
        for i in 0:n
            t1 = time()
            local SmallGroundStates,ϵsmall = DMRG(NNHandSSHtoMPO(Hnn(r,l),Hss(m,g0,r,l),l),D=D,ϵ=ϵ0,maxiteration=3,DMax=DMAX)
            local BigGroundStates,ϵbig = DMRG(NNHandSSHtoMPO(Hnn(r,l+1),Hss(m,g0,r,l+1),l+1),D=SmallGroundStates.BD,ϵ=ϵ0,maxiteration=3,DMax=DMAX)
            push!(inners,UniformInnerProductOfMPS(SmallGroundStates,BigGroundStates))
            write(io,"$(m) \t $(round(inners[end],digits=10)) \t $(round(time()-t))\n")
            push!(DF_inners,(m,ϵsmall,SmallGroundStates.BD,ϵbig,BigGroundStates.BD,inners[end],round(Int,time()-t1)))
            m = round(m*rm,digits = 10)
        end
        write(io,"finished computation in $(time()-t).\n")
        CSV.write("Gross-Neveu_Power-Mass2_r=$(r)_l=$(l)_g0=$(g0).csv",DF_inners)
    end
    gr()
    p = plot([m0*rm^i for i in 0:n],inners)
    png("Gross-Neveu_Power-Mass2_r=$(r)_l=$(l)_g0=$(g0).png")
end

function twopoint(j::Int,k::Int) #unnormalized
    Lm = [0.0 0.0;
          1.0 0.0]
    Lp = [0.0 1.0;
          0.0 0.0]
    Z = [1.0 0.0;
        0.0 -1.0]
    LmZ = kron(Lm,Z)
    ZZ = kron(Z,Z)
    ZLp = kron(Z,Lp)
    local O = Dict{Int,Array{Float64,2}}()
    O[j] = LmZ
    O[k] = ZLp
    for s in j+1:k-1
        O[s] = ZZ
    end
    return O
end

function main4(args)
    l = parse(Int,args[1+1])
    r = parse(Float64,args[2+1])
    m0 = parse(Float64,args[3+1])
    rm = parse(Float64,args[4+1]) #multiplicative coefficient for increasing m
    n = parse(Int,args[5+1]) #number of different masses
    g0 = parse(Float64,args[6+1])
    ϵg = parse(Float64,args[7+1]) #precision goal
    D = parse(Int,args[9])
    DMAX = parse(Int,args[10])
    m = m0
    local DF_inners = DataFrame(Distance = collect(0:l-1))
    open("Gross-Neveu_twopoint_l=$(l)_r=$(r)_g0=$(g0)_eps=$(ϵg).txt","w") do io
        t = time()
        for i in 1:n
            exps = Float64[]
            exps2 = ComplexF64[]
            local GS,ϵf = DMRG(NNHandSSHtoMPO(Hnn(r,l),Hss(m,g0,r,l),l),D=D,ϵ=ϵg,maxiteration=3,DMax=DMAX)
            write(io,"precision = $(ϵf)\nBondDimension = $(GS.BD)\nm = $(m)\nTwo point correlators:\n")
            for k in 1:l
                tp = twopoint(1,k)
                ex = (l-1)*1im*expectation(GS,GS,tp)
                write(io,"$(k):\t\t$(ex)\n")
                append!(exps,abs(ex))
                append!(exps2,ex)
            end
            DF_inners[Symbol("ydata_m=",round(m,digits=4))] = exps
            DF_inners[Symbol("ydata_error_m=",round(m,digits=4))] = 2*sqrt(ϵf)*exps
            DF_inners[Symbol("ActualExpectation_m=",round(m,digits=4))] = exps2
            DF_inners[Symbol("BD_m=",round(m,digits=4))] = repeat([GS.BD],l)
            DF_inners[Symbol("mdata_m=",round(m,digits=4))] = repeat([m],l)
            DF_inners[Symbol("ϵ_m=",round(m,digits=4))] = repeat([ϵf],l)
            @. model(x,p) = p[3] + p[2]*exp(-x/p[1])
            write(io,"model(x,p) = p[3] + p[2]*exp(-x/p[1])\n")
            try
                fit = curve_fit(model,0:l-1,exps,[1.0,1.0,0.0])
                write(io,"parameters:\n$(fit.param)\n")
                write(io,"Standard Deviations:\n$(stderror(fit))\n")
            catch
                write(io,"LsqFit Failed\n")
            end
            write(io,"============================================================\n")
            m = m*rm
        end
        CSV.write("Gross-Neveu_twopoint_l=$(l)_r=$(r)_g0=$(g0)_eps=$(ϵg).csv",DF_inners)
        write(io,"finished computation in $(time()-t).")
    end
end

function twopoint2(j::Int,k::Int) #unnormalized
    Lm = [0.0 0.0;
          1.0 0.0]
    I = [1.0 0.0;
         0.0 1.0]
    Lp = [0.0 1.0;
          0.0 0.0]
    Z = [1.0 0.0;
        0.0 -1.0]
    ILp = kron(I,-Lp)
    ZZ = kron(Z,Z)
    LmI = kron(Lm,I)
    local O = Dict{Int,Array{Float64,2}}()
    O[j] = ILp
    O[k] = LmI
    for s in j+1:k-1
        O[s] = ZZ
    end
    return O
end

function main5(args)
    l = parse(Int,args[1+1])
    r = parse(Float64,args[2+1])
    m0 = parse(Float64,args[3+1])
    rm = parse(Float64,args[4+1]) #multiplicative coefficient for increasing m
    n = parse(Int,args[5+1]) #number of different masses
    g0 = parse(Float64,args[6+1])
    ϵg = parse(Float64,args[7+1]) #precision goal
    D = parse(Int,args[9])
    DMAX = parse(Int,args[10])
    m = m0
    local DF_inners = DataFrame(Distance = collect(0:l-1))
    open("Gross-Neveu_twopoint2_l=$(l)_r=$(r)_g0=$(g0)_eps=$(ϵg).txt","w") do io
        t = time()
        for i in 1:n
            exps = Float64[]
            exps2 = ComplexF64[]
            local GS,ϵf = DMRG(NNHandSSHtoMPO(Hnn(r,l),Hss(m,g0,r,l),l),D=D,ϵ=ϵg,maxiteration=5,DMax=DMAX)
            write(io,"precision = $(ϵf)\nBondDimension = $(GS.BD)\nm = $(m)\nTwo point correlators 2:\n")
            for k in 1:l
                tp = twopoint2(1,k)
                ex = (l-1)*1im*expectation(GS,GS,tp)
                write(io,"$(k):\t\t$(ex)\n")
                append!(exps,abs(ex))
                append!(exps2,ex)
            end
            DF_inners[Symbol("ydata_m=",round(m,digits=4))] = exps
            DF_inners[Symbol("ydata_error_m=",round(m,digits=4))] = 2*sqrt(ϵf)*exps
            DF_inners[Symbol("ActualExpectation_m=",round(m,digits=4))] = exps2
            DF_inners[Symbol("BD_m=",round(m,digits=4))] = repeat([GS.BD],l)
            DF_inners[Symbol("mdata_m=",round(m,digits=4))] = repeat([m],l)
            DF_inners[Symbol("ϵ_m=",round(m,digits=4))] = repeat([ϵf],l)
            @. model(x,p) = p[3] + p[2]*exp(-x/p[1])
            write(io,"model(x,p) = p[3] + p[2]*exp(-x/p[1])\n")
            try
                fit = curve_fit(model,0:l-1,exps,[1.0,1.0,0.0])
                write(io,"parameters:\n$(fit.param)\n")
                write(io,"Standard Deviations:\n$(stderror(fit))\n")
            catch
                write(io,"LsqFit Failed\n")
            end
            write(io,"============================================================\n")
            m = m*rm
        end
        CSV.write("Gross-Neveu_twopoint2_l=$(l)_r=$(r)_g0=$(g0)_eps=$(ϵg).csv",DF_inners)
        write(io,"finished computation in $(time()-t).")
    end
end

function main6(args)
    l = parse(Int,args[1+1])
    r = parse(Float64,args[2+1])
    m0 = parse(Float64,args[3+1]) #The m₀ that we know has a good correlation length
    κ = parse(Float64,args[4+1]) #multiplicative span around m₀ we want to study(2 would be a good value)
    n = parse(Int,args[5+1]) #number of different masses
    g0 = parse(Float64,args[6+1])
    ϵg = parse(Float64,args[7+1]) #precision goal
    D = parse(Int,args[9])
    DMAX = parse(Int,args[10])
    rm = (κ)^(1/n)
    m = m0/κ
    local DF_TP = DataFrame(Distance = collect(0:l-1))
    local DF_fit = DataFrame(m0 = Float64[],
        Chi = Float64[],
        b = Float64[],
        error_Chi = Float64[],
        error_b = Float64[],
        inner50 = Float64[],
        error_inner50 = Float64[],
        entanglement1 = Float64[],
        entanglement2 = Float64[],
        entanglement3 = Float64[],
        entanglement4 = Float64[])
    open("Gross-Neveu_TwoPoint_l=$(l)_r=$(r)_g0=$(g0)_eps=$(ϵg)_m0=$(m0).txt","w") do io
        t = time()
        for i in 1:2n+1
            exps = Float64[]
            exps2 = ComplexF64[]
            local GS,ϵf = DMRG(NNHandSSHtoMPO(Hnn(r,l),Hss(m,g0,r,l),l),D=D,ϵ=ϵg,maxiteration=3,DMax=DMAX)
            write(io,"precision = $(ϵf)\nBondDimension = $(GS.BD)\nm = $(m)\nTwo point correlators:\n")
            for k in 1:l
                tp = twopoint(1,k)
                ex = (l-1)*1im*expectation(GS,GS,tp)
                write(io,"$(k):\t\t$(ex)\n")
                append!(exps,abs(ex))
                append!(exps2,ex)
            end
            Ms = GS.tensors
            D = GS.BD
            (σ,j,k) = size(Ms[l])
            F = svd(reshape(permutedims(Ms[l],(2,1,3)),(j,σ*k)))
            local S = F.S
            DF_TP[Symbol("InnerProduct_m=",round(m,digits=4))] = exps
            DF_TP[Symbol("InnerProduct_error_m=",round(m,digits=4))] = 2*sqrt(ϵf)*exps
            DF_TP[Symbol("ActualExpectation_m=",round(m,digits=4))] = exps2
            DF_TP[Symbol("BD_m=",round(m,digits=4))] = repeat([GS.BD],l)
            DF_TP[Symbol("m=",round(m,digits=4))] = repeat([m],l)
            DF_TP[Symbol("ϵ_m=",round(m,digits=4))] = repeat([ϵf],l)
            local small_GS,small_ϵf = DMRG(NNHandSSHtoMPO(Hnn(r,l-1),Hss(m,g0,r,l-1),l-1),D=GS.BD,ϵ=ϵg,maxiteration=3,DMax=DMAX)
            local inner = UniformInnerProductOfMPS(small_GS,GS)
            @. model(x,p) = p[2]*exp(-x/p[1])
            write(io,"model(x,p) =  p[2]*exp(-x/p[1])\n")
            try
                fit = curve_fit(model,2:l-1,exps[3:end],[1.0,1.0])
                write(io,"parameters:\n$(fit.param)\n")
                write(io,"Standard Deviations:\n$(stderror(fit))\n")
                push!(DF_fit,[m,fit.param...,stderror(fit)...,inner,hypot(ϵf,small_ϵf)*inner,S[1:4]...])
            catch e
                write(io,"LsqFit Failed\n")
                @warn e
            end
            write(io,"============================================================\n")
            m = m*rm
        end
        CSV.write("Gross-Neveu_TwoPoint_l=$(l)_r=$(r)_g0=$(g0)_eps=$(ϵg)_m0=$(m0).csv",DF_TP)
        CSV.write("Gross-Neveu_Fitting_l=$(l)_r=$(r)_g0=$(g0)_eps=$(ϵg)_m0=$(m0).csv",DF_fit)
        write(io,"finished computation in $(time()-t).")
    end
end

function Hss2(m₀::Float64,g₀::Float64,r::Float64,a::Float64;ϵ=0.0)
    a⁺ = [0.0 0.0;
          1.0 0.0]
    a⁻ = [0.0 1.0;
          0.0 0.0]
    Z = [1.0 0.0;
         0. -1.0]
    # MagneticField = ϵ*randn(Float64)*Diagonal([1.5, 0.5, -0.5, -1.5])
    # return im*(m₀-r/a)*(kron(a⁺,a⁻)-kron(a⁻,a⁺))+g₀^2/(2*a)*(kron(a⁺,a⁻)-kron(a⁻,a⁺))^2 + m₀*MagneticField
    return im*(-m₀-r/a)*(kron(a⁺,a⁻)-kron(a⁻,a⁺))+g₀^2/(2*a)*(kron(a⁺,a⁻)-kron(a⁻,a⁺))^2
end

function Hnn2(r::Float64,a::Float64)
    a⁺ = [0.0 0.0;
          1.0 0.0]
    a⁻ = [0.0 1.0;
          0.0 0.0]
    Z = [1.0 0.0;
         0. -1.0]
    II = [1. 0.;
          0. 1.]
    # return im/(2*a)*(-kron(II,a⁻,Z,a⁻)+kron(a⁻,Z,a⁻,II)+kron(II,a⁺,Z,a⁺)-kron(a⁺,Z,a⁺,II)+r*(kron(II,a⁻,a⁺,II)-kron(II,a⁺,a⁻,II)+kron(a⁺,Z,Z,a⁻)-kron(a⁻,Z,Z,a⁺)))
    return im/(2*a)*(kron(II,a⁻,Z,a⁺)-kron(a⁻,Z,a⁺,II)-kron(II,a⁺,Z,a⁻)+kron(a⁺,Z,a⁻,II)+r*(kron(II,a⁻,a⁺,II)-kron(II,a⁺,a⁻,II)+kron(a⁺,Z,Z,a⁻)-kron(a⁻,Z,Z,a⁺)))
end

function main7(args)
    l0 = parse(Int,args[2])
    l = parse(Int,args[3])
    r = parse(Float64,args[4])
    m0 = parse(Float64,args[5])
    g0 = parse(Float64,args[6])
    ϵ0 = parse(Float64,args[7])
    D = parse(Int,args[8])
    DMAX = parse(Int,args[9])
    local inners = Array{Float64,1}()
    t = time()
    local a = 1.0/(l+1)
    local DF_inners = DataFrame(Length = Int[],
                                ϵSmall = Float64[],
                                BD_Small = Int[],
                                ϵLarge = Float64[],
                                BD_Large = Int[],
                                InnerProduct = Float64[],
                                time = Float64[])
    local    SmallGroundStates,ϵ_Small = DMRG(NNHandSSHtoMPO(Hnn2(r,a),Hss2(m0,g0,r,a),l0),D=D,ϵ=ϵ0,maxiteration=3,DMax=DMAX)
    open("Fixed-Gross-Neveu_r=$(r)_m0=$(m0)_g0=$(g0)_l=$(l).txt","w") do io
        for Le in l0+1:1:l
            t1 = time()
            local    BigGroundStates,ϵ_Large = DMRG(NNHandSSHtoMPO(Hnn2(r,a),Hss2(m0,g0,r,a),Le),D=SmallGroundStates.BD,ϵ=ϵ0,maxiteration=3,DMax=DMAX)
            push!(inners,UniformInnerProductOfMPS(SmallGroundStates,BigGroundStates))
            write(io,"$(Le) \t $(round(inners[end],digits=5)) \t $(round(time()-t))\n")
            push!(DF_inners,(Le,ϵ_Small,SmallGroundStates.BD,ϵ_Large,BigGroundStates.BD,inners[end],round(time()-t1,digits=2)))
            SmallGroundStates = deepcopy(BigGroundStates)
            ϵ_Small = deepcopy(ϵ_Large)
        end
        CSV.write("Fixed-Gross-Neveu_r=$(r)_m0=$(m0)_g0=$(g0)_l=$(l).csv",DF_inners)
        write(io,"finished computation in $(time()-t).\n")
    end
end

function main()
    fun = parse(Int,ARGS[1])

    if fun == 1 #Inner products of ground states for varying lengths
        #l0,l,r,m0,g0,n=1,ϵ0,D(iniitial BD),DMAX
        main1(ARGS)
    elseif fun == 2
        main2()
    elseif fun == 3 # Inner producs for fixed length but varying mass as a power.
        # l ,r ,m0 ,rm ,n ,g0 ,ϵ0 ,D ,DMAX
        main3(ARGS)
    elseif fun == 4 # Two point function ψ1ψ̄j
        # l, r, m0, rm, n, g0, ϵg, D, DMAX
        main4(ARGS)
    elseif fun == 5 # Two point function ψ̄1ψj
        # l, r, m0, rm, n, g0, ϵg, D, DMAX
        main5(ARGS)
    elseif fun == 6 #calculating both correlation length and asymptotic value of inner products
        # l, r, m0, κ, n, g0, ϵg, D, DMAX
        main6(ARGS)
    elseif fun ==7 #Inner products of ground states for varying lengths with fixed lattice spacing
        #l0,l,r,m0,g0,ϵ0,D(iniitial BD),DMAX
        main7(ARGS)
    end
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    main()
end
