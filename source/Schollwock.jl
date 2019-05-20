#=
Ideas and Todo list:
✔ 1- Find the bug in eigen values at the beginning of left Sweep
✔ 1.1- Check with AKLT as well
✔ 1.2- Run DMRG on Gross-Neveu model
✔ 2- Update the code for 1.0
  2.1- The matrix we're solving should be explicitly Hermitian
  2.2- Fix RC and LC functions
✔ 3- Add MPO and compatibility for it in DMRG code
✔ 3.1- Automate DMRG so it increases bond dimension automaticaly after it hasn't converged after a threshold number of sweeps
  4- Clean up the code
  4.1- Use Modules and `export`
  5- Optimize the code for when to use 'eigs' or 'eigen'
  6- Look into StaticArray and add compatibility where it's a good idea.
✔ 7- Rewrite everything that's MPS2 into MPS and then switch them.
  8- Add support for tDMRG
=#

using TensorOperations
using Arpack
using LinearAlgebra
using ZChop

mutable struct MPS2
    matrices::Dict{Tuple,AbstractArray}
    length::Int #(*number of lattice sites*)
    BD::Int #(*Bond Dimension *)
    CP::Int #(*Cross-Point, it determines if the MPS2 is left-normalized(CP=length) or right-normalized(CP=1) or half of each*)
    σ::Int  # degrees of freedom per site
end

function MPS2(D::Dict{Tuple,AbstractArray})
    BD = max(max(size.(values(D))...)...)
    σ = max(last.(keys(D))...)+1
    l = max(first.(keys(D))...)
    return MPS2(D,l,BD,0,σ)
end

mutable struct MPS #This definition of MPS uses tensors on each site
    tensors::Dict{Int,AbstractArray}
    length::Int
    BD::Int
    CP::Int
    σ::Int
end

function MPS(t::Dict{Int,AbstractArray})
    BD = max(last.(size.(values(t)))...)
    σ = max(first.(size.(values(t)))...)
    return MPS(t,length(t),BD,0,σ)
end

mutable struct MPO
    tensors::Dict{Int,Array{T,4}} where T<:Number
    length::Int
    BD::Int
    σ1::Int
    σ2::Int
end

function MPO(t::Dict{Int,Tensor})  where Tensor<:AbstractArray{T,4} where T<:Number
    BD = max(last.(size.(values(t)))...)
    σ1 = max(first.(size.(values(t)))...)
    σ2 = max(map(x->x[2],size.(values(t)))...)
    return MPO(t,length(t),BD,σ1,σ2)
end

import Base.print
function print(A::Dict)
    for k in keys(A)
        println(k," => ",A[k])
    end
end


function mydump(g::MPS)
    Le = g.length
    σ = g.σ
    try
        Le*log2(σ) > 25 && throw("Too big to print")
    catch
        warn("MPS is too big to display")
        return
    end
    for i in 0:σ^Le-1
        j = deepcopy(i)
        (j,b) = divrem(j,σ)
        res = deepcopy((g.tensors[1])[b+1,:,:])
        for k in 2:Le
            (j,b) = divrem(j,σ)
            res = res*(g.tensors[k])[b+1,:,:]
        end
        res = real((res*res')[1])
        if res>10.0^(-5.0)
            println(digits(i,base=σ,pad=Le)," : ",res)
        end
    end
end


function MPS(Q::MPS2)
    L = Q.length
    d = Q.σ
    Qm = Q.matrices
    Ms = Dict{Int,AbstractArray}()
    for l in 1:L
        s = size(Qm[(l,0)])
        # println(length(s))
        dz = length(s)==2 ? zeros(Float64,(d,s[1],s[2])) : zeros(Float64,(d,s[1],1))
        # println(dz," ",size(dz))
        for sig in 1:d
            dz[sig,:,:] = Qm[(l,sig-1)]
        end
        Ms[l] = copy(dz)
    end
    return MPS(Ms,L,Q.BD,Q.CP,d)
end


function MPS2(Q::MPS)
    Ms = Dict{Tuple,AbstractArray}()
    d = Q.σ
    l = Q.length
    Qt = Q.tensors
    for i in 1:l
        for σ in 1:d
            Ms[(i,σ-1)] = Qt[i][σ,:,:]
        end
    end
    return MPS2(Ms,l,Q.BD,Q.CP,d)
end

function IncreaseBondDimension(M::MPS,D::Int)
    l = M.length
    σ = M.σ
    t = M.tensors
    T = typeof(t[1][1])
    newM = Dict{Int,Array}()
    newM[1] = zeros(T,(σ,1,D))
    newM[1][1:σ,1,1:size(t[1])[3]] = t[1]
    newM[l] = zeros(T,(σ,D,1))
    newM[l][1:σ,1:size(t[l])[2],1] = t[l]
    for i in 2:l-1
        newM[i] = zeros(T,(σ,D,D))
        newM[i][1:σ,1:size(t[i])[2],1:size(t[i])[3]] = t[i]
    end
    return MPS(newM,l,D,M.CP,σ)
end


function VonNeumannEntropy(ρ::AbstractMatrix)
    s = svdfact(ρ)[:S]
    return sum(x-> x>0 ? -x^2 * log2(x^2) :  0,s)
end

function MPS2FromPsi(ψ::AbstractArray,d::Int,n::Int;cp=n)
    length(ψ)≠d^n&&throw(error("LengthMismatch"))
    if cp == n
        As = Dict{Tuple,AbstractArray}()
        Ψ = reshape(ψ,d,:)
        F = svd(Ψ)
        U,S,V = (F.U,F.S,F.V)
        S = filter(x->abs(x)>eps(Float64),S)
        rl = length(S)
        BDlist = [rl]
        U = U[:,1:rl]
        V = V[:,1:rl]
        for i in 1:d
            As[(1,i-1)]=U[i,:]
        end
        Ψ = Diagonal(S)*V'
        for l in 2:n
            Ψ = reshape(Ψ,rl*d,:)
            F = svd(Ψ)
            U,S,V = (F.U,F.S,F.V)
            S = filter(x->abs(x)>eps(Float64),S)
            rln = length(S)
            push!(BDlist,rln)
            U = U[:,1:rln]
            V = V[:,1:rln]
            for i in 1:d
                As[(l,i-1)]=reshape(U[i,:],(rl,rln))
            end
            Ψ = Diagonal(S)*V'
            rl = copy(rln)
        end
        return MPS2(As,n,max(BDlist...),cp,d)
    elseif cp==1
        Bs = Dict{Tuple,AbstractArray}()
        Ψ = reshape(ψ,:,d)
        F = svd(Ψ)
        U,S,V = (F.U,F.S,F.V)
        S = filter(x->abs(x)>eps(Float64),S)
        rl = length(S)
        BDlist = [rl]
        U = U[:,1:rl]
        V = V[:,1:rl]
        for i in 1:d
            Bs[(n,i-1)]=V[:,i]'
        end
        Ψ = U*Diagonal(S)
        for l in n-1:-1:1
            Ψ = reshape(Ψ,:,rl*d)
            F = svd(Ψ)
            U,S,V = (F.U,F.S,F.V)
            S = filter(x->abs(x)>eps(Float64),S)
            rln = length(S)
            push!(BDlist,rln)
            U = U[:,1:rln]
            V = V[:,1:rln]
            for i in 1:d
                Bs[(l,i-1)]=reshape(V[:,i]',rln,rl)
            end
            Ψ = U*Diagonal(S)
            rl = copy(rln)
        end
        return MPS2(Bs,n,max(BDlist...),cp)
    else
        As = Dict{Tuple,AbstractArray}()
        Ψ = reshape(ψ,d,:)
        F = svd(Ψ)
        U,S,V = (F.U,F.S,F.V)
        S = filter(x->abs(x)>eps(Float64),S)
        rl = length(S)
        BDlist=[rl]
        U = U[:,1:rl]
        V = V[:,1:rl]
        for i in 1:d
            As[(1,i-1)]=U[i,:]
        end
        Ψ = Diagonal(S)*V'
        for l in 2:cp
            Ψ = reshape(Ψ,rl*d,:)
            F = svd(Ψ)
            U,S,V = (F.U,F.S,F.V)
            S = filter(x->abs(x)>eps(Float64),S)
            rln = length(S)
            push!(BDlist,rln)
            U = U[:,1:rln]
            V = V[:,1:rln]
            for i in 1:d
                As[(l,i-1)]=reshape(U[i,:],rl,rln)
            end
            Ψ = Diagonal(S)*V'
            rl = copy(rln)
        end
        # Ψ = V'
        Ψ = reshape(ψ,:,d)
        F = svd(Ψ)
        U,S,V = (F.U,F.S,F.V)
        S = filter(x->abs(x)>eps(Float64),S)
        rl = length(S)
        BDlist2=[rl]
        U = U[:,1:rl]
        V = V[:,1:rl]
        for i in 1:d
            As[(n,i-1)]=V[:,i]'
        end
        Ψ = U*Diagonal(S)
        for l in n-1:-1:cp
            Ψ = reshape(Ψ,:,rl*d)
            F = svd(Ψ)
            U,S,V = (F.U,F.S,F.V)
            S = filter(x->abs(x)>eps(Float64),S)
            rln = length(S)
            push!(BDlist2,rln)
            U = U[:,1:rln]
            V = V[:,1:rln]
            for i in 1:d
                As[(l,i-1)]=reshape(V[:,i]',rln,rl)
            end
            Ψ = U*Diagonal(S)
            rl = copy(rln)
        end
        return MPS2(As,n,max(BDlist...,BDlist2...),cp)
    end
end

# Here's a sanity check to see the MPS2 works.
# ψ = [i==0&&j==0&&k==0&&l==0?1:0 for i in 0:2, j in 0:2, k in 0:2, l in 0:2]
# AMatrices = MPS2FromPsi(ψ,3,4)
# prod(i->AMatrices.matrices[i,0], 1:4)
# ψ = [0.25 for i in 0:1, j in 0:1, k in 0:1, l in 0:1]
# AMatrices = MPS2FromPsi(ψ,2,4)
# prod(i->AMatrices.matrices[i,1], 1:4)
# typeof(AMatrices)

function inner(Mψ::MPS2,Mϕ::MPS2,σ::Int,n::Int)
    M=1.
    # C=Array{Any}(0) #these are defined for house keeping. They are useful in DMRG calculations.
    # push!(C,M)
    for i in 1:n
        M=sum(d->Mψ.matrices[i,d-1]'*(M*Mϕ.matrices[i,d-1]),1:σ)
        # push!(C,M)
    end
    try
        size(M)!=(1,1) && throw(error("Bad Size"))
    catch e
        write(stderr,"Possible Bad Size")
    end
    return M
end

# ψ = [0.25 for i in 0:1, j in 0:1, k in 0:1, l in 0:1]
# AMatrices = MPS2FromPsi(ψ,2,4)
# println(inner(AMatrices,AMatrices,2,4))

function inner(Mψ::MPS,Mϕ::MPS)
    M=reshape([1.],(1,1))
    n = Mψ.length
    Mψt = Mψ.tensors
    Mϕt = Mϕ.tensors
    for i in 1:n
        Mψti = Mψt[i]
        Mϕti = Mϕt[i]
        @tensor M[a,b]:= conj(Mψti[σ,α,a])*M[α,β]*Mϕti[σ,β,b]
    end
    size(M)!=(1,1) && throw(error("Bad Size"))
    return M[1,1]
end

function normalize_fast(M::MPS)
    norm_MPS = sqrt(abs(inner(M,M)))
    M.tensors[1] /= norm_MPS
    return M
end

import LinearAlgebra.normalize
function normalize(M::MPS)
    norm_MPS = sqrt(abs(inner(M,M)))
    for i in 1:M.length
        M.tensors[i] /= norm_MPS^(1/M.length)
    end
    return M
end

function expectation(Mψ::MPS,Mϕ::MPS,Os::Dict{Int,Array{T,2}} where T)
    Mψ.length != Mϕ.length ? throw(ErrorException("length mismatch")) : l = Mψ.length
    Mψt = Mψ.tensors
    Mϕt = Mϕ.tensors
    C = reshape([1.0],(1,1))
    for i in 1:l
        tempψ = Mψt[i]
        tempϕ = Mϕt[i]
        if i in keys(Os)
            O_temp = Os[i]
            @tensor C[a,b] := conj(tempϕ[σp,α,a])*C[α,β]*O_temp[σp,σ]*tempψ[σ,β,b]
        else
            @tensor C[a,b] := conj(tempϕ[σ,α,a])*C[α,β]*tempψ[σ,β,b]
        end
    end
    size(C) != (1,1) && throw(error("Bad Size"))
    return C[1,1]
end

function expectation(Mψ::MPS2,Mϕ::MPS2,Os::Dict{Int,T} where T <: AbstractArray,σ::Int)
    Mψ.length ≠ Mϕ.length  ? throw(ErrorException("length mismatch")) : n=Mψ.length
    @tensor CM[alm,alp,σp]:=C[alm,almp]*Mψ.matrices[1,σp][almp,alp]
    @tensor OCM[σ,alm,alp]:=CM[alm,alp,σ]
    @tensor C[al,alp]:=Mϕ.matrices[1,σ][alm,al]'*OCM[σ,alm,alp]
    Cs = [C] # these are defined for house keeping and DMRG
    for i in 2:n
        O = Os[i]
        @tensor CM[alm,alp,σp]:=C[alm,almp]*Mψ.matrices[i,σp][almp,alp]
        @tensor OCM[σ,alm,alp]:=O[σ,σp]*CM[alm,alp,σp]
        @tensor C[al,alp]:=Mϕ.matrices[i,σ][alm,al]'*OCM[σ,alm,alp]
        push!(Cs,C)
    end
    return C
end


function LC(M::MPS) #LeftCanonical form (Eq. 136 from paper)
    As = Dict{Int,AbstractArray}()
    DummyM = reshape([reshape(M.tensors[1][σ,:,:],(1,:)) for σ in 1:M.σ],(M.σ,:))
    F = svd(DummyM)
    U,S,V = (F.U,F.S,F.V)
    S = filter(x->abs(x)>eps(Float64),S)
    rl = length(S)
    U = U[:,1:rl]
    V = V[:,1:rl]
    As[1] = reshape(U,(M.σ,1,rl))
    for i in 2:M.length
        Mt = M.tensors[i]
        @tensor DummyM[σ,s,a] := Diagonal(S)[s,s1]*V[s1,a1]'*Mt[σ,a1,a]
        F = svd(reshape(DummyM,(M.σ*rl,:)))
        U,S,V = (F.U,F.S,F.V)
        S = filter(x->abs(x)>eps(Float64),S)
        rln = length(S)
        U = U[:,1:rln]
        V = V[:,1:rln]
        As[i] = reshape(U,(M.σ,rl,rln))
        rl = rln
    end
    return MPS(As,M.length,M.BD,M.length,M.σ)
end

function LC(M::MPS,D::Int) #LeftCanonical form with a limited Bond Dimension
    As = Dict{Int,AbstractArray}()
    DummyM = reshape([reshape(M.tensors[1][σ,:,:],(1,:)) for σ in 1:M.σ],(M.σ,:))
    F = svd(DummyM)
    U,S,V = (F.U,F.S,F.V)
    S = filter(x->abs(x)>eps(Float64),S)
    rl = min(D,length(S))
    U = U[:,1:rl]
    V = V[:,1:rl]
    As[1] = reshape(U,(M.σ,1,rl))
    for i in 2:M.length
        Mt = M.tensors[i]
        @tensor DummyM[σ,s,a] := Diagonal(S)[s,s1]*V[s1,a1]'*Mt[σ,a1,a]
        F = svd(reshape(DummyM,(M.σ*rl,:)))
        U,S,V = (F.U,F.S,F.V)
        S = filter(x->abs(x)>eps(Float64),S)
        rln = min(D,length(S))
        S = S[1:rln]
        U = U[:,1:rln]
        V = V[:,1:rln]
        As[i] = reshape(U,(M.σ,rl,rln))
        rl = rln
    end
    return MPS(As,M.length,D,M.length,M.σ)
end

function RC(M::MPS) #RightCanonical form (Eq. 137)
    Bs = Dict{Int,AbstractArray}()
    l = M.length
    (d1,d2,d3) = size((M.tensors)[l])
    DummyM = permutedims(M.tensors[l],(2,1,3))
    DummyM = reshape(DummyM,(d2,:))
    F = lq(DummyM)
    L,Q =(Array(F.L),Array(F.Q))
    Bs[l] = permutedims(reshape(Q,(:,d1,d3)),(2,1,3))
    for i in l-1:-1:1
        Mt = M.tensors[i]
        @tensor DummyM[σ,a,s] := Mt[σ,a,a1]*L[a1,s]
        (d1,d2,d3) = size(DummyM)
        F = lq(reshape(permutedims(DummyM,(2,1,3)),(d2,:)))
        L,Q =(Array(F.L),Array(F.Q))
        Bs[i] = permutedims(reshape(Q,(:,d1,d3)),(2,1,3))
    end
    return MPS(Bs,M.length,M.BD,1,M.σ)
end

function RC(M::MPS,D::Int) #RightCanonical form with a limited Bond Dimension
    Bs = Dict{Int,AbstractArray}()
    l = M.length
    DummyM = reshape(M.tensors[l],(:,M.σ))
    F = svd(DummyM)
    U,S,V = (F.U,F.S,F.V)
    # S = filter(x->abs(x)>eps(Float64),S)
    rl = min(D)#,length(S))
    U = U[:,1:rl]
    V = V[:,1:rl]
    Bs[l] = reshape(V',(M.σ,rl,1))
    for i in l-1:-1:1
        Mt = M.tensors[i]
        @tensor DummyM[σ,a,s] := Mt[σ,a,a1]*U[a1,s1]*Diagonal(S)[s1,s]
        F = svd(reshape(DummyM,(:,M.σ*rl)))
        U,S,V = (F.U,F.S,F.V)
        # S = filter(x->abs(x)>eps(Float64),S)
        rln = min(D)#,length(S))
        # println(S)
        S = S[1:rln]
        U = U[:,1:rln]
        V = V[:,1:rln]
        Bs[i] = reshape(V',(M.σ,rln,rl))
        rl = rln
    end
    return MPS(Bs,M.length,D,1,M.σ)
end


function L(M1::MPS,M2::MPS,k::Int) #This is a helper function used in Iterative Compression
    M1t = M1.tensors
    M2t = M2.tensors
    @tensor DummyM[a1,b1] := M2t[1][σ,α,a1]'*M1t[1][σ,α,b1]
    for i in 2:k
        M2ti = M2t[i]
        M1ti = M1t[i]
        @tensor DummyM[a,b] := M2ti[σ,α,a]'*DummyM[α,β]*M1ti[σ,β,b]
    end
    return DummyM
end
function L(M1t::Dict,M2t::Dict,k::Int) #This is a helper function used in Iterative Compression
    @tensor DummyM[a1,b1] := M2t[1][σ,α,a1]'*M1t[1][σ,α,b1]
    for i in 2:k
        M2ti = M2t[i]
        M1ti = M1t[i]
        @tensor DummyM[a,b] := M2ti[σ,α,a]'*DummyM[α,β]*M1ti[σ,β,b]
    end
    return DummyM
end
function R(M1::MPS,M2::MPS,k::Int) #This is a helper function used in Iterative Compression
    M1t = M1.tensors
    M2t = M2.tensors
    l = M1.length
    M2.length≠l&&throw(error("LengthMismatch"))
    M2tl = M2t[l]
    M1tl = M1t[l]
    @tensor DummyM[a1,b1] := M2tl[σ,a1,α]'*M1tl[σ,b1,α]
    for i in l-1:-1:k
        M2ti = M2t[i]
        M1ti = M1t[i]
        @tensor DummyM[a,b] := M2ti[σ,a,α]'*DummyM[α,β]*M1ti[σ,b,β]
    end
    return DummyM
end
function R(M1t::Dict,M2::MPS,k::Int) #This is a helper function used in Iterative Compression
    M2t = M2.tensors
    l = M1.length
    M2tl = M2t[l]
    M1tl = M1t[l]
    @tensor DummyM[a1,b1] := M2tl[σ,a1,α]'*M1tl[σ,b1,α]
    for i in l-1:-1:k
        M2ti = M2t[i]
        M1ti = M1t[i]
        @tensor DummyM[a,b] := M2ti[σ,a,α]'*DummyM[α,β]*M1ti[σ,b,β]
    end
    return DummyM
end

function IterativeCompression1(M::MPS,D::Int;ϵ=eps(Float64),maxiteration = 100)
    Mt = LC(M,D).tensors
    l = M.length
    counter = 0
    Λ = inner(Mt,Mt)
    Bs = Dict{Int,AbstractArray}()
    while abs(1-Λ)>ϵ && counter < maxiteration
        for (k,v) in pairs(Mt)
            Mt[k] = v/sqrt(Λ)
        end
        @tensor LRM[σ,a,b] := L(Mt,M,l-1)[a,ap]*M[l][σ,ap,b]
        DummyM = reshape(LRM,(:,M.σ))
        B = qrfact(DummyM')[:Q]'
        Bs[l] = reshape(B,size(LRM))
        for i in l-1:-1:2
            @tensor RM[σ,ai,aimp] := R(Bs,M,i+1)[ai,aip]*M[i][σ,aimp,aip]
            @tensor LRM[σ,a,b] := L(Mt,M,i-1)[a,ap]*RM[σ,ap,b]
            DummyM = reshape(LRM,(:,size(LRM)[3]*M.σ))
            B = qrfact(DummyM')[:Q]'
            Bs[i] = reshape(B,size(LRM))
        end
        @tensor LRM[σ,a,b]:= R(Bs,M,2)[b,ap]*M[1][σ,a,ap]
        DummyM = reshape(LRM,(M.σ,:))
        B = qrfact(DummyM)[:Q]
        Mt[1] = reshape(B,size(LRM))
        for i in 2:l-1
            @tensor RM[σ,ai,aimp] := R(Bs,M,i+1)[ai,aip]*M[i][σ,aimp,aip]
            @tensor LRM[σ,a,b] := L(Mt,M,i-1)[a,ap]*RM[σ,ap,b]
            DummyM = reshape(LRM,(size(LRM)[2]*M.σ,:))
            B = qrfact(DummyM)[:Q]
            Mt[i] = reshape(B,size(LRM))
        end
        @tensor LRM[σ,a,b] := L(Mt,M,l-1)[a,ap]*M[l][σ,ap,b]
        Mt[l] = LRM
        counter += 1
        Λ = inner(Mt,Mt)
    end
    if counter >= maxiteration
         throw("Did not converge with the given precision and bond dimension")
    end
    return MPS(Mt,l,D,l,M.σ)
end

function ⊗(W::Dict,M::MPS)
    l = M.length
    Mt = M.tensors
    WM = Dict{Int,AbstractArray}()
    D = 1
    if ndims(W[1]) < 4
        W[1] = reshape(W[1],(:,:,1,:))
    end
    if ndims(Mt[1]) < 3
        Mt[1] = reshape(Mt[1],(:,1,:))
    end
    if ndims(W[l]) < 4
        W[l] = reshape(W[l],(:,:,:,1))
    end
    if ndims(Mt[l]) < 3
        Mt[l] = reshape(Mt[l],(:,:,1))
    end
    for i in 1:l
        Wi = W[i]
        Mti = Mt[i]
        @tensor DummyM[σ1,aw,am,bw,bm] := Wi[σ1,σ,aw,bw]*Mti[σ,am,bm]
        (σ1,aw,am,bw,bm) = size(DummyM)
        # println(i," ",size(DummyM))
        D = max(aw,am,D)
        WM[i] = reshape(DummyM,(σ1,aw*am,bw*bm))
    end
    return MPS(WM,l,M.BD*D,M.CP,size(W[1])[1])
end

function MyEig(H::Array{T,2} where T<:Number,v::Array{T} where T<:Number)
    H[diagind(H)] .= real(diag(H))
    try
        (dλ,dϕ) = eigs(Hermitian(H);nev=1,which=:SR,v0=reshape(v,(:)))
        return (dλ[1],dϕ)
    catch
        println("size of H: ",size(H))
        println("using eigen instead")
        S = eigen(Hermitian(H),1:1)
        dλ = S.values[1]
        dϕ = S.vectors[:,1]
        return (dλ,dϕ)
    end
end

function SixIndexEigs(Li::AbstractArray,Hi::AbstractArray,Ri::AbstractArray,Bsi)
    @tensor LWR[σ,alm,al,σp,almp,alp] := Li[alm,blm,almp]*Hi[σ,σp,blm,bl]*Ri[al,bl,alp]
    (d1,d2,d3,d4,d5,d6) = size(LWR)
    DH = reshape(LWR,(d1*d2*d3,d4*d5*d6))
    # println(DH)
    (dλ,dϕ) = MyEig(DH,Bsi)
    println(dλ)
    return (dϕ,dλ,(d1,d2,d3))
end

function FourIndexEigs(LWR::AbstractArray,Bsi::AbstractArray)
    (d1,d2,d3,d4) = size(LWR)                                   # ↑
    DH = reshape(LWR,(d1*d2,d3*d4))                             # ↑
    # println(DH)
    DummyBs = deepcopy(reshape(Bsi,(:)))
    (dλ,dϕ) = MyEig(DH,Bsi)
    println(dλ)
    return (dϕ,dλ,(d1,d2))
end

function DMRG(H::Dict,l;D=2,σ=max(first.(size.(values(H)))...),ϵ=10000*eps(Float64),maxiteration = 30,DMax=50)#ψ0 = Dict{Int,Array{ComplexF64,3}}())
    #initial random state
    local eps0 = ϵ+1.0
    local MBs
    while eps0 > ϵ && D<DMax
        println("##############################################################")
        println("Now using bond dimension D=$(D)")
        println("##############################################################")
        ψ0 = Dict{Int,Array{ComplexF64,3}}()
        ψ0[1] = rand(ComplexF64,(σ,1,D))
        for i in 2:l-1
            ψ0[i] = rand(ComplexF64,(σ,D,D))
        end
        ψ0[l] = rand(ComplexF64,(σ,D,1))
        Ms = RC(MPS(ψ0,l,D,l,σ))#,D)       #Turn into a Right Canonical MPS
        norm_of_MPS = sqrt(abs(inner(Ms,Ms)))
        println(norm_of_MPS)
        Ms.tensors[l] = Ms.tensors[l]/norm_of_MPS
        norm_of_MPS = sqrt(abs(inner(Ms,Ms)))
        println(norm_of_MPS)
        Bs = Ms.tensors                 #We save the tensors in Bs without the details we put into MPS
        Rs = Dict{Int,AbstractArray}()  # A list of saved contractions of H between two states from the right hand side.
        Ls = Dict{Int,AbstractArray}()  # A list of saved contractions of H between two states from the left hand side.
        #initializing Rs
        Bsi = deepcopy(Bs[l])
        Hi = deepcopy(H[l])
        DummyBsi = Bsi[:,:,1]
        DummyHi = Hi[:,:,:,1]
        @tensor DummyRs[a,b,ap] := conj(DummyBsi[σ1,a])*DummyHi[σ1,σ2,b]*DummyBsi[σ2,ap]
        @tensor H22[a,b,bp,ap] := DummyBsi[σ,a]*DummyHi[σ,σp,b]*DummyHi[σp,σpp,bp]*conj(DummyBsi[σpp,ap])
        Rs[l] = deepcopy(DummyRs)
        for i in l-1:-1:1
            Bsi = deepcopy(Bs[i])
            Hi = deepcopy(H[i])
            Rsi = deepcopy(Rs[i+1])
            @tensor DummyRs[a,b,ap] := conj(Bsi[σ1,a,a1])*Hi[σ1,σ2,b,b1]*Bsi[σ2,ap,c1]*Rsi[a1,b1,c1]
            @tensor H22[a,b,bp,ap] := H22[a1,b1,bp1,ap1]*Bsi[σ,a,a1]*Hi[σ,σp,b,b1]*Hi[σp,σpp,bp,bp1]*conj(Bsi[σpp,ap,ap1])
            Rs[i] = deepcopy(DummyRs)
        end
        H2 = deepcopy(Rs[1])[1]
        H22 = deepcopy(H22[1])
        H2 = typeof(H2) <: AbstractArray{} ? H2[1] : H2
        eps0 = abs((abs(H22) - (H2*conj(H2)))/abs(H22))
        println("initial <H^2>-<H>^2=$(eps0)")
        for iterations in 1:maxiteration
            #Initializing Right Sweep
            t0 = time()
            Hi = deepcopy(H[1])
            Ri = deepcopy(Rs[2])
            DummyHi = deepcopy(Hi[:,:,1,:])
            @tensor LWR[σ,a,σp,ap] := DummyHi[σ,σp,b]*Ri[a,b,ap]           # Preparing the matirx which we want to find its groundstate.
            (dϕ,H2,(d1,d2)) = FourIndexEigs(LWR,Bs[1])
            F = qr(reshape(dϕ,(d1,:)))
            dQ,dR = (Array(F.Q),Array(F.R))
            Bs[1] = reshape(dQ,(d1,1,:))
            Bsi = deepcopy(Bs[2])
            @tensor DBsi[σ,a,b] := dR[a,di]*Bsi[σ,di,b]
            Bs[2] = deepcopy(DBsi)
            Ri = deepcopy(Rs[3])
            Hi = deepcopy(H[2])
            @tensor DummyRs[a,b,apr] := Ri[apl,bpl,aplpr]*conj(DBsi[σ,a,apl])*Hi[σ,σpr,b,bpl]*Bsi[σpr,apr,aplpr]
            Rs[2] = deepcopy(DummyRs)
            Bsi = deepcopy(Bs[1])
            @tensor DummyLs[a,b,ap] := conj(dQ[σ,a])*DummyHi[σ,σp,b]*dQ[σp,ap]
            Ls[1] = deepcopy(DummyLs)
            #Right Sweep loop
            for i in 2:l-1
                (dϕ,H2,(d1,d2,d3))=SixIndexEigs(Ls[i-1],H[i],Rs[i+1],Bs[i])
                F = qr(reshape(dϕ,(d1*d2,:)))
                dQ,dR = (Array(F.Q),Array(F.R))
                Bs[i] = reshape(dQ,(d1,d2,:))
                Bsi = deepcopy(Bs[i+1])
                @tensor Bsi[σ,a,b] := dR[a,c]*Bsi[σ,c,b]
                Bs[i+1] = deepcopy(Bsi)
                if i<l-1
                    Ri = deepcopy(Rs[i+2])
                    Hi = deepcopy(H[i+1])
                    @tensor DummyRs[a,b,apr] := Ri[apl,bpl,aplpr]*conj(Bsi[σ,a,apl])*Hi[σ,σpr,b,bpl]*Bsi[σpr,apr,aplpr]
                    Rs[i+1] = deepcopy(DummyRs)
                elseif i == l-1
                    Hi = deepcopy(H[l])
                    Bsi = deepcopy(Bs[l])
                    DummyBsi = Bsi[:,:,1]
                    DummyHi = Hi[:,:,:,1]
                    @tensor DummyRs[a,b,ap] := conj(DummyBsi[σ1,a])*DummyHi[σ1,σ2,b]*DummyBsi[σ2,ap]
                    Rs[l] = deepcopy(DummyRs)
                end
                Bsi = deepcopy(Bs[i])
                Hi = deepcopy(H[i])
                Li = deepcopy(Ls[i-1])
                @tensor DummyLs[a,b,ap] := Li[am,bm,amp]*conj(Bsi[σ,am,a])*Hi[σ,σp,bm,b]*Bsi[σp,amp,ap]
                Ls[i] = deepcopy(DummyLs)
            end
            println("finished iteration = $(iterations) right sweep in $(time() - t0).")
            #Potential break
            HBs = H⊗MPS(Bs,l,D,1,σ)
            H22 = inner(HBs,HBs)
            H22 = H22[1]
            println("H2 :",H2)
            println("H22 :" ,H22)
            eps0 = abs((abs(H22) - (H2*conj(H2)))/abs(H22))
            println("rel_tol = $(eps0)")
            # if eps0 <= ϵ
            #     break
            # end
            #Initializing Left Sweep
            t0 = time()
            Hi = H[l]
            Li = Ls[l-1]
            DummyHi = deepcopy(Hi[:,:,:,1])
            @tensor LWR[σ,a,σp,ap] := Li[a,b,ap] * DummyHi[σ,σp,b]          # Preparing the matirx which we want to find its groundstate.
            (dϕ,H2,(d1,d2)) = FourIndexEigs(LWR,Bs[l])
            dϕ = permutedims(reshape(dϕ,(d1,d2)),(2,1))
            F = lq(dϕ)
            dL,dQ =(Array(F.L),Array(F.Q))
            dQ = permutedims(reshape(dQ,(:,d1)),(2,1))
            Bs[l] = reshape(dQ,(d1,:,1))
            Bsi = deepcopy(Bs[l-1])
            @tensor dBsi[σ,a,b] := Bsi[σ,a,k]*dL[k,b]
            Bs[l-1] = deepcopy(dBsi)
            Bsi = deepcopy(Bs[l])
            @tensor DummyRs[a,b,ap] := conj(dQ[σ1,a])*DummyHi[σ1,σ2,b]*dQ[σ2,ap]
            Rs[l] = deepcopy(DummyRs)
            #Left Sweep loop
            for i in l-1:-1:2
                (dϕ,H2,(d1,d2,d3))=SixIndexEigs(Ls[i-1],H[i],Rs[i+1],Bs[i])
                dϕ = reshape(dϕ,(d1,d2,d3))
                dϕ = permutedims(dϕ,(2,1,3))
                F = lq(reshape(dϕ,(d2,:)))
                (dL,dQ) =(Array(F.L),Array(F.Q))
                dQ = permutedims(reshape(dQ,(:,d1,d3)),(2,1,3))
                Bs[i] = dQ
                Bsi = deepcopy(Bs[i-1])
                @tensor Bsi[σ,a,b] := Bsi[σ,a,di]*dL[di,b]
                Bs[i-1] = deepcopy(Bsi)
                if i>2
                    Li = deepcopy(Ls[i-2])
                    Hi = deepcopy(H[i-1])
                    @tensor DummyLs[a,b,apr] := Li[ami,bmi,amipr]*conj(Bsi[σpr,ami,a])*Hi[σ,σpr,bmi,b]*Bsi[σ,amipr,apr]
                    Ls[i-1] = deepcopy(DummyLs)
                elseif i == 2
                    Hi = deepcopy(H[1])
                    Bsi = deepcopy(Bs[1])
                    DummyHi = Hi[:,:,1,:]
                    DummyBsi = deepcopy(Bsi[:,1,:])
                    @tensor DummyLs[a,b,ap] := conj(DummyBsi[σ,a])*DummyHi[σ,σp,b]*DummyBsi[σp,ap]
                    Ls[1] = deepcopy(DummyLs)
                end
                Hi = deepcopy(H[i])
                Ri = deepcopy(Rs[i+1])
                @tensor DummyRs[a,b,ap] := conj(dQ[σ1,a,am])*Hi[σ1,σ2,b,bm]*dQ[σ2,ap,amp]*Ri[am,bm,amp]
                Rs[i] = deepcopy(DummyRs)
            end
            println("finished iteration = $(iterations) left sweep in $(time() - t0).")
            norm_of_MPS = sqrt(abs(inner(MPS(Bs,l,D,1,σ),MPS(Bs,l,D,1,σ))))
            Bs[l] = Bs[l]/norm_of_MPS
            MBs = MPS(Bs,l,D,1,σ)
            HBs = H⊗MBs
            H22 = inner(HBs,HBs)
            H22 = H22[1]
            println("H2 :",H2)
            println("H22 :" ,H22)
            eps0 = abs((abs(H22) - (H2*conj(H2)))/abs(H22))
            println("rel_tol = $(eps0)")
            if eps0 <= ϵ
                break
            end
        end
        if eps0 != 0.0
            D += round(Int,log(eps0/ϵ))+1
        end
    end
    return MBs,eps0
end

function DMRG(H::MPO;D=2,ϵ=10000*eps(Float64),maxiteration = 30,DMax=50)
    H.σ1 != H.σ2 && throw("Bad Hamiltonian size, the Hamiltonian must be Hermitian")
    DMRG(H.tensors,H.length,D=D,σ=H.σ1,ϵ=ϵ,maxiteration=maxiteration,DMax=DMax)
end

function DMRG(H::MPO,ψ0::MPS;ϵ=10000*eps(Float64),maxiteration=30,DMax=50)
    #initial random state
    local eps0 = ϵ+1.0
    D = ψ0.BD
    local MBs
    l = H.length
    σ = ψ0.σ
    Ht = H.tensors
    ψ1 = ψ0
    while eps0 > ϵ && D<DMax
        println("##############################################################")
        println("Now using bond dimension D=$(D)")
        println("##############################################################")
        if ψ1.BD < D
            ψ1 = IncreaseBondDimension(ψ1,D)
        end
        Ms = RC(ψ1)#,D)       #Turn into a Right Canonical MPS
        norm_of_MPS = sqrt(abs(inner(Ms,Ms)))
        println(norm_of_MPS)
        Ms.tensors[l] = Ms.tensors[l]/norm_of_MPS
        norm_of_MPS = sqrt(abs(inner(Ms,Ms)))
        println(norm_of_MPS)
        Bs = Ms.tensors                 #We save the tensors in Bs without the details we put into MPS
        Rs = Dict{Int,AbstractArray}()  # A list of saved contractions of H between two states from the right hand side.
        Ls = Dict{Int,AbstractArray}()  # A list of saved contractions of H between two states from the left hand side.
        #initializing Rs
        Bsi = deepcopy(Bs[l])
        Hi = deepcopy(Ht[l])
        DummyBsi = Bsi[:,:,1]
        DummyHi = Hi[:,:,:,1]
        @tensor DummyRs[a,b,ap] := conj(DummyBsi[σ1,a])*DummyHi[σ1,σ2,b]*DummyBsi[σ2,ap]
        @tensor H22[a,b,bp,ap] := DummyBsi[σ,a]*DummyHi[σ,σp,b]*DummyHi[σp,σpp,bp]*conj(DummyBsi[σpp,ap])
        Rs[l] = deepcopy(DummyRs)
        for i in l-1:-1:1
            Bsi = deepcopy(Bs[i])
            Hi = deepcopy(Ht[i])
            Rsi = deepcopy(Rs[i+1])
            @tensor DummyRs[a,b,ap] := conj(Bsi[σ1,a,a1])*Hi[σ1,σ2,b,b1]*Bsi[σ2,ap,c1]*Rsi[a1,b1,c1]
            @tensor H22[a,b,bp,ap] := H22[a1,b1,bp1,ap1]*Bsi[σ,a,a1]*Hi[σ,σp,b,b1]*Hi[σp,σpp,bp,bp1]*conj(Bsi[σpp,ap,ap1])
            Rs[i] = deepcopy(DummyRs)
        end
        H2 = deepcopy(Rs[1])[1]
        H22 = deepcopy(H22[1])
        H2 = typeof(H2) <: AbstractArray{} ? H2[1] : H2
        eps0 = abs((abs(H22) - (H2*conj(H2)))/abs(H22))
        println("initial <H^2>-<H>^2=$(eps0)")
        for iterations in 1:maxiteration
            #Initializing Right Sweep
            t0 = time()
            Hi = deepcopy(Ht[1])
            Ri = deepcopy(Rs[2])
            DummyHi = deepcopy(Hi[:,:,1,:])
            @tensor LWR[σ,a,σp,ap] := DummyHi[σ,σp,b]*Ri[a,b,ap]           # Preparing the matirx which we want to find its groundstate.
            (dϕ,H2,(d1,d2)) = FourIndexEigs(LWR,Bs[1])
            F = qr(reshape(dϕ,(d1,:)))
            dQ,dR = (Array(F.Q),Array(F.R))
            Bs[1] = reshape(dQ,(d1,1,:))
            Bsi = deepcopy(Bs[2])
            @tensor DBsi[σ,a,b] := dR[a,di]*Bsi[σ,di,b]
            Bs[2] = deepcopy(DBsi)
            Ri = deepcopy(Rs[3])
            Hi = deepcopy(Ht[2])
            @tensor DummyRs[a,b,apr] := Ri[apl,bpl,aplpr]*conj(DBsi[σ,a,apl])*Hi[σ,σpr,b,bpl]*Bsi[σpr,apr,aplpr]
            Rs[2] = deepcopy(DummyRs)
            Bsi = deepcopy(Bs[1])
            @tensor DummyLs[a,b,ap] := conj(dQ[σ,a])*DummyHi[σ,σp,b]*dQ[σp,ap]
            Ls[1] = deepcopy(DummyLs)
            #Right Sweep loop
            for i in 2:l-1
                (dϕ,H2,(d1,d2,d3))=SixIndexEigs(Ls[i-1],Ht[i],Rs[i+1],Bs[i])
                F = qr(reshape(dϕ,(d1*d2,:)))
                dQ,dR = (Array(F.Q),Array(F.R))
                Bs[i] = reshape(dQ,(d1,d2,:))
                Bsi = deepcopy(Bs[i+1])
                @tensor Bsi[σ,a,b] := dR[a,c]*Bsi[σ,c,b]
                Bs[i+1] = deepcopy(Bsi)
                if i<l-1
                    Ri = deepcopy(Rs[i+2])
                    Hi = deepcopy(Ht[i+1])
                    @tensor DummyRs[a,b,apr] := Ri[apl,bpl,aplpr]*conj(Bsi[σ,a,apl])*Hi[σ,σpr,b,bpl]*Bsi[σpr,apr,aplpr]
                    Rs[i+1] = deepcopy(DummyRs)
                elseif i == l-1
                    Hi = deepcopy(Ht[l])
                    Bsi = deepcopy(Bs[l])
                    DummyBsi = Bsi[:,:,1]
                    DummyHi = Hi[:,:,:,1]
                    @tensor DummyRs[a,b,ap] := conj(DummyBsi[σ1,a])*DummyHi[σ1,σ2,b]*DummyBsi[σ2,ap]
                    Rs[l] = deepcopy(DummyRs)
                end
                Bsi = deepcopy(Bs[i])
                Hi = deepcopy(Ht[i])
                Li = deepcopy(Ls[i-1])
                @tensor DummyLs[a,b,ap] := Li[am,bm,amp]*conj(Bsi[σ,am,a])*Hi[σ,σp,bm,b]*Bsi[σp,amp,ap]
                Ls[i] = deepcopy(DummyLs)
            end
            println("finished iteration = $(iterations) right sweep in $(time() - t0).")
            #Potential break
            HBs = Ht⊗MPS(Bs,l,D,1,σ)
            H22 = inner(HBs,HBs)
            H22 = H22[1]
            println("H2 :",H2)
            println("H22 :" ,H22)
            eps0 = abs((abs(H22) - (H2*conj(H2)))/abs(H22))
            println("rel_tol = $(eps0)")
            # if eps0 <= ϵ
            #     break
            # end
            #Initializing Left Sweep
            t0 = time()
            Hi = Ht[l]
            Li = Ls[l-1]
            DummyHi = deepcopy(Hi[:,:,:,1])
            @tensor LWR[σ,a,σp,ap] := Li[a,b,ap] * DummyHi[σ,σp,b]          # Preparing the matirx which we want to find its groundstate.
            (dϕ,H2,(d1,d2)) = FourIndexEigs(LWR,Bs[l])
            dϕ = permutedims(reshape(dϕ,(d1,d2)),(2,1))
            F = lq(dϕ)
            dL,dQ =(Array(F.L),Array(F.Q))
            dQ = permutedims(reshape(dQ,(:,d1)),(2,1))
            Bs[l] = reshape(dQ,(d1,:,1))
            Bsi = deepcopy(Bs[l-1])
            @tensor dBsi[σ,a,b] := Bsi[σ,a,k]*dL[k,b]
            Bs[l-1] = deepcopy(dBsi)
            Bsi = deepcopy(Bs[l])
            @tensor DummyRs[a,b,ap] := conj(dQ[σ1,a])*DummyHi[σ1,σ2,b]*dQ[σ2,ap]
            Rs[l] = deepcopy(DummyRs)
            #Left Sweep loop
            for i in l-1:-1:2
                (dϕ,H2,(d1,d2,d3))=SixIndexEigs(Ls[i-1],Ht[i],Rs[i+1],Bs[i])
                dϕ = reshape(dϕ,(d1,d2,d3))
                dϕ = permutedims(dϕ,(2,1,3))
                F = lq(reshape(dϕ,(d2,:)))
                (dL,dQ) =(Array(F.L),Array(F.Q))
                dQ = permutedims(reshape(dQ,(:,d1,d3)),(2,1,3))
                Bs[i] = dQ
                Bsi = deepcopy(Bs[i-1])
                @tensor Bsi[σ,a,b] := Bsi[σ,a,di]*dL[di,b]
                Bs[i-1] = deepcopy(Bsi)
                if i>2
                    Li = deepcopy(Ls[i-2])
                    Hi = deepcopy(Ht[i-1])
                    @tensor DummyLs[a,b,apr] := Li[ami,bmi,amipr]*conj(Bsi[σpr,ami,a])*Hi[σ,σpr,bmi,b]*Bsi[σ,amipr,apr]
                    Ls[i-1] = deepcopy(DummyLs)
                elseif i == 2
                    Hi = deepcopy(Ht[1])
                    Bsi = deepcopy(Bs[1])
                    DummyHi = Hi[:,:,1,:]
                    DummyBsi = deepcopy(Bsi[:,1,:])
                    @tensor DummyLs[a,b,ap] := conj(DummyBsi[σ,a])*DummyHi[σ,σp,b]*DummyBsi[σp,ap]
                    Ls[1] = deepcopy(DummyLs)
                end
                Hi = deepcopy(Ht[i])
                Ri = deepcopy(Rs[i+1])
                @tensor DummyRs[a,b,ap] := conj(dQ[σ1,a,am])*Hi[σ1,σ2,b,bm]*dQ[σ2,ap,amp]*Ri[am,bm,amp]
                Rs[i] = deepcopy(DummyRs)
            end
            println("finished iteration = $(iterations) left sweep in $(time() - t0).")
            norm_of_MPS = sqrt(abs(inner(MPS(Bs,l,D,1,σ),MPS(Bs,l,D,1,σ))))
            Bs[l] = Bs[l]/norm_of_MPS
            MBs = MPS(Bs,l,D,1,σ)
            HBs = Ht⊗MBs
            H22 = inner(HBs,HBs)
            H22 = H22[1]
            println("H2 :",H2)
            println("H22 :" ,H22)
            eps0 = abs((abs(H22) - (H2*conj(H2)))/abs(H22))
            println("rel_tol = $(eps0)")
            if eps0 <= ϵ
                break
            end
        end
        D += round(Int,log(eps0/ϵ))+1
    end
    return MBs,eps0
end

function kron_factor(H::AbstractArray{T,2},a1::Int=0,b1::Int=0,a2::Int=0,b2::Int=0;rel_tol=1.0e6*eps(Float64)) where T<:Number
    if a1 ==0 || b1 == 0 || a2 == 0 || b2 == 0
        l = LinearAlgebra.checksquare(H)
        sqrt(l) == isqrt(l) ? a1 = a2 = b1 = b2 = isqrt(l) : throw("The size of the Hamiltonian term should be a perfect square times a perfect square, like 4x4 or 9x9")
    else
        size(H) != (a1*b1,a2*b2) && throw("the sizes given in the input don't match the matrix")
    end
    H1 = reshape(H,(b1,a1,b2,a2))
    H2 = permutedims(H1,(1,3,2,4))
    H3 = permutedims(reshape(H2,(b1*b2,a1*a2)),(2,1))
    F = svd(H3)
    normH = norm(H)
    out = Dict{Int,Tuple{Array{T,2},Array{T,2}}}()
    counter = 1
    for s in 1:length(F.S)
        if F.S[s] > rel_tol
            out[counter] = (sqrt(F.S[s])*zchop(reshape(F.U[:,s],(a1,a2)),normH*rel_tol),sqrt(F.S[s])*zchop(reshape(F.Vt[s,:],(b1,b2)),normH*rel_tol))
            counter += 1
        end
    end
    total = zeros(Float64,(a1*b1,a2*b2))
    for i in keys(out)
        b = out[i]
        total += kron(b[1],b[2])
    end
    if norm(total-H)/normH>rel_tol
        println("H:     ")
        display(H)
        println("total: ")
        display(total)
        println(norm(total-H)/normH)
        throw("The factorization result is not equivalent to the input matrix")
    end
    return out
end


function NNHtoMPO(H::Array{T,2} where T<:Number,n::Int)
    H = Hermitian(H)
    l = LinearAlgebra.checksquare(H)
    √l == isqrt(l) ? m = isqrt(l) : throw("The size of the generic Nearest Neighbor Hamiltonian term should be a perfect square times a perfect square, like 4x4 or 9x9")
    s = kron_factor(H, m, m, m, m)
    len = length(s)
    W = Dict{Int,Array{ComplexF64,4}}()
    for k in 2:n-1
        W[k] = zeros(ComplexF16,(m,m,len+2,len+2))
        W[k][:,:,1,1] = Matrix{ComplexF64}(I,m,m)
        W[k][:,:,len+2,len+2] = Matrix{ComplexF64}(I,m,m)
        row_number = 2
        for el in keys(s)
            W[k][:,:,row_number,1] = s[el][2]
            W[k][:,:,len+2,row_number] = s[el][1]
            row_number += 1
        end
    end
    row_number = 2
    W[n] = zeros(ComplexF64,(m,m,len+2,1))
    W[1] = zeros(ComplexF64,(m,m,1,len+2))
    for el in keys(s)
        W[n][:,:,row_number,1] = s[el][2]
        W[1][:,:,1,row_number] = s[el][1]
        row_number += 1
    end
    W[n][:,:,1,1] = Matrix{ComplexF64}(I,m,m)
    W[1][:,:,1,len+2] = Matrix{ComplexF64}(I,m,m)
    return MPO(W)
end

function NNHandSSHtoMPO(H1::Array{T,2},H2::Array{T,2},n::Int) where T<:Number
    # H1 = (H1+H1')/2.
    H1 = Hermitian(H1)
    l = LinearAlgebra.checksquare(H1)
    √l == isqrt(l) ? m = isqrt(l) : throw("The size of the generic Nearest Neighbor Hamiltonian term should be a perfect square times a perfect square, like 4x4 or 9x9")
    s = kron_factor(H1, m, m, m, m)
    len = length(s)
    W = Dict{Int,Array{ComplexF64,4}}()
    for k in 2:n-1
        W[k] = zeros(ComplexF16,(m,m,len+2,len+2))
        W[k][:,:,1,1] = Matrix{ComplexF64}(I,m,m)
        W[k][:,:,len+2,len+2] = Matrix{ComplexF64}(I,m,m)
        row_number = 2
        for el in keys(s)
            W[k][:,:,row_number,1] = s[el][2]
            W[k][:,:,len+2,row_number] = s[el][1]
            row_number += 1
        end
        W[k][:,:,len+2,1] = H2
    end
    row_number = 2
    W[n] = zeros(ComplexF64,(m,m,len+2,1))
    W[1] = zeros(ComplexF64,(m,m,1,len+2))
    for el in keys(s)
        W[n][:,:,row_number,1] = s[el][2]
        W[1][:,:,1,row_number] = s[el][1]
        row_number += 1
    end
    W[n][:,:,1,1] = Matrix{ComplexF64}(I,m,m)
    W[n][:,:,len+2,1] = H2
    W[1][:,:,1,1] = H2
    W[1][:,:,1,len+2] = Matrix{ComplexF64}(I,m,m)
    return MPO(W)
end


function CutEntanglement(M::MPS,n::Int;after::Bool=true)
    if after
        n ≤ 0 || n ≥ M.length && throw("The bond you are trying to cut must be inside the MPS.")
        Ms = M.tensors
        l = M.length
        D = M.BD
        (σ,j,k) = size(Ms[n])
        F = svd(reshape(Ms[n],(σ*j,k)))
        S = F.S[1]
        Ms[n] = reshape(F.U[:,1],(σ,j,1))
        Vd = reshape(F.Vt[1,:],(1,k))
        DummyM = Ms[n+1]
        @tensor newM[s,a,b] := S*Vd[a,m]*DummyM[s,m,b]
        Ms[n+1] = newM
    else
        n ≤ 1 || n > M.length && throw("The bond you are trying to cut must be inside the MPS.")
        Ms = M.tensors
        l = M.length
        D = M.BD
        (σ,j,k) = size(Ms[n])
        F = svd(reshape(permutedims(Ms[n],(2,1,3)),(j,σ*k)))
        S = F.S[1]
        U = reshape(F.U[:,1],(j,1))
        Ms[n] = permutedims(reshape(F.Vt[1,:],(1,σ,k)),(2,1,3))
        DummyM = Ms[n-1]
        @tensor newM[s,a,b] := DummyM[s,a,m]*S*U[m,b]
        Ms[n-1] = newM
    end
    return MPS(Ms,l,D,M.CP,M.σ)
end
