"""
    `NPMHModel` is the internal model structure.
"""
struct NPMHModel{K,E,A} <: AbstractMCMC.AbstractModel
    "Auxiliary kernel."
    auxiliary_kernel::K
    "Log likelihood function."
    loglikelihood
    "Stock measure for entropy."
    entdist::E
    "Stock measure for auxiliary space."
    auxdist::A

    NPMHModel(
        auxiliary_kernel,
        turing_model;
        sampler=DynamicPPL.SampleFromPrior(),
        entdist=Normal(0,1),
        auxdist=Normal(0,1),
    ) =
    new{
        typeof(auxiliary_kernel),
        typeof(entdist),
        typeof(auxdist)
    }(
        auxiliary_kernel,
        np_gen_logπ(sampler,turing_model),
        entdist,
        auxdist
    )
end

# evaluate the loglikelihood of a sample
logpdf(model::NPMHModel, x) = model.loglikelihood(x)

# get functions
getkernel(model::NPMHModel) = model.auxiliary_kernel
getloglikelihood(model::NPMHModel) = model.loglikelihood
getentdist(model::NPMHModel) = model.entdist
getauxdist(model::NPMHModel) = model.auxdist

"""
    `AbstractAuxKernel` is the abstract type of all auxiliary kernels.

- `Random.rand` returns the `n`-th element of the random sample from the kernel conditioned on the vector `x::Vector{Float64}`.
- `Random.randn` returns a random sample from the kernel conditioned on the vector `x::Vector{Float64}`.
- `Distributions.loglikelihood` returns the log likelihood of the kernel conditioned on `x` and `v`.
"""
abstract type AbstractAuxKernel end

"""
    `AuxKernel` constructs an auxiliary kernel of the type Vector{Float64} -> Vector{Distributions.UnivariateDistribution}
"""
struct AuxKernel{T} <: AbstractAuxKernel
    "Auxiliary Kernel." # auxiliary kernel function
    auxkernel::T

    AuxKernel(auxkernel) = new{typeof(auxkernel)}(auxkernel)
end

Random.rand(rng, x, k::AuxKernel, n::Int) = Random.randn(rng, x, k)[n]

function Random.randn(rng::Random.AbstractRNG, x, k::AuxKernel)
    dists = k.auxkernel(x)
    v = []
    for dist in dists
        if typeof(dist) <: Distributions.UnivariateDistribution
            append!(v, Random.rand(rng, dist))
        else
            error("Auxiliary kernel conditioned on x is not univariate.")
        end
    end
    return v
end

function Distributions.loglikelihood(k::AuxKernel, x, v)
    dists = k.auxkernel(x)
    n = length(dists)
    logp = 0
    for i in 1:n
        if typeof(dists[i]) <: Distributions.UnivariateDistribution
            logp += Distributions.loglikelihood(dists[i], v[i])
        else
            error("Auxiliary kernel conditioned on x is not univariate.")
        end
    end
    return logp
end

"""
    `ModelAuxKernel` constructs an auxiliary kernel via the `@model` macro.
"""
struct ModelAuxKernel{T,S} <: AbstractAuxKernel
    "Kernel Model."
    kmodel::T
    "Kernel Sampler"
    ksampler::S

    ModelAuxKernel(kmodel; ksampler = DynamicPPL.SampleFromPrior()) = new{typeof(kmodel), typeof(ksampler)}(kmodel, ksampler)
end

Random.rand(rng::Random.AbstractRNG, x, k::ModelAuxKernel, n::Int) = Random.randn(rng, x, k)[n]

function Random.randn(rng::Random.AbstractRNG, x, k::ModelAuxKernel)
    model = k.kmodel(x)
    vi = VarInfo(model, k.ksampler)
    v = vi[k.ksampler]
    return v
end

function Distributions.loglikelihood(k::ModelAuxKernel, x, v)
    logp = gen_logπ(k.ksampler, k.kmodel(x))(v)
    return logp
end

"""
    `PointwiseAuxKernel` constructs an auxiliary kernel for each dimension

- `kernel` should be of type (Int, Vector{Float64}) -> Distributions.Sampleable
"""
struct PointwiseAuxKernel{T} <: AbstractAuxKernel
    "Kernel."
    kernel::T

    PointwiseAuxKernel(kernel) = new{typeof(kernel)}(kernel)
end

Random.rand(rng::Random.AbstractRNG, x, k::PointwiseAuxKernel, n::Int) = Random.rand(rng, k.kernel(n, x))

Random.randn(rng::Random.AbstractRNG, x, k::PointwiseAuxKernel) = map(i->Random.rand(rng, x, k, i), 1:length(x))

Distributions.loglikelihood(k::PointwiseAuxKernel, x, v) =
sum(i->Distributions.loglikelihood(k.kernel(i, x), v[i]), 1:length(x), init=0.0)
