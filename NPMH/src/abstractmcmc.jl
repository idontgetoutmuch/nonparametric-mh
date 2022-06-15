# NonparametricMH
struct NPMHSampler <: AbstractMCMC.AbstractSampler end

# state of NonparametricMH
struct NPMHState{S,L}
    "Sample of the NonparametricMH."
    sample::S # S := Vector{Float64}
    "Log-likelihood of the sample."
    loglikelihood::L

    NPMHState(sample, loglikelihood) = new{typeof(sample), typeof(loglikelihood)}(sample, loglikelihood)
end

# first step of the NonparametricMH
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    ::NPMHSampler;
    kwargs...
)
    xsample, logp = initial_sample(rng, model)
    return xsample, NPMHState(xsample, logp)
end

# subsequent steps of the NonparametricMH
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    ::NPMHSampler,
    state::NPMHState;
    kwargs...,
)
    # get the current state (xsample, vsample)

    xsample = state.sample
    # sample from the auxiliary kernel
    kernel = getkernel(model)
    vsample = Random.randn(rng, xsample, kernel)

    # compute the proposed state by applying involution
    xsample, vsample, newxsample, newvsample = proposal(rng, model, xsample, vsample)

    # accept or reject proposed state
    entdist = getentdist(model)
    auxdist = getauxdist(model)

    xk, _, xlogp = logpdf(model, xsample)
    xlogp += sum(t->Distributions.loglikelihood(entdist, t), xsample[xk+1:end], init=0.0)
    vlogp = Distributions.loglikelihood(kernel, xsample[1:xk], vsample[1:xk])
    vlogp += sum(t->Distributions.loglikelihood(auxdist, t), vsample[xk+1:end], init=0.0)

    newxk, _, newxlogp = logpdf(model, newxsample)
    newxlogp += sum(t->Distributions.loglikelihood(entdist, t), newxsample[newxk+1:end], init=0.0)
    newvlogp = Distributions.loglikelihood(kernel, newxsample[1:newxk], newvsample[1:newxk])
    newvlogp += sum(t->Distributions.loglikelihood(auxdist, t), newvsample[newxk+1:end], init=0.0)

    # log absolute value of the jacobian determinant is 0

    logα = newxlogp + newvlogp - xlogp - vlogp

    if -Random.randexp(rng) < logα
        # accept
        nextsample, nextstate =
        newxsample[1:newxk],
        NPMHState(newxsample[1:newxk], newxlogp)
    else
        # reject
        nextsample, nextstate =
        xsample[1:xk],
        NPMHState(xsample[1:xk], xlogp)
    end

    return nextsample, nextstate
end
