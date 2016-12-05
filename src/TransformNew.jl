# Convert between different parameterizations.

module Transform

# TODO: don't import Model; transformations should operate on
# generic ParamSets
using ..Model
using ..SensitiveFloats
import ..Log

export DataTransform, ParamBounds, ParamBox, SimplexBox,
       get_mp_transform, enforce_bounds!,
       VariationalParams, FreeVariationalParams

# A vector of variational parameters.  The outer index is
# of celestial objects, and the inner index is over individual
# parameters for that object (referenced using ParamIndex).
typealias VariationalParams{NumType <: Number} Vector{Vector{NumType}}
typealias FreeVariationalParams{NumType <: Number} Vector{Vector{NumType}}

########################
# Elementary Functions #
########################

"""
Unconstrain x in the unit interval to lie in R.
"""
inv_logit(x) = -log(1.0 / x - 1)

"""
Convert x in R to lie in the unit interval.
"""
logit(x) = 1.0 / (1.0 + exp(-x))

"""
Convert an (n - 1)-vector of real numbers to an n-vector on the simplex, where
the last entry implicitly has the untransformed value 1.
"""
function simplex_constrain!(z, x)
    T = eltype(z)
    z[end] = zero(T)
    if any(n -> n == Inf, x)
        # If more than 1 entry in x is Inf, it may be because the
        # the last entry in z is 0. Here we set all those entries to the
        # same value, though that may not be strictly correct.
        for i in eachindex(x)
            z[i] = ifelse(x[i] == Inf, one(T), zero(T))
        end
        z_sum = sum(z)
        for i in eachindex(x)
            z[i] /= z_sum
        end
    else
        for i in eachindex(x)
            z[i] = exp(x[i])
        end
        z_sum = sum(z) + 1
        for i in eachindex(x)
            z[i] /= z_sum
        end
        z[end] = inv(z_sum)
    end
    return z
end

"""
Convert an n-vector on the simplex to an (n - 1)-vector in R^{n -1}.  Entries
are expressed relative to the last element.
"""
function simplex_unconstrain!(x, z)
    log_last = log(last(z))
    for i in eachindex(x)
        x[i] = log(z[i]) - log_last
    end
    return x
end

#######################
# Transform Box Types #
#######################

immutable ParamBox
    lb::Float64  # lower bound
    ub::Float64  # upper bound
    scale::Float64
    function ParamBox(lb, ub, scale)
        @assert lb > -Inf # Not supported
        @assert scale > 0.0
        @assert lb < ub
        return new(lb, ub, scale)
    end
end

immutable SimplexBox
    lb::Float64  # lower bound
    scale::Float64
    n::Int
    function SimplexBox(lb, scale, n)
        @assert n >= 2
        @assert 0.0 <= lb < 1 / n
        return new(lb, scale, n)
    end
end

# The vector of transform parameters for a Symbol.
typealias ParamBounds Dict{Symbol, Union{Vector{ParamBox}, Vector{SimplexBox}}}

##############################
# "Free Transform" Functions #
##############################

function unbox_parameter(param, pb::ParamBox)
    @assert (pb.lb <= param <= pb.ub) "param outside bounds: $param ($(pb.lb), $(pb.ub))"
    shifted_param = param - pb.lb
    # exp and the logit functions handle infinities, so parameters can equal the bounds.
    k = (pb.ub == Inf) ? log(shifted_param) : inv_logit(shifted_param / (pb.ub - pb.lb))
    return k * pb.scale
end

function box_parameter(free_param, pb::ParamBox)
    scaled_param = free_param / pb.scale
    k = (pb.ub == Inf) ? exp(scaled_param) : logit(scaled_param) * (pb.ub - pb.lb)
    return k + pb.lb
end

"""
Convert an unconstrained (n-1)-vector to a simplicial n-vector, z, such that
  - sum(z) = 1
  - z >= sb.lb
See notes for a derivation and reasoning.
"""
function simplexify_parameter!(free_param, sb::SimplexBox)
    @assert length(free_param) == (sb.n - 1)
    z = zeros(eltype(free_param), length(free_param) + 1)
    simplex_constrain!(z, free_param ./ sb.scale)
    for i in eachindex(z)
        z[i] = (1 - sb.n * sb.lb) * z[i] + sb.lb
    end
    return z
end

"""
Invert the transformation simplexify_parameter() by converting an n-vector
on a simplex to R^{n - 1}.
"""
function unsimplexify_parameter(param, sb::SimplexBox)
    @assert length(param) == sb.n
    # @assert all(k -> km >= sb.lb, param)
    # @assert (abs(sum(param) - 1) < 1e-14, abs(sum(param) - 1))
    z = [(p - sb.lb) / (1 - sb.n * sb.lb) for p in param]
    x = zeros(eltype(param), length(param) - 1)
    simplex_unconstrain(x, z)
    scale!(x, sb.scale)
    return x
end

end # module
