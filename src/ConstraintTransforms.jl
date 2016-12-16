# Convert between different parameterizations.

module ConstraintTransforms

# TODO: don't import Model; transformations should operate on
# generic ParamSets
using ..Model
using ..SensitiveFloats
import ..Log

export DataTransform, ParamBounds, BoxConstraint, SimplexConstraint,
       get_mp_transform, enforce_bounds!,
       VariationalParams, FreeVariationalParams

# A vector of variational parameters.  The outer index is
# of celestial objects, and the inner index is over individual
# parameters for that object (referenced using ParamIndex).
typealias VariationalParams{NumType <: Number} Vector{Vector{NumType}}
typealias FreeVariationalParams{NumType <: Number} Vector{Vector{NumType}}

# The vector of transform parameters for a Symbol.
typealias ParamBounds Dict{Symbol, Union{Vector{BoxConstraint}, Vector{SimplexConstraint}}}

##################################
# Elementary Transform Functions #
##################################

inv_logit(x) = -log(1.0 / x - 1)

logit(x) = 1.0 / (1.0 + exp(-x))

function simplexify!(out, x)
    T = eltype(out)
    out[end] = zero(T)
    has_infs = any(isinf, x)
    if has_infs
        for i in eachindex(x)
            out[i] = ifelse(x[i] == Inf, one(T), zero(T))
        end
        out_sum = sum(out)
    else
        for i in eachindex(x)
            out[i] = exp(x[i])
        end
        out_sum = sum(out) + one(T)
    end
    for i in eachindex(x)
        out[i] /= out_sum
    end
    has_infs && (out[end] = inv(out_sum))
    return out
end

function unsimplexify!(out, x)
    log_last = log(last(x)))
    for i in eachindex(out)
        out[i] = log(x[i]) - log_last
    end
    return out
end

#########
# Types #
#########

abstract Constraint

immutable BoxConstraint <: Constraint
    lower::Float64
    upper::Float64
    scale::Float64
    function BoxConstraint(lower, upper, scale)
        @assert lower > -Inf
        @assert scale > 0.0
        @assert lower < upper
        return new(lower, upper, scale)
    end
end

immutable SimplexConstraint <: Constraint
    lower::Float64
    scale::Float64
    n::Int
    function SimplexConstraint(lower, scale, n)
        @assert n >= 2
        @assert 0.0 <= lower < 1 / n
        return new(lower, scale, n)
    end
end

immutable Transform{C<:Constraint}
    free_indices::Vector{Int}
    bound_indices::Vector{Int}
    constraints::Vector{C}
end

typealias BoxTransform Transform{BoxConstraint}
typealias SimplexTransform Transform{SimplexConstraint}

###########################################################
# `constrain`/`constrain!` & `unconstrain`/`unconstrain!` #
###########################################################

unconstrain!(out, x, transforms::Vector) = (for t in transforms; unconstrain!(out, x, t); end)

constrain!(out, x, transforms::Vector) = (for t in transforms; constrain!(out, x, t); end)

# box #
#-----#

# unconstrain/unconstrain!

function unconstrain(x, c::BoxConstraint)
    shifted = x - c.lower
    k = isinf(c.upper) ? log(shifted) : inv_logit(shifted / (c.upper - c.lower))
    return k * c.scale
end

function unconstrain!(out, x, transform::BoxTransform)
    for i in eachindex(transform.bound_indices)
        free_index = transform.free_indices[i]
        bound_index = transform.bound_indices[i]
        constraint = transform.constraints[i]
        out[free_index] = unconstrain(x[bound_index], constraint)
    end
    return out
end

# constrain/constrain!

function constrain(x, c::BoxConstraint)
    scaled = x / c.scale
    k = isinf(c.upper) ? exp(scaled) : (logit(scaled) * (c.upper - c.lower))
    return k + c.lower
end

function constrain!(out, x, transform::BoxTransform)
    for i in eachindex(transform.bound_indices)
        free_index = transform.free_indices[i]
        bound_index = transform.bound_indices[i]
        constraint = transform.constraints[i]
        out[bound_index] = unconstrain(x[free_index], constraint)
    end
    return out
end

# simplex #
#---------#

# unconstrain/unconstrain!

function unconstrain!(out, x, c::SimplexConstraint)
    for i in eachindex(x)
        x[i] = (x[i] - c.lower) / (1 - c.n * c.lower)
    end
    unsimplexify!(out, x)
    scale!(out, box.scale)
    return out
end

function unconstrain!(out, x, transform::SimplexTransform)
    for i in 1:size(transform.bound_indices, 2)
        x_i = x[view(transform.bound_indices, :, i)]
        out_i = view(out, view(transform.free_indices, :, i))
        constraint = transform.constraints[i]
        unconstrain!(out_i, x_i, constraint)
    end
    return out
end

# constrain/constrain!

function constrain!(out, x, c::SimplexConstraint)
    scale!(inv(c.scale), x)
    simplexify!(out, x)
    for i in eachindex(out)
        out[i] = (1 - c.n * c.lower) * out[i] + c.lower
    end
    return out
end

function constrain!(out, x, transform::SimplexTransform)
    for i in 1:size(transform.bound_indices, 2)
        x_i = x[view(transform.free_indices, :, i)]
        out_i = view(out, view(transform.bound_indices, :, i))
        constraint = transform.constraints[i]
        unconstrain!(out_i, x_i, constraint)
    end
    return out
end

end # module
