"""
Store pre-allocated memory in this data structures, which contains
intermediate values used in the ELBO calculation.
"""
type HessianSubmatrices{NumType <: Number}
    u_u::Matrix{NumType}
    shape_shape::Matrix{NumType}
end


"""
Pre-allocated memory for efficiently accumulating certain sub-matrices
of the E_G_s and E_G2_s Hessian.

Args:
    NumType: The numeric type of the hessian.
    i: The type of celestial source, from 1:Ia
"""
function HessianSubmatrices(NumType::DataType, i::Int)
    @assert 1 <= i <= Ia
    shape_p = length(shape_standard_alignment[i])

    u_u = zeros(NumType, 2, 2)
    shape_shape = zeros(NumType, shape_p, shape_p)
    HessianSubmatrices{NumType}(u_u, shape_shape)
end


type ElboIntermediateVariables{NumType <: Number}

    bvn_derivs::BivariateNormalDerivatives{NumType}

    # Vectors of star and galaxy bvn quantities from all sources for a pixel.
    # The vector has one element for each active source, in the same order
    # as ea.active_sources.

    # TODO: you can treat this the same way as E_G_s and not keep a vector around.
    fs0m_vec::Vector{SensitiveFloat{NumType}}
    fs1m_vec::Vector{SensitiveFloat{NumType}}

    # Brightness values for a single source
    E_G_s::SensitiveFloat{NumType}
    E_G2_s::SensitiveFloat{NumType}
    var_G_s::SensitiveFloat{NumType}

    # Subsets of the Hessian of E_G_s and E_G2_s that allow us to use BLAS
    # functions to accumulate Hessian terms. There is one submatrix for
    # each celestial object type in 1:Ia
    E_G_s_hsub_vec::Vector{HessianSubmatrices{NumType}}
    E_G2_s_hsub_vec::Vector{HessianSubmatrices{NumType}}

    # Expected pixel intensity and variance for a pixel from all sources.
    E_G::SensitiveFloat{NumType}
    var_G::SensitiveFloat{NumType}

    # Pre-allocated memory for the gradient and Hessian of combine functions.
    combine_grad::Vector{NumType}
    combine_hess::Matrix{NumType}

    # A placeholder for the log term in the ELBO.
    elbo_log_term::SensitiveFloat{NumType}

    # The ELBO itself.
    elbo::SensitiveFloat{NumType}
end


"""
Args:
    - S: The total number of sources
    - num_active_sources: The number of actives sources (with deriviatives)
    - calculate_gradient: If false, only calculate values
    - calculate_hessian: If false, only calculate gradients. Note that if
                calculate_gradient = false, then hessians will not be
                calculated irrespective of the value of calculate_hessian.
"""
function ElboIntermediateVariables(NumType::DataType,
                                   S::Int,
                                   num_active_sources::Int;
                                   calculate_gradient::Bool=true,
                                   calculate_hessian::Bool=true)
    @assert NumType <: Number

    bvn_derivs = BivariateNormalDerivatives{NumType}(NumType)

    # fs0m and fs1m accumulate contributions from all bvn components
    # for a given source.
    fs0m_vec = Array(SensitiveFloat{NumType}, S)
    fs1m_vec = Array(SensitiveFloat{NumType}, S)
    for s = 1:S
        fs0m_vec[s] = SensitiveFloat{NumType}(length(StarPosParams), 1,
                                    calculate_gradient, calculate_hessian)
        fs1m_vec[s] = SensitiveFloat{NumType}(length(GalaxyPosParams), 1,
                                    calculate_gradient, calculate_hessian)
    end

    E_G_s = SensitiveFloat{NumType}(length(CanonicalParams), 1,
                                    calculate_gradient, calculate_hessian)
    E_G2_s = SensitiveFloat(E_G_s)
    var_G_s = SensitiveFloat(E_G_s)

    E_G_s_hsub_vec =
        HessianSubmatrices{NumType}[ HessianSubmatrices(NumType, i) for i=1:Ia ]
    E_G2_s_hsub_vec =
        HessianSubmatrices{NumType}[ HessianSubmatrices(NumType, i) for i=1:Ia ]

    E_G = SensitiveFloat{NumType}(length(CanonicalParams), num_active_sources,
                                  calculate_gradient, calculate_hessian)
    var_G = SensitiveFloat(E_G)

    combine_grad = zeros(NumType, 2)
    combine_hess = zeros(NumType, 2, 2)

    elbo_log_term = SensitiveFloat(E_G)
    elbo = SensitiveFloat(E_G)

    ElboIntermediateVariables{NumType}(
        bvn_derivs, fs0m_vec, fs1m_vec,
        E_G_s, E_G2_s, var_G_s, E_G_s_hsub_vec, E_G2_s_hsub_vec,
        E_G, var_G, combine_grad, combine_hess,
        elbo_log_term, elbo)
end


function clear!{NumType <: Number}(elbo_vars::ElboIntermediateVariables{NumType})
    #TODO: don't allocate memory here?
    elbo_vars.bvn_derivs = BivariateNormalDerivatives{NumType}(NumType)

    for s = 1:length(elbo_vars.fs0m_vec)
        clear!(elbo_vars.fs0m_vec[s])
        clear!(elbo_vars.fs1m_vec[s])
    end

    clear!(elbo_vars.E_G_s)
    clear!(elbo_vars.E_G2_s)
    clear!(elbo_vars.var_G_s)

    for i in 1:Ia
        fill!(elbo_vars.E_G_s_hsub_vec[i].u_u, zero(NumType))
        fill!(elbo_vars.E_G_s_hsub_vec[i].shape_shape, zero(NumType))
        fill!(elbo_vars.E_G2_s_hsub_vec[i].u_u, zero(NumType))
        fill!(elbo_vars.E_G2_s_hsub_vec[i].shape_shape, zero(NumType))
    end

    clear!(elbo_vars.E_G)
    clear!(elbo_vars.var_G)

    fill!(elbo_vars.combine_grad, zero(NumType))
    fill!(elbo_vars.combine_hess, zero(NumType))

    clear!(elbo_vars.elbo_log_term)
    clear!(elbo_vars.elbo)
end


"""
If Infs/NaNs have crept into the ELBO evaluation (a symptom of poorly conditioned optimization),
this helps catch them immediately.
"""
function assert_all_finite{NumType <: Number}(sf::SensitiveFloat{NumType})
    @assert isfinite(sf.v[]) "Value is Inf/NaNs"
    @assert all(isfinite, sf.d) "Gradient contains Inf/NaNs"
    @assert all(isfinite, sf.h) "Hessian contains Inf/NaNs"
end


"""
Some parameter to a function has invalid values. The message should explain what parameter is
invalid and why.
"""
type InvalidInputError <: Exception
    message::String
end


"""
ElboArgs stores the arguments needed to evaluate the variational objective
function.
"""
type ElboArgs{NumType <: Number}
    # the overall number of sources: we don't necessarily visit them
    # all or optimize them all, but if we do visit a pixel where any
    # of these are active, we use it in the elbo calculation
    S::Int64

    # the number of images
    N::Int64

    # The number of components in the point spread function.
    psf_K::Int64
    images::Vector{Image}
    vp::VariationalParams{NumType}

    # subimages is a better name for patches: regions of an image
    # around a particular light source
    patches::Matrix{SkyPatch}

    # the sources to optimize
    active_sources::Vector{Int}

    # Bivarite normals will not be evaulated at points further than this many
    # standard deviations away from their mean.  See its usage in the ELBO and
    # bivariate normals for details.
    #
    # If this is set to Inf, the bivariate normals will be evaluated at all
    # points irrespective of their distance from the mean.
    num_allowed_sd::Float64

    elbo_vars::ElboIntermediateVariables
end


function ElboArgs{NumType <: Number}(
            images::Vector{Image},
            vp::VariationalParams{NumType},
            patches::Matrix{SkyPatch},
            active_sources::Vector{Int};
            psf_K::Int=2,
            num_allowed_sd::Float64=Inf,
            calculate_gradient=true,
            calculate_hessian=true)
    N = length(images)
    S = length(vp)

    @assert psf_K > 0
    @assert size(patches, 1) == S
    @assert size(patches, 2) == N

    for img in images, ep in img.epsilon_mat
        if ep <= 0.0
            msg = string("You must set all values of epsilon_mat > 0 ",
                         "for all images included in ElboArgs")
            throw(InvalidInputError(msg))
        end
    end

    @assert(length(active_sources) <= 5 || !calculate_hessian,
            "too many active_sources to store a hessian")
    @assert(all([all(isfinite, vs) for vs in vp]),
            "VariationalParameters contains NaNs or Infs")

    elbo_vars = ElboIntermediateVariables(NumType,
                                          S,
                                          length(active_sources);
                                          calculate_gradient=calculate_gradient,
                                          calculate_hessian=calculate_hessian)
    ElboArgs(S, N, psf_K, images, vp, patches,
             active_sources, num_allowed_sd, elbo_vars)
end
