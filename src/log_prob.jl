# Functions to compute the log probability of star parameters and galaxy
# parameters given pixel data



#using ..SensitiveFloats

include("bivariate_normals.jl")

using Distributions

import ..SensitiveFloats: SensitiveFloat

#using ..Model
#using Celeste
#import ..Model: PsfComponent, psf_K, galaxy_prototypes, D, Ia,
#                      prior, load_prior
#using Celeste: Model #, ElboDeriv, Infer
#import Celeste: WCSUtils, PSF, RunCamcolField, load_images
#import Celeste.ElboDeriv: ActivePixel, BvnComponent, GalaxyCacheComponent


###################################################################
# Structs formerly from ElboDeriv that are useful for both ELBO   #
# optimization and log_prob calculations                          #
###################################################################

"""
ElboArgs stores the arguments needed to evaluate the variational objective
function
"""
type ElboArgs{NumType <: Number}
    S::Int64
    N::Int64
    images::Vector{TiledImage}
    vp::VariationalParams{NumType}
    tile_source_map::Vector{Matrix{Vector{Int}}}
    patches::Matrix{SkyPatch}
    active_sources::Vector{Int}
end


function ElboArgs{NumType <: Number}(
            images::Vector{TiledImage},
            vp::VariationalParams{NumType},
            tile_source_map::Vector{Matrix{Vector{Int}}},
            patches::Matrix{SkyPatch},
            active_sources::Vector{Int})
    N = length(images)
    S = length(vp)
    @assert length(tile_source_map) == N
    @assert size(patches, 1) == S
    @assert size(patches, 2) == N
    ElboArgs(S, N, images, vp, tile_source_map, patches, active_sources)
end



# TODO: the identification of active pixels should go in pre-processing
type ActivePixel
    # image index
    n::Int

    # Linear tile index:
    tile_ind::Int

    # Location in tile:
    h::Int
    w::Int
end


"""
Get the active pixels (pixels for which the active sources are present).
TODO: move this to pre-processing and use it instead of setting low-signal
pixels to NaN.
"""
function get_active_pixels{NumType <: Number}(
                    ea::ElboArgs{NumType})
    active_pixels = ActivePixel[]

    for n in 1:ea.N, tile_ind in 1:length(ea.images[n].tiles)
        tile_sources = ea.tile_source_map[n][tile_ind]
        if length(intersect(tile_sources, ea.active_sources)) > 0
            tile = ea.images[n].tiles[tile_ind]
            h_width, w_width = size(tile.pixels)
            for w in 1:w_width, h in 1:h_width
                if !Base.isnan(tile.pixels[h, w])
                    push!(active_pixels, ActivePixel(n, tile_ind, h, w))
                end
            end
        end
    end

    active_pixels
end


type ElboIntermediateVariables{NumType <: Number}

    bvn_derivs::BivariateNormalDerivatives{NumType}

    # Vectors of star and galaxy bvn quantities from all sources for a pixel.
    # The vector has one element for each active source, in the same order
    # as ea.active_sources.

    # TODO: you can treat this the same way as E_G_s and not keep a vector around.
    fs0m_vec::Vector{SensitiveFloat{StarPosParams, NumType}}
    fs1m_vec::Vector{SensitiveFloat{GalaxyPosParams, NumType}}

    # Brightness values for a single source
    E_G_s::SensitiveFloat{CanonicalParams, NumType}
    E_G2_s::SensitiveFloat{CanonicalParams, NumType}
    var_G_s::SensitiveFloat{CanonicalParams, NumType}

    # Subsets of the Hessian of E_G_s and E_G2_s that allow us to use BLAS
    # functions to accumulate Hessian terms. There is one submatrix for
    # each celestial object type in 1:Ia
    E_G_s_hsub_vec::Vector{HessianSubmatrices{NumType}}
    E_G2_s_hsub_vec::Vector{HessianSubmatrices{NumType}}

    # Expected pixel intensity and variance for a pixel from all sources.
    E_G::SensitiveFloat{CanonicalParams, NumType}
    var_G::SensitiveFloat{CanonicalParams, NumType}

    # Pre-allocated memory for the gradient and Hessian of combine functions.
    combine_grad::Vector{NumType}
    combine_hess::Matrix{NumType}

    # A placeholder for the log term in the ELBO.
    elbo_log_term::SensitiveFloat{CanonicalParams, NumType}

    # The ELBO itself.
    elbo::SensitiveFloat{CanonicalParams, NumType}

    # If false, do not calculate hessians or derivatives.
    calculate_derivs::Bool

    # If false, do not calculate hessians.
    calculate_hessian::Bool
end


"""
Args:
    - S: The total number of sources
    - num_active_sources: The number of actives sources (with deriviatives)
    - calculate_derivs: If false, only calculate values
    - calculate_hessian: If false, only calculate gradients. Note that if
                calculate_derivs = false, then hessians will not be
                calculated irrespective of the value of calculate_hessian.
"""
function ElboIntermediateVariables(NumType::DataType,
                                   S::Int,
                                   num_active_sources::Int;
                                   calculate_derivs::Bool=true,
                                   calculate_hessian::Bool=true)
    @assert NumType <: Number

    bvn_derivs = BivariateNormalDerivatives{NumType}(NumType)

    # fs0m and fs1m accumulate contributions from all bvn components
    # for a given source.
    fs0m_vec = Array(SensitiveFloat{StarPosParams, NumType}, S)
    fs1m_vec = Array(SensitiveFloat{GalaxyPosParams, NumType}, S)
    for s = 1:S
        fs0m_vec[s] = zero_sensitive_float(StarPosParams, NumType)
        fs1m_vec[s] = zero_sensitive_float(GalaxyPosParams, NumType)
    end

    E_G_s = zero_sensitive_float(CanonicalParams, NumType, 1)
    E_G2_s = zero_sensitive_float(CanonicalParams, NumType, 1)
    var_G_s = zero_sensitive_float(CanonicalParams, NumType, 1)

    E_G_s_hsub_vec =
        HessianSubmatrices{NumType}[ HessianSubmatrices(NumType, i) for i=1:Ia ]
    E_G2_s_hsub_vec =
        HessianSubmatrices{NumType}[ HessianSubmatrices(NumType, i) for i=1:Ia ]

    E_G = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)
    var_G = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)

    combine_grad = zeros(NumType, 2)
    combine_hess = zeros(NumType, 2, 2)

    elbo_log_term =
        zero_sensitive_float(CanonicalParams, NumType, num_active_sources)
    elbo = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)

    ElboIntermediateVariables{NumType}(
        bvn_derivs, fs0m_vec, fs1m_vec,
        E_G_s, E_G2_s, var_G_s, E_G_s_hsub_vec, E_G2_s_hsub_vec,
        E_G, var_G, combine_grad, combine_hess,
        elbo_log_term, elbo, calculate_derivs, calculate_hessian)
end




"""
Populate fs0m_vec and fs1m_vec for all sources for a given pixel.

Args:
    - elbo_vars: Elbo intermediate values.
    - ea: Model parameters
    - tile_sources: A vector of integers of sources in 1:ea.S affecting the tile
    - tile: An ImageTile
    - h, w: The integer locations of the pixel within the tile
    - gal_mcs: Galaxy components
    - star_mcs: Star components

Returns:
    Updates elbo_vars.fs0m_vec and elbo_vars.fs1m_vec in place with the total
    shape contributions to this pixel's brightness.
"""
function populate_fsm_vecs!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    ea::ElboArgs{NumType},
                    tile_sources::Vector{Int},
                    tile::ImageTile,
                    h::Int, w::Int,
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    star_mcs::Array{BvnComponent{NumType}, 2})
    x = Float64[tile.h_range[h], tile.w_range[w]]
    for s in tile_sources
        # ensure tile.b is a filter band, not an image's index
        @assert 1 <= tile.b <= B
        wcs_jacobian = ea.patches[s, tile.b].wcs_jacobian
        active_source = s in ea.active_sources

        calculate_hessian =
            elbo_vars.calculate_hessian && elbo_vars.calculate_derivs && active_source
        clear!(elbo_vars.fs0m_vec[s], calculate_hessian)
        for k = 1:psf_K # PSF component
            accum_star_pos!(
                elbo_vars, s, star_mcs[k, s], x, wcs_jacobian, active_source)
        end

        clear!(elbo_vars.fs1m_vec[s], calculate_hessian)
        for i = 1:2 # Galaxy types
            for j in 1:8 # Galaxy component
                # If i == 2 then there are only six galaxy components.
                if (i == 1) || (j <= 6)
                    for k = 1:psf_K # PSF component
                        accum_galaxy_pos!(
                            elbo_vars, s, gal_mcs[k, j, i, s], x, wcs_jacobian,
                            active_source)
                    end
                end
            end
        end
    end
end



###########################################################################
# prior data types - for computing prior log probs of star/gal parameters #
###########################################################################
type StarPrior
    brightness::LogNormal
    color_component::Categorical
    colors::Vector{MvLogNormal}
end


type GalaxyPrior
    brightness::LogNormal
    color_component::Categorical
    colors::Vector{MvLogNormal}
    gal_scale::LogNormal
    gal_ab::Beta
    gal_fracdev::Beta
end


type Prior
    is_star::Bernoulli
    star::StarPrior
    galaxy::GalaxyPrior
end


###################################
# log likelihood function makers  #
###################################


"""
Creates a vectorized version of the star logpdf

Args:
  - images: Vector of TiledImage types (data for log_likelihood)
  - active_pixels: Vector of ActivePixels on which the log_likelihood is based
  - ea: ElboArgs book keeping argument - keeps the
"""
function make_star_logpdf(images::Vector{TiledImage},
                          active_pixels::Vector{ActivePixel},
                          ea::ElboArgs)

    # define star prior log probability density function
    prior    = construct_prior()
    subprior = prior.star

    function star_logprior(state::Vector{Float64})
        brightness, colors, u = state[1], state[2:5], state[6:end]
        return color_logprior(brightness, colors, prior, true)
    end

    # define star log joint probability density function
    function star_logpdf(state::Vector{Float64})
        ll_prior = star_logprior(state)

        brightness, colors, position = state[1], state[2:5], state[6:end]
        dummy_gal_shape = [.1, .1, .1, .1]
        ll_like  = state_log_likelihood(true, brightness, colors, position,
                                        dummy_gal_shape, images,
                                        active_pixels, ea)
        return ll_like + ll_prior
    end

    return star_logpdf, star_logprior
end


function make_galaxy_logpdf(images::Vector{TiledImage},
                            active_pixels::Vector{ActivePixel},
                            ea::ElboArgs)

    # define star prior log probability density function
    prior    = load_prior()
    subprior = prior.galaxy

    function galaxy_logprior(state::Vector{Float64})
        brightness, colors, u, gal_shape = state[1], state[2:5], state[6:7], state[8:end]
        # brightness prior
        ll_b = color_logprior(brightness, colors, prior, true)
        ll_s = shape_logprior(gal_shape, prior)
        return ll_b + ll_s
    end

    # define star log joint probability density function
    function galaxy_logpdf(state::Vector{Float64})
        ll_prior = galaxy_logprior(state)

        brightness, colors, position, gal_shape = state[1], state[2:5], state[6:7], state[8:end]
        ll_like  = state_log_likelihood(false, brightness, colors, position,
                                        gal_shape, images,
                                        active_pixels, ea)
        return ll_like + ll_prior
    end
    return galaxy_logpdf, galaxy_logprior
end


"""
Log likelihood of a single source given source params.

Args:
  - is_star: bool describing the type of source
  - brightness: log r-band value
  - colors: array of colors
  - position: ra/dec of source
  - gal_shape: vector of galaxy shape params (used if is_star=false)
  - images: vector of TiledImage types (data for likelihood)
  - active_pixels: vector of ActivePixels over which the ll is summed
  - ea: ElboArgs object that maintains params for all sources

Returns:
  - result: a scalar describing the log likelihood of the
            (brightness,colors,position,gal_shape) params conditioned on
            the rest of the args
"""
function state_log_likelihood(is_star::Bool,
                              brightness::Float64,
                              colors::Vector{Float64},
                              position::Vector{Float64},
                              gal_shape::Vector{Float64},
                              images::Vector{TiledImage},
                              active_pixels::Vector{ActivePixel},
                              ea::ElboArgs)

    # TODO: cache the background rate image!! --- does not need to be recomputed at each ll eval

    # convert brightness/colors to fluxes for scaling
    fluxes = colors_to_fluxes(brightness, colors)

    # make sure elbo-args reflects the position and galaxy shape passed in for
    # the first source in the elbo args (first is current source, the rest are
    # conditioned on)
    ea.vp[1][ids.u[1]]    = position[1]
    ea.vp[1][ids.u[2]]    = position[2]
    ea.vp[1][ids.e_dev]   = gal_shape[1]
    ea.vp[1][ids.e_axis]  = gal_shape[2]
    ea.vp[1][ids.e_angle] = gal_shape[3]
    ea.vp[1][ids.e_scale] = gal_shape[4]

    # create objects needed to compute the mean poisson value per pixel
    # (similar to ElboDeriv.process_active_pixels!)
    elbo_vars =
      ElboIntermediateVariables(Float64, ea.S, length(ea.active_sources))
    elbo_vars.calculate_derivs = false
    elbo_vars.calculate_hessian= false

    # load star/gal mixture components
    star_mcs_vec = Array(Array{BvnComponent{Float64}, 2}, ea.N)
    gal_mcs_vec  = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N)
    for b=1:ea.N
        star_mcs_vec[b], gal_mcs_vec[b] =
            load_bvn_mixtures(ea, b,
                              calculate_derivs=elbo_vars.calculate_derivs,
                              calculate_hessian=elbo_vars.calculate_hessian)
    end

    # iterate over the pixels, summing pixel-specific poisson rates
    ll = 0.
    for pixel in active_pixels
        tile         = ea.images[pixel.n].tiles[pixel.tile_ind]
        tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
        this_pixel   = tile.pixels[pixel.h, pixel.w]
        pixel_band   = tile.b

        # compute the unit-flux pixel values
        populate_fsm_vecs!(elbo_vars, ea, tile_sources, tile,
                           pixel.h, pixel.w,
                           gal_mcs_vec[pixel.n], star_mcs_vec[pixel.n])

        # TODO incorporate background rate for pixel into this epsilon_mat
        # compute the background rate for this pixel
        background_rate = tile.epsilon_mat[pixel.h, pixel.w]
        #for s in tile_sources
        #    println("tile source s: ", s)
        #    state = states[s]
        #    rate += state.is_star ? elbo_vars.fs0m_vec[s].v : elbo_vars.fs1m_vec[s].v
        #end

        # this source's rate, add to background for total
        this_rate  = is_star ? elbo_vars.fs0m_vec[1].v : elbo_vars.fs1m_vec[1].v
        pixel_rate = fluxes[pixel_band]*this_rate + background_rate

        # multiply by image's gain for this pixel
        rate     = pixel_rate * tile.iota_vec[pixel.h]
        pixel_ll = logpdf(Poisson(rate[1]), round(Int, this_pixel))
        ll += pixel_ll
    end
    return ll
end


#####################
# util funs         #
#####################


function color_logprior(brightness::Float64,
                        colors::Vector{Float64},
                        prior::Prior,
                        is_star::Bool)
    subprior = is_star ? prior.star : prior.galaxy
    ll_brightness = logpdf(subprior.brightness, brightness)
    ll_component  = [logpdf(subprior.colors[k], colors) for k in 1:2]
    ll_color      = logsumexp(ll_component + log(subprior.color_component.p))
    return ll_brightness + ll_color
end


function shape_logprior(gal_shape::Vector{Float64}, prior::Prior)
    # position and gal_angle have uniform priors--we ignore them
    gdev, gaxis, gangle, gscale = gal_shape
    ll_shape = 0.
    ll_shape += logpdf(prior.galaxy.gal_scale, gscale)
    ll_shape += logpdf(prior.galaxy.gal_ab, gaxis)
    ll_shape += logpdf(prior.galaxy.gal_fracdev, gdev)
    return ll_shape
end


function logsumexp(a::Vector{Float64})
    a_max = maximum(a)
    out = log(sum(exp(a - a_max)))
    out += a_max
    return out
end


function colors_to_fluxes(brightness::Float64, colors::Vector{Float64})
    ret    = Array(Float64, 5)
    ret[3] = exp(brightness)
    ret[4] = ret[3] * exp(colors[3]) #vs[ids.c1[3, i]])
    ret[5] = ret[4] * exp(colors[4]) #vs[ids.c1[4, i]])
    ret[2] = ret[3] / exp(colors[2]) #vs[ids.c1[2, i]])
    ret[1] = ret[2] / exp(colors[1]) #vs[ids.c1[1, i]])
    ret
end


function construct_prior()
    pp = load_prior()

    star_prior = StarPrior(
        LogNormal(pp.r_mean[1], pp.r_var[1]),
        Categorical(pp.k[:,1]),
        [MvLogNormal(pp.c_mean[:,k,1], pp.c_cov[:,:,k,1]) for k in 1:2])

    gal_prior = GalaxyPrior(
        LogNormal(pp.r_mean[2], pp.r_var[2]),
        Categorical(pp.k[:,2]),
        [MvLogNormal(pp.c_mean[:,k,2], pp.c_cov[:,:,k,2]) for k in 1:2],
        LogNormal(0, 10),
        Beta(1, 1),
        Beta(1, 1))

    return Prior(Bernoulli(.5), star_prior, gal_prior)
end
