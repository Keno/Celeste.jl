using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform
using PyPlot

using ForwardDiff

import Synthetic
import WCS


# For debugging:
using ElboDeriv.bvn_ids
using ElboDeriv.eval_bvn_pdf
using ElboDeriv.BvnDerivs
NumType = Float64

println("Running hessian tests.")

x = Float64[2.0, 3.0]
sigma = Float64[1.0 0.2; 0.2 1.0]
offset = Float64[0.5, 0.5]
weight = 1.0

bvn = ElboDeriv.BvnComponent(offset, sigma, weight);
bvn_sf = ElboDeriv.bvn_derivs(bvn, x);


function f{T <: Number}(x::Vector{Number}, sigma::Matrix{Number})
  weight * ((x - offset)' * (sigma \ (x - offset)))[1,1]
end

function wrap(x::Vector{Float64}, sigma::Matrix{Float64})
  par = zeros(Float64, bvn_ids.length)
  par[bvn_ids.x] = x
  par[bvn_ids.sig] = [ sigma[1, 1], sigma[1, 2], sigma[2, 2]]
  par
end

function f_wrap{T <: Number}(par::Vector{T})
  x_loc = par[bvn_ids.x]
  s_vec = par[bvn_ids.sig]
  sig_loc = T[s_vec[1] s_vec[2]; s_vec[2] s_vec[3]]
  println(sig_loc)
  f(x_loc, sig_loc)
end

par = wrap(x, sigma);
@test_approx_eq bvn_sf.v f_wrap(par)

ad_g = ForwardDiff.gradient(f_wrap);
ad_g(par)



function test_brightness_hessian()
    blob, mp, three_bodies = gen_three_body_dataset();
    kept_ids = [ ids.r1; ids.r2; ids.c1[:]; ids.c2[:] ];
    omitted_ids = setdiff(1:length(ids), kept_ids);

    transform = Transform.get_identity_transform(length(ids), mp.S);
    ad_hess = zeros(Float64, length(kept_ids) * mp.S, length(kept_ids) * mp.S);
    for i = 1:Ia, b = [3,4,2,5,1], squares in [false, true]
        # Write the brightness as an ordinary function of a ModelParams
        # so we can use the objective function autodiff hessian logic.
        function wrap_source_brightness{NumType <: Number}(
            mp::ModelParams{NumType})
          ret = zero_sensitive_float(CanonicalParams, NumType, mp.S);
          for s=1:mp.S
            sb = ElboDeriv.SourceBrightness(mp.vp[s]);
            if squares
              ret.v = sb.E_l_a[b, i].v
              ret.d[:, s] = sb.E_l_a[b, i].d
              ret.hs[s][:, :] = sb.E_l_a[b, i].hs[1][:, :]
            else
              ret.v = sb.E_ll_a[b, i].v
              ret.d[:, s] = sb.E_ll_a[b, i].d
              ret.hs[s][:, :] = sb.E_ll_a[b, i].hs[1][:, :]
            end
          end
          ret
        end

        bright = wrap_source_brightness(mp);
        objective = OptimizeElbo.ObjectiveWrapperFunctions(
          wrap_source_brightness, mp, transform, kept_ids, omitted_ids);
        x = transform.vp_to_array(mp.vp, omitted_ids);

        # Sanity check.
        @test_approx_eq objective.f_value(x[:]) bright.v

        # Compare the AD hessian with the exact hessian.
        objective.f_ad_hessian!(x[:], ad_hess);
        for s=1:mp.S
          hess_ind = (1:length(kept_ids)) + (s - 1) * length(kept_ids)
          hess0 = ad_hess[hess_ind, hess_ind]
          hess1 = bright.hs[s][kept_ids, kept_ids]
          @test_approx_eq(
            ad_hess[hess_ind, hess_ind], bright.hs[s][kept_ids, kept_ids])
        end
    end
end


function test_multiply_sf()
  # Test for hessians.
  # Two sets of ids with some overlap and some disjointness.
  p = length(ids)
  S = 2

  ids1 = find((1:p) .% 2 .== 0)
  ids2 = setdiff(1:p, ids1)
  ids1 = union(ids1, 1:5)
  ids2 = union(ids2, 1:5)

  l1 = zeros(Float64, S * p);
  l2 = zeros(Float64, S * p);
  l1[ids1] = rand(length(ids1))
  l2[ids2] = rand(length(ids2))
  l1[ids1 + p] = rand(length(ids1))
  l2[ids2 + p] = rand(length(ids2))

  sigma1 = zeros(Float64, S * p, S * p);
  sigma2 = zeros(Float64, S * p, S * p);
  sigma1[ids1, ids1] = rand(length(ids1), length(ids1));
  sigma2[ids2, ids2] = rand(length(ids2), length(ids2));
  sigma1[ids1 + p, ids1 + p] = rand(length(ids1), length(ids1));
  sigma2[ids2 + p, ids2 + p] = rand(length(ids2), length(ids2));
  sigma1 = 0.5 * (sigma1 + sigma1')
  sigma2 = 0.5 * (sigma2 + sigma2')

  x = 0.1 * rand(S * p);

  function testfun1(x)
    (l1' * x + 0.5 * x' * sigma1 * x)[1,1]
  end

  function testfun2(x)
    (l2' * x + 0.5 * x' * sigma2 * x)[1,1]
  end

  function testfun(x)
    testfun1(x) * testfun2(x)
  end

  ret1 = zero_sensitive_float(CanonicalParams, Float64, S);
  ret2 = zero_sensitive_float(CanonicalParams, Float64, S);
  s_ind = Array(UnitRange{Int64}, 2);
  s_ind[1] = 1:p
  s_ind[2] = (1:p) + p

  ret1.v = testfun1(x)
  fill!(ret1.d, 0.0);
  for s=1:S
    fill!(ret1.hs[s], 0.0);
    ret1.d[:, s] = l1[s_ind[s]] + sigma1[s_ind[s], s_ind[s]] * x[s_ind[s]];
    ret1.hs[s] = sigma1[s_ind[s], s_ind[s]];
  end

  ret2.v = testfun2(x)
  fill!(ret2.d, 0.0);
  for s=1:S
    fill!(ret2.hs[s], 0.0);
    ret2.d[:, s] = l2[s_ind[s]] + sigma2[s_ind[s], s_ind[s]] * x[s_ind[s]];
    ret2.hs[s] = sigma2[s_ind[s], s_ind[s]];
  end

  hess = zeros(Float64, S * p, S * p);
  grad = ForwardDiff.gradient(testfun1, x);
  ForwardDiff.hessian!(hess, testfun1, x);
  for s=1:S
    @test_approx_eq(ret1.d[:, s], grad[s_ind[s]])
    @test_approx_eq(ret1.hs[s], hess[s_ind[s], s_ind[s]])
  end

  grad = ForwardDiff.gradient(testfun2, x);
  ForwardDiff.hessian!(hess, testfun2, x);
  for s=1:S
    @test_approx_eq(ret2.d[:, s], grad[s_ind[s]])
    @test_approx_eq(ret2.hs[s], hess[s_ind[s], s_ind[s]])
  end


  grad = ForwardDiff.gradient(testfun, x);
  ForwardDiff.hessian!(hess, testfun, x);

  sf1 = deepcopy(ret1);
  sf2 = deepcopy(ret2);
  multiply_sf!(sf1, sf2, ids1=ids1, ids2=ids2);

  for s=1:S
    @test_approx_eq(sf1.d[:, s], grad[s_ind[s]])
    @test_approx_eq(sf1.hs[s][:], hess[s_ind[s], s_ind[s]])
  end
end


function test_set_hess()
  sf = zero_sensitive_float(CanonicalParams, 2);
  CelesteTypes.set_hess!(sf, 2, 3, 5.0);
  @test_approx_eq sf.hs[1][2, 3] 5.0
  @test_approx_eq sf.hs[1][3, 2] 5.0

  CelesteTypes.set_hess!(sf, 4, 4, 6.0);
  @test_approx_eq sf.hs[1][4, 4] 6.0

  CelesteTypes.set_hess!(sf, 2, 3, 2, 7.0);
  @test_approx_eq sf.hs[2][2, 3] 7.0
  @test_approx_eq sf.hs[2][3, 2] 7.0

  CelesteTypes.set_hess!(sf, 4, 4, 2, 8.0);
  @test_approx_eq sf.hs[2][4, 4] 8.0
end



test_set_hess()
test_multiply_sf()
test_brightness_hessian()
