#pragma once

#include "gradient.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <utility>
#include <vector>

namespace diff::min {

namespace mp = boost::mp11;

namespace detail {

// Compile-time indices of each element of SubSyms within AllSyms.
// Both are mp_lists of std::integral_constant<char,C>; AllSyms is sorted.
// Returns std::array<std::size_t, mp_size<SubSyms>::value>.
template <typename AllSyms, typename SubSyms>
constexpr auto sub_indices() {
    constexpr std::size_t NS = mp::mp_size<SubSyms>::value;
    std::array<std::size_t, NS> idx{};
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((idx[I] = mp::mp_find<AllSyms, mp::mp_at_c<SubSyms, I>>::value), ...);
    }(std::make_index_sequence<NS>{});
    return idx;
}

} // namespace detail

// NR §15.5 — Levenberg-Marquardt nonlinear least-squares fitting.
//
// Minimizes χ² = Σᵢ [wᵢ (yᵢ − f(xᵢ; a))]²  over parameters a ∈ ℝᴺ.
//
// Template parameters:
//   Expr      — model expression containing both parameter and input Variables
//   ParamSyms — mp_list<ic<char>,...> of symbols treated as parameters (LM optimizes)
//   InputSyms — mp_list<ic<char>,...> of symbols treated as per-point inputs
//
// For each data point the expression is evaluated at the current parameter
// vector and the given input; the Jacobian row ∂f/∂aⱼ is obtained from
// reverse-mode AD for free.
template <CExpression Expr,
          typename ParamSyms = extract_symbols_from_expr_t<Expr>,
          typename InputSyms = mp::mp_list<>>
struct LevenbergMarquardt {
    using AllSyms    = extract_symbols_from_expr_t<Expr>;
    using value_type = typename Expr::value_type;

    static constexpr std::size_t N    = mp::mp_size<ParamSyms>::value;
    static constexpr std::size_t K    = mp::mp_size<InputSyms>::value;
    static constexpr std::size_t NALL = mp::mp_size<AllSyms>::value;

    using ParamVec = Eigen::Vector<value_type, static_cast<int>(N)>;
    using InputVec = Eigen::Vector<value_type, static_cast<int>(K)>;
    using AllVec   = Eigen::Vector<value_type, static_cast<int>(NALL)>;

    // Compile-time index mappings into the AllSyms-ordered gradient array.
    static constexpr auto PARAM_IDX = detail::sub_indices<AllSyms, ParamSyms>();
    static constexpr auto INPUT_IDX = detail::sub_indices<AllSyms, InputSyms>();

    struct DataPoint {
        InputVec   input;
        value_type target;
        value_type weight{1}; // 1/σᵢ — default: unweighted (σᵢ = 1)
    };

    Expr       expr;
    value_type ftol;
    int        itmax;

    static constexpr value_type LAMBDA_INIT{1e-3};
    static constexpr value_type LAMBDA_UP{10};
    static constexpr value_type LAMBDA_DOWN{0.1};

    explicit LevenbergMarquardt(Expr e,
                                value_type ftol_ = value_type{1e-8},
                                int        itmax_ = 1000)
        : expr(std::move(e)), ftol(ftol_), itmax(itmax_) {}

    // Build the AllSyms-sized update vector from params and a per-point input.
    AllVec make_all_vec(const ParamVec& params, const InputVec& input) const {
        AllVec v;
        for (std::size_t j = 0; j < N; ++j)
            v[static_cast<int>(PARAM_IDX[j])] = params[static_cast<int>(j)];
        for (std::size_t k = 0; k < K; ++k)
            v[static_cast<int>(INPUT_IDX[k])] = input[static_cast<int>(k)];
        return v;
    }

    // Evaluate residual vector r (length M) and Jacobian J (M×N) at params.
    // r[i] = wᵢ (yᵢ − f(xᵢ; a)),  J[i,j] = −wᵢ ∂f/∂aⱼ
    auto eval_rJ(const ParamVec& params, const std::vector<DataPoint>& data) {
        const int M = static_cast<int>(data.size());
        using DynVec = Eigen::VectorX<value_type>;
        using JMat   = Eigen::Matrix<value_type, Eigen::Dynamic,
                                     static_cast<int>(N)>;
        DynVec r(M);
        JMat   J(M, static_cast<int>(N));

        for (int i = 0; i < M; ++i) {
            const AllVec av = make_all_vec(params, data[i].input);
            expr.update(AllSyms{}, av);

            const value_type fi = expr.eval();
            r[i] = data[i].weight * (data[i].target - fi);

            const auto g = diff::gradient<DiffMode::Reverse>(expr);
            for (std::size_t j = 0; j < N; ++j)
                J(i, static_cast<int>(j)) =
                    -data[i].weight * g[PARAM_IDX[j]];
        }
        return std::pair{r, J};
    }

    // Fit parameters to data. Returns optimised parameter vector.
    ParamVec fit(ParamVec params, const std::vector<DataPoint>& data) {
        using std::abs;
        using NMat = Eigen::Matrix<value_type, static_cast<int>(N),
                                               static_cast<int>(N)>;
        using NVec = Eigen::Vector<value_type, static_cast<int>(N)>;

        value_type lambda = LAMBDA_INIT;
        auto [r, J]       = eval_rJ(params, data);
        value_type chi2   = r.squaredNorm();

        for (int iter = 0; iter < itmax; ++iter) {
            const NMat JtJ  = J.transpose() * J;
            const NVec beta = -(J.transpose() * r); // descent direction: −Jᵀr

            // Marquardt damping: α = Jᵀ J + λ diag(Jᵀ J)
            NMat alpha = JtJ;
            for (int j = 0; j < static_cast<int>(N); ++j)
                alpha(j, j) *= (value_type{1} + lambda);

            const NVec da        = alpha.ldlt().solve(beta);
            const ParamVec p_new = params + da;

            auto [r_new, J_new] = eval_rJ(p_new, data);
            const value_type chi2_new = r_new.squaredNorm();

            if (chi2_new < chi2) {
                // Step accepted
                lambda *= LAMBDA_DOWN;

                // Convergence: step small relative to current params
                if (da.norm() < ftol * (params.norm() + ftol))
                    return p_new;

                // Convergence: chi-squared barely changed
                if (abs(chi2 - chi2_new) < ftol * (value_type{1} + chi2))
                    return p_new;

                params = p_new;
                chi2   = chi2_new;
                r      = std::move(r_new);
                J      = std::move(J_new);
            } else {
                // Step rejected — increase damping, keep old params/J
                lambda *= LAMBDA_UP;
            }
        }
        return params;
    }
};

template <CExpression Expr>
LevenbergMarquardt(Expr) -> LevenbergMarquardt<Expr>;
template <CExpression Expr, typename T>
LevenbergMarquardt(Expr, T) -> LevenbergMarquardt<Expr>;

// Convenience alias for the common case where all symbols are parameters
// (no explicit input variable — user bakes x_i into separate expressions
//  or uses a single-variable expression).
template <CExpression Expr>
using LM = LevenbergMarquardt<Expr>;

} // namespace diff::min
