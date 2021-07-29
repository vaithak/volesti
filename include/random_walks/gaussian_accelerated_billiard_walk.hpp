// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2021 Vaibhav Thakkar

// Contributed and/or modified by Vaibhav Thakkar, as part of Google Summer of Code 2021 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_GAUSSIAN_ACCELERATED_BILLIARD_WALK_HPP
#define RANDOM_WALKS_GAUSSIAN_ACCELERATED_BILLIARD_WALK_HPP

#include "convex_bodies/orderpolytope.h"
#include "convex_bodies/ellipsoid.h"
#include "convex_bodies/ballintersectconvex.h"
#include "generators/boost_random_number_generator.hpp"
#include "sampling/ellipsoid.hpp"
#include "random_walks/uniform_billiard_walk.hpp"

template <typename Point>
struct compute_diameter<OrderPolytope<Point>>
{
template <typename NT>
static NT compute(OrderPolytope<Point> const& P)
{
    NT diameter = std::sqrt(NT(P.dimension()));
    return diameter;
}
};

template <typename Point>
struct compute_diameter<BallIntersectPolytope<OrderPolytope<Point>, Ellipsoid<Point> > >
{
template <typename NT>
static NT compute(BallIntersectPolytope<OrderPolytope<Point>, Ellipsoid<Point>> const& P)
{
    NT polytope_diameter = std::sqrt(NT(P.dimension()));
    return std::min(polytope_diameter, (NT(2) * P.radius()));
}
};


// Billiard walk which accelarates each step for uniform distribution and also takes into account
// the shape of the polytope for generating directions.

struct GaussianAcceleratedBilliardWalk
{
    GaussianAcceleratedBilliardWalk(double L)
            :   param(L, true)
    {}

    GaussianAcceleratedBilliardWalk()
            :   param(0, false)
    {}

    struct parameters
    {
        parameters(double L, bool set)
                :   m_L(L), set_L(set)
        {}
        double m_L;
        bool set_L;
    };

    struct update_parameters
    {
        update_parameters()
                :   facet_prev(0), hit_ball(false), inner_vi_ak(0.0), ball_inner_norm(0.0)
        {}
        int facet_prev;
        bool hit_ball;
        double inner_vi_ak;
        double ball_inner_norm;
    };

    parameters param;


    template
            <
                    typename Polytope,
                    typename RandomNumberGenerator
            >
    struct Walk
    {
        typedef typename Polytope::PointType Point;
        // typedef typename Polytope::MT MT;
        typedef typename Point::FT NT;

        template <typename GenericPolytope, typename MT>
        Walk(GenericPolytope const& P,
             Point const& p,
             MT const& L,   // cholesky matrix for Gaussian Direction
             RandomNumberGenerator &rng)
        {
            _update_parameters = update_parameters();
            _Len = compute_diameter<GenericPolytope>
                ::template compute<NT>(P);

            // Removed as will be used for sparse matrices only
            // _AA.noalias() = P.get_mat() * P.get_mat().transpose();
            initialize(P, p, L, rng);
        }

        template <typename GenericPolytope, typename MT>
        Walk(GenericPolytope const& P,
             Point const& p,
             MT const& L,   // cholesky matrix for Gaussian Direction
             RandomNumberGenerator &rng,
             parameters const& params)
        {
            _update_parameters = update_parameters();
            _Len = params.set_L ? params.m_L
                              : compute_diameter<GenericPolytope>
                                ::template compute<NT>(P);

            // Removed as will be used for sparse matrices only
            // _AA.noalias() = P.get_mat() * P.get_mat().transpose();
            initialize(P, p, L, rng);
        }

        template
                <
                        typename GenericPolytope,
                        typename MT
                >
        inline void apply(GenericPolytope const& P,
                          Point &p,       // a point to return the result
                          MT const& L,    // cholesky matrix for Gaussian Direction
                          unsigned int const& walk_length,
                          RandomNumberGenerator &rng)
        {
            unsigned int n = P.dimension();
            NT T;
            const NT dl = 0.995;
            int it;

            for (auto j=0u; j<walk_length; ++j)
            {
                T = -std::log(rng.sample_urdist()) * _Len;
                _v = GetGaussianDirection<Point>::apply(n, L, rng);
                Point p0 = _p;

                it = 0;
                while (it < 100*n)
                {
                    std::pair<NT, int> pbpair
                            = P.line_positive_intersect(_p, _v, _lambdas, _Av, _lambda_prev, _update_parameters);
                    if (T <= pbpair.first) {
                        _p += (T * _v);
                        _lambda_prev = T;
                        break;
                    }
                    _lambda_prev = dl * pbpair.first;
                    _p += (_lambda_prev * _v);
                    T -= _lambda_prev;
                    P.compute_reflection(_v, _p, _update_parameters);
                    it++;
                }
                if (it == 100*n) _p = p0;
            }
            p = _p;
        }

        inline void update_delta(NT L)
        {
            _Len = L;
        }

    private :

        template
                <
                        typename GenericPolytope,
                        typename MT
                >
        inline void initialize(GenericPolytope const& P,
                               Point const& p,  // a point to start
                               MT const& L,    // cholesky matrix for Gaussian Direction
                               RandomNumberGenerator &rng)
        {
            unsigned int n = P.dimension();
            const NT dl = 0.995;
            _lambdas.setZero(P.num_of_hyperplanes());
            _Av.setZero(P.num_of_hyperplanes());
            _p = p;
            _v = GetGaussianDirection<Point>::apply(n, L, rng);

            NT T = -std::log(rng.sample_urdist()) * _Len;
            Point p0 = _p;
            int it = 0;

            std::pair<NT, int> pbpair
                    = P.line_first_positive_intersect(_p, _v, _lambdas, _Av, _update_parameters);
            if (T <= pbpair.first) {
                _p += (T * _v);
                _lambda_prev = T;
                return;
            }
            _lambda_prev = dl * pbpair.first;
            _p += (_lambda_prev * _v);
            T -= _lambda_prev;
            P.compute_reflection(_v, _p, _update_parameters);

            while (it <= 100*n)
            {
                std::pair<NT, int> pbpair
                        = P.line_positive_intersect(_p, _v, _lambdas, _Av, _lambda_prev, _update_parameters);
                if (T <= pbpair.first) {
                    _p += (T * _v);
                    _lambda_prev = T;
                    break;
                } else if (it == 100*n) {
                    _lambda_prev = rng.sample_urdist() * pbpair.first;
                    _p += (_lambda_prev * _v);
                    break;
                }
                _lambda_prev = dl * pbpair.first;
                _p += (_lambda_prev * _v);
                T -= _lambda_prev;
                P.compute_reflection(_v, _p, _update_parameters);
                it++;
            }
        }

        NT _Len;
        Point _p;
        Point _v;
        NT _lambda_prev;
        // MT _AA;  // Removed as will be used for sparse matrices only
        update_parameters _update_parameters;
        typename Point::Coeff _lambdas;
        typename Point::Coeff _Av;
    };

};


#endif
