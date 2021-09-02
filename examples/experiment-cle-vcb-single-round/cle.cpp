#include "Eigen/Eigen"
#include <vector>
#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/ellipsoid.h"
#include "preprocess/max_inscribed_ellipsoid.hpp"
#include "random_walks/uniform_billiard_walk.hpp"
#include "random_walks/uniform_accelerated_billiard_walk.hpp"
#include "generators/boost_random_number_generator.hpp"
#include "volume_cooling_ellipsoids.hpp"

#include <iostream>
#include <fstream>
#include "misc.h"
#include "poset.h"

typedef double NT;
typedef Cartesian <NT> Kernel;
typedef typename Kernel::Point Point;
typedef BoostRandomNumberGenerator<boost::mt19937, NT> RNGType;
typedef HPolytope<Point> HPOLYTOPE;
typedef Ellipsoid<Point> EllipsoidType;
typedef typename HPOLYTOPE::MT MT;
typedef typename HPOLYTOPE::VT VT;

NT calculateLinearExtension(HPOLYTOPE& HP) {
    // Setup parameters for calculating volume and rounding
    unsigned int d = HP.dimension();
    unsigned int walk_len = 1;
    NT e = 0.1;

    // ------------------------------------------------------------------------------------------------
    unsigned int max_iter = 150;
    NT tol = std::pow(10, -6.0), reg = std::pow(10, -4.0);

    std::pair<Point, NT> innerBall = HP.ComputeInnerBall();
    if (innerBall.second < 0.0)
        throw std::runtime_error("no inscribed ball");

    VT x0 = innerBall.first.getCoefficients();
    Eigen::SparseMatrix<NT> A = HP.get_mat().sparseView();
    std::pair<std::pair<MT, VT>, bool> inscribed_ellipsoid_res = max_inscribed_ellipsoid<MT>(A,
                                                                                             HP.get_vec(),
                                                                                             x0,
                                                                                             max_iter,
                                                                                             tol,
                                                                                             reg);
    if (!inscribed_ellipsoid_res.second) // not converged
        throw std::runtime_error("no inscribed ellipsoid");

    MT A_ell = inscribed_ellipsoid_res.first.first.inverse();
    EllipsoidType inscribed_ellipsoid(A_ell);
    MT L = inscribed_ellipsoid.Lcov();

    HP.shift(inscribed_ellipsoid_res.first.second);
    NT round_val = L.transpose().determinant();
    HP.linear_transformIt(L);

    NT volume = volume_cooling_balls<AcceleratedBilliardWalk, RNGType>(HP, e, walk_len).second;
    volume = volume * round_val;

    std::cout << "reflections:" << HPOLYTOPE::reflection_count << "," << std::endl;

    // multiplying by d factorial, d = number of elements
    for(NT i=(NT)d; i>1; i-=1) {
        volume = volume * i;
    }
    

    return volume;
}

template <>
int HPOLYTOPE::reflection_count = 0;

/**

 Usage: ./cle INSTANCE

 example:
    ./cle instances/bipartite_0.5_008_0.txt

*/
int main(int argc, char* argv[]) {
    std::string filename (argv[1]);
    std::ifstream in;
    in.open(filename);
    std::pair<bool, Poset> read_poset = read_poset_from_file_adj_matrix(in);

    if( !read_poset.first ) {
        std::cerr << "error in reading data from file" << std::endl;
        return -1;
    }

    // ---------------- Create HPOLYTOPE ---------------------
    Poset poset = read_poset.second;
    int num_relations = poset.num_relations();
    int d = poset.num_elem();

    MT A = Eigen::MatrixXd::Zero(2*d + num_relations, d);
    VT b = Eigen::MatrixXd::Zero(2*d + num_relations, 1);

    // first add (ai >= 0) or (-ai <= 0) rows
    A.topLeftCorner(d, d) = -Eigen::MatrixXd::Identity(d, d);

    // next add (ai <= 1) rows
    A.block(d, 0, d, d) = Eigen::MatrixXd::Identity(d, d);
    b.block(d, 0, d, 1) = Eigen::MatrixXd::Constant(d, 1, 1.0);

    // next add the relations
    for(int idx=0; idx<num_relations; ++idx) {
        std::pair<int, int> edge  = poset.get_relation(idx);
        A(2*d + idx, edge.first)  = 1;
        A(2*d + idx, edge.second) = -1;
    }
    HPOLYTOPE HP(d, A, b);
    //--------------------------------------------------------

    std::cout << calculateLinearExtension(HP) << std::endl;
    return 0;
}
