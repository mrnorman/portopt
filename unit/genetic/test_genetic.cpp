
#include "YAKL.h"
#include "Gradient_Descent_Adam_FD.h"
#include "Genetic_Algorithm.h"

typedef double real;
typedef yakl::Array<real      ,1,yakl::memDevice> real1d;
typedef yakl::Array<real const,1,yakl::memDevice> realConst1d;
typedef yakl::Array<real const,2,yakl::memDevice> realConst2d;
real constexpr constr_tol = 1.e-2;
using yakl::c::parallel_for;
using yakl::c::SimpleBounds;

real1d constraint_cost(realConst2d params) {
  int population_size = params.extent(1);
  real1d cost("cost",population_size);
  parallel_for( YAKL_AUTO_LABEL() , population_size , YAKL_LAMBDA (int iens) {
    real a21 = params(0,iens);
    real a31 = params(1,iens);
    real a32 = params(2,iens);
    real b1  = params(3,iens);
    real b2  = params(4,iens);
    real b3  = params(5,iens);
    real cond1 = std::abs( 1.000000000000000000*b1+1.000000000000000000*b2+1.000000000000000000*b3-1.000000000000000000 );
    real cond2 = std::abs( 1.000000000000000000*a21*b2+1.000000000000000000*a31*b3+1.000000000000000000*a32*b3-0.5000000000000000000 );
    real cond3 = std::abs( 1.000000000000000000*(a21*a21)*b2+1.000000000000000000*(a31*a31)*b3+2.000000000000000000*a31*a32*b3+1.000000000000000000*(a32*a32)*b3-0.3333333333333333333 );
    real cond4 = std::abs( 1.000000000000000000*a21*a32*b3-0.1666666666666666667 );
    real cond_cost = std::max(cond1,cond2);
    cond_cost = std::max(cond_cost,cond3);
    cond_cost = std::max(cond_cost,cond4);
    cost(iens) = cond_cost;
  });
  return cost;
};


real1d closeness_cost(realConst2d params, realConst1d params_init) {
  int num_parameters  = params.extent(0);
  int population_size = params.extent(1);
  real1d cost("cost",population_size);
  parallel_for( YAKL_AUTO_LABEL() , population_size , YAKL_LAMBDA (int iens) {
    real num = 0;
    real den = 0;
    for (int iparam = 0; iparam < num_parameters; iparam++) {
      num += std::abs( params(iparam,iens) - params_init(iparam) );
      den += std::abs( params_init(iparam) );
    }
    cost(iens) = num / std::max(1.e-20,den);
  });
  return cost;
};


real1d compute_cost(realConst2d params, realConst1d params_init) {
  auto constr = constraint_cost(params);
  auto close  = closeness_cost (params,params_init);
  for (int iens = 0; iens < params.extent(1); iens++) {
    constr(iens) = std::max(0.,constr(iens)-constr_tol)*1.e10 + close(iens);
  }
  return constr;
};

int main( int argc , char **argv ) {
  MPI_Init(&argc,&argv);
  yakl::init();
  {
    int num_parameters = 6;
    real1d params_init("params_init",num_parameters);
    params_init = 0.5;

    int num_generations = 1e2;
    real1d lbounds("lbounds",num_parameters);
    real1d ubounds("ubounds",num_parameters);
    lbounds = -0.5;
    ubounds = 2;
    portopt::Genetic_Algorithm<real> trainer( lbounds , ubounds , 1e3 , 10 , 0.1 );

    yakl::timer_start("genetic");
    for (int igen = 0; igen < num_generations; igen++) {
      auto ensemble = trainer.get_ensemble();
      compute_cost( ensemble.get_parameters() , params_init ).deep_copy_to(ensemble.get_cost());
      trainer.update_from_ensemble( ensemble );
    }
    yakl::timer_stop("genetic");

    auto ensemble = trainer.get_ensemble();
    auto best_parameters = trainer.get_best_parameters();
    auto constr = constraint_cost( best_parameters.reshape(num_parameters,1)               );
    auto close  = closeness_cost ( best_parameters.reshape(num_parameters,1) , params_init );
    std::cout << "Max Constraint Violation: " << yakl::intrinsics::maxval( constr ) << std::endl;
    std::cout << "Closeness Rel L1: "         << yakl::intrinsics::sum(close)/close.size() << std::endl;
    std::cout << best_parameters;
  }
  yakl::finalize();
  MPI_Finalize();
}


