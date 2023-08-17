
#include "YAKL.h"
#include "Gradient_Descent_Adam_FD.h"
#include "Genetic_Algorithm.h"

int main( int argc , char **argv ) {
  MPI_Init(&argc,&argv);
  yakl::init();

  {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    typedef double real;
    typedef yakl::Array<real      ,1,yakl::memHost> real1d;
    typedef yakl::Array<real const,2,yakl::memHost> realConst2d;

    real constexpr constr_tol = 1.e-8;

    int num_parameters = 6;
    real1d parameters_init("parameters_init",num_parameters);
    parameters_init = 0.5;

    auto constraint_cost = [=] (realConst2d params) -> real1d {
      real1d cost("cost",params.extent(1));
      for (int iens = 0; iens < params.extent(1); iens++) {
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
      }
      return cost;
    };

    auto closeness_cost = [=] (realConst2d params) -> real1d {
      real1d cost("cost",params.extent(1));
      for (int iens = 0; iens < params.extent(1); iens++) {
        real num = 0;
        real den = 0;
        for (int iparam = 0; iparam < num_parameters; iparam++) {
          num += std::abs( params(iparam,iens) - parameters_init(iparam) );
          den += std::abs( parameters_init(iparam) );
        }
        cost(iens) = num / std::max(1.e-20,den);
      }
      return cost;
    };

    auto compute_cost = [=] (realConst2d params) -> real1d {
      auto constr = constraint_cost(params);
      auto close  = closeness_cost (params);
      for (int iens = 0; iens < params.extent(1); iens++) {
        constr(iens) = std::max(0.,constr(iens)-constr_tol)*1.e10 + close(iens);
      }
      return constr;
    };

    int num_epochs = 1e5;
    portopt::Gradient_Descent_Adam_FD<real,yakl::memHost> trainer( parameters_init.createHostCopy() );

    yakl::timer_start("training_time");
    for (int pwr = -3 ; pwr >= -9; pwr--) {
      trainer = portopt::Gradient_Descent_Adam_FD<real,yakl::memHost>( trainer.get_parameters() , std::pow(10.,(double)pwr) );
      for (int iepoch = 0; iepoch < num_epochs; iepoch++) {
        auto ensemble = trainer.get_ensemble();
        compute_cost( ensemble.get_parameters() ).deep_copy_to(ensemble.get_cost());
        trainer.update_from_ensemble( ensemble );
        trainer.increment_epoch();
      }
    }
    yakl::timer_stop("training_time");

    auto ensemble = trainer.get_ensemble();
    auto constr = constraint_cost( ensemble.get_parameters() );
    auto close  = closeness_cost ( ensemble.get_parameters() );
    std::cout << "Max Constraint Violation: " << yakl::intrinsics::maxval( constr ) << std::endl;
    std::cout << "Closeness Rel L1: "         << yakl::intrinsics::sum(close)/close.size() << std::endl;
    std::cout << "Params: " << trainer.get_parameters();

    if (! trainer.parameters_identical_across_tasks()) yakl::yakl_throw("ERROR: parameters are not the same");
  }


  yakl::finalize();
  MPI_Finalize();
}


