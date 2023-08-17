
#pragma once

#include <algorithm>
#include <map>

namespace portopt {

  template <class real = float, int Mem = yakl::memDevice>
  class Gradient_Descent_Adam_FD {
    public:
    typedef typename yakl::Array<real,1,Mem> real1d;
    typedef typename yakl::Array<real,2,Mem> real2d;
    real static constexpr tiny = std::numeric_limits<real>::min()*10;
    real static constexpr eps  = std::numeric_limits<real>::epsilon();

    class Ensemble {
      public:
      Ensemble () = default;
      ~Ensemble() = default;
      Ensemble            (Ensemble const & rhs) { copy(rhs);               };
      Ensemble            (Ensemble const &&rhs) { copy(rhs);               };
      Ensemble & operator=(Ensemble const & rhs) { copy(rhs); return *this; };
      Ensemble & operator=(Ensemble const &&rhs) { copy(rhs); return *this; };

      Ensemble(real2d const &a, real1d const &b) { parameters = a;  cost = b; }

      real2d get_parameters    () const { return parameters; }
      real1d get_cost          () const { return cost; }
      int    get_num_parameters() const { return parameters.extent(0); }
      int    get_ensemble_size () const { return parameters.extent(1); }

      protected:
      real2d parameters; // Parameters for each particle, dimensioned as num_parameters,ensemble_size
      real1d cost;       // Cost for each particl ein this ensemble
      void copy(Ensemble const &rhs) {
        parameters = rhs.parameters;
        cost       = rhs.cost;
      }
    };

    Gradient_Descent_Adam_FD()  = default;
    ~Gradient_Descent_Adam_FD() = default;
    Gradient_Descent_Adam_FD            (Gradient_Descent_Adam_FD const & rhs) { copy(rhs);               };
    Gradient_Descent_Adam_FD            (Gradient_Descent_Adam_FD const &&rhs) { copy(rhs);               };
    Gradient_Descent_Adam_FD & operator=(Gradient_Descent_Adam_FD const & rhs) { copy(rhs); return *this; };
    Gradient_Descent_Adam_FD & operator=(Gradient_Descent_Adam_FD const &&rhs) { copy(rhs); return *this; };


    Gradient_Descent_Adam_FD( real1d parameters  ,
                              real alpha = static_cast<real>(0.001) ,
                              real beta1 = static_cast<real>(0.9  ) ,
                              real beta2 = static_cast<real>(0.999) ) {
      int num_parameters = parameters.extent(0);
      this->num_updates = 0;
      this->alpha       = alpha;
      this->beta1       = beta1;
      this->beta2       = beta2;
      this->beta1_pow   = beta1;
      this->beta2_pow   = beta2;
      this->m           = real1d("m",num_parameters);
      this->v           = real1d("v",num_parameters);
      this->m           = 0;
      this->v           = 0;
      this->parameters  = parameters;
    }



    bool   is_initialized    () const { return parameters.initialized(); }
    int    get_num_parameters() const { return parameters.extent(0)    ; }
    int    get_num_ensembles () const { return parameters.extent(0)+1  ; }
    real1d get_parameters    () const { return parameters              ; }
    size_t get_num_updates   () const { return num_updates             ; }
    real   get_alpha         () const { return alpha                   ; }
    real   get_beta1         () const { return beta1                   ; }
    real   get_beta2         () const { return beta2                   ; }
    


    Ensemble get_ensemble() {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      YAKL_SCOPE( parameters , this->parameters );
      auto num_parameters = get_num_parameters();
      real delta = std::sqrt(eps);
      // real delta = 10*eps;
      int ensemble_size = num_parameters+1;
      real2d ensemble_parameters("ensemble_parameters",num_parameters,ensemble_size);
      real1d ensemble_cost      ("ensemble_cost"                     ,ensemble_size);
      if constexpr (Mem == yakl::memDevice) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,ensemble_size) ,
                                          YAKL_LAMBDA (int iparam, int iens) {
          ensemble_parameters(iparam,iens) = parameters(iparam);
          if (iparam == iens) ensemble_parameters(iparam,iens) += std::max( delta , delta * std::abs(ensemble_parameters(iparam,iens)) );
          if (iparam == 0) ensemble_cost(iens) = 0;
        });
      } else {
        for (int iparam = 0; iparam < num_parameters; iparam++) {
          for (int iens = 0; iens < ensemble_size; iens++) {
            ensemble_parameters(iparam,iens) = parameters(iparam);
            if (iparam == iens) ensemble_parameters(iparam,iens) += std::max( delta , delta * std::abs(ensemble_parameters(iparam,iens)) );
            if (iparam == 0) ensemble_cost(iens) = 0;
          }
        }
      }
      return Ensemble( ensemble_parameters , ensemble_cost );
    }



    void update_from_ensemble( Ensemble const &ensemble , MPI_Comm comm = MPI_COMM_WORLD ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_parameters      = get_num_parameters();
      auto ensemble_costs     = ensemble.get_cost();
      auto ensemble_parameters = ensemble.get_parameters();

      YAKL_SCOPE( parameters , this->parameters );
      YAKL_SCOPE( alpha      , this->alpha      );
      YAKL_SCOPE( beta1      , this->beta1      );
      YAKL_SCOPE( beta2      , this->beta2      );
      YAKL_SCOPE( beta1_pow  , this->beta1_pow  );
      YAKL_SCOPE( beta2_pow  , this->beta2_pow  );
      YAKL_SCOPE( m          , this->m          );
      YAKL_SCOPE( v          , this->v          );

      if constexpr (Mem == yakl::memDevice) {
        parallel_for( YAKL_AUTO_LABEL() , num_parameters , YAKL_LAMBDA (int iparam) {
          real g = ( ensemble_costs    (       iparam) - ensemble_costs    (       num_parameters) ) /
                   ( ensemble_parameters(iparam,iparam) - ensemble_parameters(iparam,num_parameters) );
          m(iparam) = beta1 * m(iparam) + (1-beta1) * g;
          v(iparam) = beta2 * v(iparam) + (1-beta2) * g*g;
          real mhat = m(iparam)/(1-beta1_pow);
          real vhat = v(iparam)/(1-beta2_pow);
          parameters(iparam) -= alpha * mhat / std::max( std::sqrt(vhat) , static_cast<real>(1.e-50) );
        });
      } else {
        for (int iparam = 0; iparam < num_parameters; iparam++) {
          real g = ( ensemble_costs    (       iparam) - ensemble_costs    (       num_parameters) ) /
                   ( ensemble_parameters(iparam,iparam) - ensemble_parameters(iparam,num_parameters) );
          // std::cout << "gradient: " << std::scientific << std::setprecision(16) << g << std::endl;
          m(iparam) = beta1 * m(iparam) + (1-beta1) * g;
          v(iparam) = beta2 * v(iparam) + (1-beta2) * g*g;
          real mhat = m(iparam)/(1-beta1_pow);
          real vhat = v(iparam)/(1-beta2_pow);
          parameters(iparam) -= alpha * mhat / std::max( std::sqrt(vhat) , static_cast<real>(1.e-50) );
        }
      }
      num_updates++;
    }



    void increment_epoch() { beta1_pow *= beta1;    beta2_pow *= beta2; }



    bool parameters_identical_across_tasks( MPI_Comm comm = MPI_COMM_WORLD ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      YAKL_SCOPE( parameters , this->parameters );
      auto parameters_host     = parameters.createHostCopy();
      auto parameters_min_host = parameters.createHostObject();
      auto parameters_max_host = parameters.createHostObject();
      MPI_Datatype data_type;
      if constexpr (std::is_same<real,float >::value) { data_type = MPI_FLOAT;  }
      if constexpr (std::is_same<real,double>::value) { data_type = MPI_DOUBLE; }
      MPI_Allreduce( parameters_host.data() , parameters_min_host.data() , parameters.size() , data_type , MPI_MIN , comm );
      MPI_Allreduce( parameters_host.data() , parameters_max_host.data() , parameters.size() , data_type , MPI_MAX , comm );
      if constexpr (Mem == yakl::memDevice) {
        auto parameters_min = parameters_min_host.createDeviceCopy();
        auto parameters_max = parameters_max_host.createDeviceCopy();
        yakl::ScalarLiveOut<bool> is_same(true);
        parallel_for( YAKL_AUTO_LABEL() , get_num_parameters() , YAKL_LAMBDA (int iparam) {
          if ( parameters(iparam) != parameters_min(iparam) ||
               parameters(iparam) != parameters_max(iparam)  ) is_same = false;
        });
        return is_same.hostRead();
      } else {
        for (int iparam = 0; iparam < get_num_parameters(); iparam++) {
          if ( parameters(iparam) != parameters_min_host(iparam) ||
               parameters(iparam) != parameters_max_host(iparam)  ) return false;
        }
        return true;
      }
    }



    protected:

    real1d parameters ; // parameters being trained (num_parameters)
    size_t num_updates; // The number of model updates performes thus far
    real   alpha      ;
    real   beta1      ;
    real   beta2      ;
    real   beta1_pow  ;
    real   beta2_pow  ;
    real1d m          ;
    real1d v          ;

    void copy(Gradient_Descent_Adam_FD const &rhs) {
      parameters  = rhs.parameters ;
      num_updates = rhs.num_updates;
      alpha       = rhs.alpha      ;
      beta1       = rhs.beta1      ;
      beta2       = rhs.beta2      ;
      beta1_pow   = rhs.beta1_pow  ;
      beta2_pow   = rhs.beta2_pow  ;
      m           = rhs.m          ;
      v           = rhs.v          ;
    }
  };

}


