
#pragma once

#include <algorithm>
#include <map>
#include <random>

namespace portopt {

  template <class real = float>
  class Random_Walk {
    public:
    typedef typename yakl::Array<real      ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real      ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real      ,2,yakl::memHost  > realHost2d;

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

    Random_Walk()  = default;
    ~Random_Walk() = default;
    Random_Walk            (Random_Walk const & rhs) { copy(rhs);               };
    Random_Walk            (Random_Walk const &&rhs) { copy(rhs);               };
    Random_Walk & operator=(Random_Walk const & rhs) { copy(rhs); return *this; };
    Random_Walk & operator=(Random_Walk const &&rhs) { copy(rhs); return *this; };


    Random_Walk( real1d initial_parameters        ,
                 int    num_guesses        = 100  ,
                 real   step_size          = 0.01 ) {
      std::random_device rd;
      this->parameters   = initial_parameters;
      this->num_guesses  = num_guesses;
      this->step_size    = step_size;
      this->rand_seed    = rd();
    }


    bool   is_initialized    () const { return parameters.initialized(); }
    int    get_num_parameters() const { return parameters.size()       ; }
    real1d get_parameters    () const { return parameters              ; }
    real   get_step_size     () const { return step_size               ; }
    int    get_num_guesses   () const { return num_guesses             ; }


    void set_step_size  (real step_size  ) { this->step_size   = step_size  ; }
    void set_num_guesses(int  num_guesses) { this->num_guesses = num_guesses; }
    

    Ensemble get_ensemble() const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int num_parameters = get_num_parameters();
      std::mt19937_64                   gen(rand_seed);
      std::uniform_real_distribution<>  uniform(-1.,1.);
      realHost2d random_pool_host("random_pool_host",num_parameters,num_guesses);
      for (int iparam = 0; iparam < num_parameters; iparam++) {
        for (int iens = 0; iens < num_guesses; iens++) {
          random_pool_host(iparam,iens) = uniform(gen);
        }
      }
      auto random_pool = random_pool_host.createDeviceObject();
      random_pool_host.deep_copy_to(random_pool);
      real2d ensemble_parameters("ensemble_parameters",num_parameters,num_guesses);
      YAKL_SCOPE( parameters  , this->parameters  );
      YAKL_SCOPE( num_guesses , this->num_guesses );
      YAKL_SCOPE( step_size   , this->step_size   );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_parameters,num_guesses) ,
                                        YAKL_LAMBDA (int iparam, int iens) {
        if (iens < num_guesses-1) {
          ensemble_parameters(iparam,iens) = parameters(iparam) + random_pool(iparam,iens)*step_size;
        } else {
          ensemble_parameters(iparam,iens) = parameters(iparam);
        }
      });
      return Ensemble( ensemble_parameters , real1d("ensemble_cost",num_guesses) );
    }


    void update_from_ensemble( Ensemble const &ensemble , MPI_Comm comm = MPI_COMM_WORLD ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int min_ind = yakl::intrinsics::minloc(ensemble.get_cost());
      auto ensemble_params = ensemble.get_parameters();
      YAKL_SCOPE( parameters  , this->parameters  );
      parallel_for( YAKL_AUTO_LABEL() , get_num_parameters() , YAKL_LAMBDA (int iparam) {
        parameters(iparam) = ensemble_params(iparam,min_ind);
      });
    }

    protected:

    real1d  parameters ;
    int     num_guesses;
    real    step_size  ;
    size_t  rand_seed  ;


    void copy(Random_Walk const &rhs) {
      this->parameters  = rhs.parameters ;
      this->num_guesses = rhs.num_guesses;
      this->step_size   = rhs.step_size  ;
      this->rand_seed   = rhs.rand_seed  ;
    }
  };

}


