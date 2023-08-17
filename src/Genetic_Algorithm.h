
#pragma once

#include <algorithm>
#include <map>
#include <random>

namespace portopt {

  template <class real = float>
  class Genetic_Algorithm {
    public:
    typedef typename yakl::Array<bool      ,1,yakl::memDevice> bool1d;
    typedef typename yakl::Array<bool      ,1,yakl::memHost  > boolHost1d;
    typedef typename yakl::Array<int       ,1,yakl::memDevice> int1d;
    typedef typename yakl::Array<int       ,2,yakl::memHost  > intHost2d;
    typedef typename yakl::Array<int       ,3,yakl::memHost  > intHost3d;
    typedef typename yakl::Array<int  const,1,yakl::memDevice> intConst1d;
    typedef typename yakl::Array<real      ,1,yakl::memDevice> real1d;
    typedef typename yakl::Array<real      ,1,yakl::memHost  > realHost1d;
    typedef typename yakl::Array<real const,1,yakl::memDevice> realConst1d;
    typedef typename yakl::Array<real      ,2,yakl::memDevice> real2d;
    typedef typename yakl::Array<real      ,2,yakl::memHost  > realHost2d;
    typedef typename yakl::Array<real      ,3,yakl::memHost  > realHost3d;

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

    Genetic_Algorithm()  = default;
    ~Genetic_Algorithm() = default;
    Genetic_Algorithm            (Genetic_Algorithm const & rhs) { copy(rhs);               };
    Genetic_Algorithm            (Genetic_Algorithm const &&rhs) { copy(rhs);               };
    Genetic_Algorithm & operator=(Genetic_Algorithm const & rhs) { copy(rhs); return *this; };
    Genetic_Algorithm & operator=(Genetic_Algorithm const &&rhs) { copy(rhs); return *this; };


    Genetic_Algorithm( real1d lbounds                  ,
                       real1d ubounds                  ,
                       real   population_size    = 100 ,
                       int    tournament_size    = 3   ,
                       real   body_snatch_rate   = 0.1 ) {
      std::random_device rd;
      int num_parameters = lbounds.size();
      if (ubounds.size() != num_parameters) yakl::yakl_throw("ERROR: lbounds.size() != ubounds.size()");
      this->parameters         = real2d("ga_parameters",num_parameters,population_size);
      this->lbounds            = lbounds;
      this->ubounds            = ubounds;
      this->lbounds_host       = lbounds.createHostCopy();
      this->ubounds_host       = ubounds.createHostCopy();
      this->best_parameters    = real1d("best_parameters",num_parameters);
      this->best_cost          = std::numeric_limits<real>::max();
      this->body_snatch_rate   = body_snatch_rate;
      this->tournament_size    = tournament_size;
      this->generation         = 0;
      this->rand_seed          = rd();
      std::mt19937_64                   gen(rand_seed);
      std::uniform_real_distribution<>  uniform(0.,1.);
      realHost2d parameters_host("ga_parameters",num_parameters,population_size);
      for (int iparam = 0; iparam < num_parameters; iparam++) {
        real lb = lbounds_host(iparam);
        real ub = ubounds_host(iparam);
        for (int iens = 0; iens < population_size; iens++) { parameters_host(iparam,iens) = uniform(gen)*(ub-lb)+lb; }
      }
      parameters_host.deep_copy_to(parameters);
      rand_seed += num_parameters*population_size;
    }


    bool   is_initialized      () const { return parameters.initialized(); }
    int    get_num_parameters  () const { return parameters.extent(0)    ; }
    int    get_num_ensembles   () const { return parameters.extent(1)    ; }
    int    get_population_size () const { return parameters.extent(1)    ; }
    int    get_generation      () const { return generation              ; }
    real2d get_parameters      () const { return parameters              ; }
    real1d get_best_parameters () const { return best_parameters         ; }
    real   get_best_cost       () const { return best_cost               ; }
    real1d get_lbounds         () const { return lbounds                 ; }
    real1d get_ubounds         () const { return ubounds                 ; }
    real   get_body_snatch_rate() const { return body_snatch_rate        ; }
    int    get_tournament_size () const { return tournament_size         ; }


    void set_tournament_size   (int  n) { this->tournament_size    = n; }
    void set_body_snatch_rate  (real r) { this->body_snatch_rate   = r; }
    

    Ensemble get_ensemble() const {
      return Ensemble( this->parameters.createDeviceCopy() , real1d("ensemble_cost",get_population_size()) );
    }


    void update_from_ensemble( Ensemble const &ensemble , MPI_Comm comm = MPI_COMM_WORLD ) {
      int num_reproducing = (static_cast<int>((1-get_body_snatch_rate())*get_population_size())/2)*2;
      store_best_cost_and_parameters(ensemble);
      // Choose parents through random cage matches in the population
      auto parent_indices = cage_match(ensemble,num_reproducing);
      // Each pair of parents make a pair of children, who subsequently kill and replace their parents
      // Then, mutate a random portion of the children as recompense for their moral failures
      not_safe_for_work(parent_indices,num_reproducing);
      // Increment the generation counter
      generation++;
    }


    void store_best_cost_and_parameters(Ensemble const &ensemble) {
      using yakl::c::parallel_for;
      auto num_parameters      = get_num_parameters();
      auto population_size     = get_population_size();
      auto ensemble_cost       = ensemble.get_cost();
      auto ensemble_parameters = ensemble.get_parameters();
      real min_ensemble_cost = yakl::intrinsics::minval( ensemble_cost  );
      if (min_ensemble_cost < best_cost) {
        best_cost = min_ensemble_cost;
        yakl::ScalarLiveOut<int> best_ind(-1);
        parallel_for( YAKL_AUTO_LABEL() , population_size , YAKL_LAMBDA (int iens) {
          if (ensemble_cost(iens) == min_ensemble_cost) best_ind = iens;
        });
        parallel_for( YAKL_AUTO_LABEL() , num_parameters , YAKL_LAMBDA (int iparam) {
          best_parameters(iparam) = ensemble_parameters(iparam,best_ind());
        });
      }
    }


    // Select population_size parents to create children. For each parent selection, select a random
    // group of tournament_size participants to have a cage match. The victor (with the lowest cost)
    // is chosen to be a parent and reproduce.
    int1d cage_match(Ensemble const &ensemble, int num_reproducing) {
      using yakl::c::parallel_for;
      auto cost = ensemble.get_cost();
      std::mt19937_64                   gen(rand_seed);
      std::uniform_int_distribution<>   uniform(0,get_population_size()-1);
      intHost2d random_pool_host("random_pool_host",num_reproducing,tournament_size);
      for (int iens = 0; iens < num_reproducing; iens++) {
        for (int ienemy = 0; ienemy < tournament_size; ienemy++) {
          random_pool_host(iens,ienemy) = uniform(gen);
        }
      }
      rand_seed += num_reproducing*tournament_size;
      auto random_pool = random_pool_host.createDeviceObject();
      random_pool_host.deep_copy_to(random_pool);
      int1d parent_indices("parent_indices",num_reproducing);
      parallel_for( YAKL_AUTO_LABEL() , num_reproducing , YAKL_LAMBDA (int iens) {
        real best_cost_loc = std::numeric_limits<real>::max();
        for (int ienemy = 0; ienemy < tournament_size; ienemy++) {
          int irand = random_pool(iens,ienemy);
          if (cost(irand) < best_cost_loc) { parent_indices(iens) = irand;    best_cost_loc = cost(irand); }
        }
      });
      return parent_indices;
    }


    // Loop through parents (victors of random cage matches) two at a time, and create a pair of children
    // to replace the pair of parents by using a random convex combination of the parents' parameters
    void not_safe_for_work(intConst1d parent_indices, int num_reproducing) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      auto num_parameters  = get_num_parameters ();
      auto population_size = get_population_size();
      std::mt19937_64                   gen(rand_seed);
      std::uniform_real_distribution<>  uniform(0,1);
      realHost2d random_pool_host ("random_pool_host" ,num_parameters,num_reproducing);
      realHost2d random_pool2_host("random_pool_host2",num_parameters,population_size-num_reproducing);
      for (int iparam = 0; iparam < num_parameters; iparam++) {
        for (int iens = 0; iens < num_reproducing; iens++) { random_pool_host(iparam,iens) = uniform(gen); }
        real lb = lbounds_host(iparam);
        real ub = ubounds_host(iparam);
        for (int iens = num_reproducing; iens < population_size; iens++) {
          random_pool2_host(iparam,iens-num_reproducing) = uniform(gen)*(ub-lb)+lb;
        }
      }
      rand_seed += num_parameters*population_size*2;
      auto random_pool  = random_pool_host .createDeviceObject();
      auto random_pool2 = random_pool2_host.createDeviceObject();
      random_pool_host .deep_copy_to(random_pool );
      if (population_size > num_reproducing) random_pool2_host.deep_copy_to(random_pool2);
      real2d parameters_new("ga_parameters",num_parameters,population_size);
      parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(num_parameters,{0,population_size-1,2}) ,
                                        YAKL_LAMBDA (int iparam, int iens) {
        if (iens < num_reproducing) {
          real gene1 = parameters(iparam,parent_indices(iens  ));
          real gene2 = parameters(iparam,parent_indices(iens+1));
          real rn1 = random_pool(iparam,iens  );
          real rn2 = random_pool(iparam,iens+1);
          parameters_new(iparam,iens  ) = rn1 * gene1 + (1-rn1) * gene2;
          parameters_new(iparam,iens+1) = rn2 * gene1 + (1-rn2) * gene2;
        } else {
          parameters_new(iparam,iens  ) = random_pool2(iparam,iens  -num_reproducing);
          parameters_new(iparam,iens+1) = random_pool2(iparam,iens+1-num_reproducing);
        }
      });
      parameters = parameters_new;
    }

    protected:

    real2d     parameters        ;
    real1d     lbounds           ;
    real1d     ubounds           ;
    realHost1d lbounds_host      ;
    realHost1d ubounds_host      ;
    real1d     best_parameters   ;
    real       best_cost         ;
    real       body_snatch_rate  ;
    int        tournament_size   ;
    int        generation        ;
    size_t     rand_seed         ;

    void copy(Genetic_Algorithm const &rhs) {
      this->parameters         = rhs.parameters        ;
      this->lbounds            = rhs.lbounds           ;
      this->ubounds            = rhs.ubounds           ;
      this->lbounds_host       = rhs.lbounds_host      ;
      this->ubounds_host       = rhs.ubounds_host      ;
      this->best_parameters    = rhs.best_parameters   ;
      this->best_cost          = rhs.best_cost         ;
      this->body_snatch_rate   = rhs.body_snatch_rate  ;
      this->tournament_size    = rhs.tournament_size   ;
      this->generation         = rhs.generation        ;
      this->rand_seed          = rhs.rand_seed         ;
    }
  };

}


