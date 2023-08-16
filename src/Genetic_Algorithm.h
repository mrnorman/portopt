
#pragma once

#include <algorithm>
#include <map>
#include <random>

namespace portopt {

  template <class real = float, int Mem = yakl::memDevice>
  class Genetic_Algorithm {
    public:
    typedef typename yakl::Array<bool      ,1,Mem> bool1d;
    typedef typename yakl::Array<int       ,1,Mem> int1d;
    typedef typename yakl::Array<int  const,1,Mem> intConst1d;
    typedef typename yakl::Array<real      ,1,Mem> real1d;
    typedef typename yakl::Array<real const,1,Mem> realConst1d;
    typedef typename yakl::Array<real      ,2,Mem> real2d;

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


    Genetic_Algorithm( real1d lbounds                   ,
                       real1d ubounds                   ,
                       real   population_size    = 100  ,
                       int    tournament_size    = 3    ,
                       real   mutation_rate      = 0.1  ,
                       real   mutation_magnitude = 0.5  ) {
      int num_parameters = lbounds.size();
      if (ubounds.size() != num_parameters) yakl::yakl_throw("ERROR: lbounds.size() != ubounds.size()");
      this->parameters         = real2d("ga_parameters",num_parameters,population_size);
      this->lbounds            = lbounds;
      this->ubounds            = ubounds;
      this->best_parameters    = real1d("best_parameters",num_parameters);
      this->best_cost          = std::numeric_limits<real>::max();
      this->mutation_rate      = mutation_rate;
      this->mutation_magnitude = mutation_magnitude;
      this->tournament_size    = tournament_size;
      this->generation         = 0;
      std::random_device                rd;
      std::mt19937_64                   gen(rd());
      std::uniform_real_distribution<>  uniform(0.,1.);
      // TODO: Make host Array, and copy to device Array afterward
      for (int iparam = 0; iparam < num_parameters; iparam++) {
        real lb = lbounds(iparam);
        real ub = ubounds(iparam);
        for (int iens = 0; iens < population_size; iens++) {
          parameters(iparam,iens) = uniform(gen) * (ub-lb) + lb;
        }
      }
    }



    bool   is_initialized        () const { return parameters.initialized(); }
    int    get_num_parameters    () const { return parameters.extent(0)    ; }
    int    get_num_ensembles     () const { return parameters.extent(1)    ; }
    int    get_population_size   () const { return parameters.extent(1)    ; }
    int    get_generation        () const { return generation              ; }
    real2d get_parameters        () const { return parameters              ; }
    real1d get_best_parameters   () const { return best_parameters         ; }
    real   get_best_cost         () const { return best_cost               ; }
    real1d get_lbounds           () const { return lbounds                 ; }
    real1d get_ubounds           () const { return ubounds                 ; }
    real   get_mutation_rate     () const { return mutation_rate           ; }
    real   get_mutation_magnitude() const { return mutation_magnitude      ; }
    int    get_tournament_size   () const { return tournament_size         ; }



    void set_tournament_size   (int  n) { this->tournament_size    = n; }
    void set_mutation_rate     (real r) { this->mutation_rate      = r; }
    void set_mutation_magnitude(real r) { this->mutation_magnitude = r; }
    


    Ensemble get_ensemble() const {
      if constexpr (Mem == yakl::memDevice) {
        return Ensemble( this->parameters.createDeviceCopy() , real1d("ensemble_cost",get_population_size()) );
      } else {
        return Ensemble( this->parameters.createHostCopy  () , real1d("ensemble_cost",get_population_size()) );
      }
    }



    void update_from_ensemble( Ensemble const &ensemble , MPI_Comm comm = MPI_COMM_WORLD ) {
      // Get the best parameters for this ensemble
      store_best_cost_and_parameters(ensemble);
      // Choose parents through random cage matches in the population
      auto parent_indices = cage_match(ensemble);
      // Each pair of parents make a pair of children, who subsequently kill and replace their paratens
      not_safe_for_work(parent_indices);
      // Pour radioactive ooze over a random portion of the population
      secret_of_the_ooze();
      // Increment the generation counter
      generation++;
    }



    void store_best_cost_and_parameters(Ensemble const &ensemble) {
      auto num_parameters      = get_num_parameters();
      auto population_size     = get_population_size();
      auto ensemble_cost       = ensemble.get_cost();
      auto ensemble_parameters = ensemble.get_parameters();
      real min_ensemble_cost = yakl::intrinsics::minval( ensemble_cost  );
      if (min_ensemble_cost < best_cost) {
        best_cost = min_ensemble_cost;
        for (int iens = 0; iens < population_size; iens++) {
          if (ensemble_cost(iens) == min_ensemble_cost) {
            for (int iparam = 0; iparam < num_parameters; iparam++) {
              best_parameters(iparam) = ensemble_parameters(iparam,iens);
            }
            break;
          }
        }
      }
    }



    // Select population_size parents to create children. For each parent selection, select a random
    // group of tournament_size participants to have a cage match. The victor (with the lowest cost)
    // is chosen to be a parent and reproduce.
    int1d cage_match(Ensemble const &ensemble) const {
      auto population_size = get_population_size();
      auto cost            = ensemble.get_cost();
      int1d parent_indices("parent_indices",population_size);
      std::random_device                rd;
      std::mt19937_64                   gen(rd());
      std::uniform_int_distribution<>   uniform(0,population_size-1);
      for (int iens = 0; iens < population_size; iens++) {
        real best_cost_loc = std::numeric_limits<real>::max();
        for (int ienemy = 0; ienemy < tournament_size; ienemy++) {
          int irand = uniform(gen);
          if (cost(irand) < best_cost_loc) { parent_indices(iens) = irand;    best_cost_loc = cost(irand); }
        }
      }
      return parent_indices;
    }



    // Loop through parents (victors of random cage matches) two at a time, and create a pair of children
    // to replace the pair of parents by using a random convex combination of the parents' parameters
    void not_safe_for_work(intConst1d parent_indices) {
      auto num_parameters  = get_num_parameters ();
      auto population_size = get_population_size();
      std::random_device                rd;
      std::mt19937_64                   gen(rd());
      std::uniform_real_distribution<>  uniform(0,1);
      real2d parameters_new("ga_parameters",num_parameters,population_size);
      for (int iparam = 0; iparam < num_parameters; iparam++) {
        for (int iens = 0; iens < population_size; iens += 2) {
          real gene1 = parameters(iparam,parent_indices(iens  ));
          real gene2 = parameters(iparam,parent_indices(iens+1));
          real rn = uniform(gen);
          parameters_new(iparam,iens  ) = rn * gene1 + (1-rn) * gene2;
          rn = uniform(gen);
          parameters_new(iparam,iens+1) = rn * gene1 + (1-rn) * gene2;
        }
      }
      parameters = parameters_new;
    }



    // Pass through the population, and mercilessly cover a mutation_rate proportion of them with radioactive ooze.
    // This mutates their parameters according to a random uniform number determined by mutation_magnitude in magnitude
    void secret_of_the_ooze() {
      auto num_parameters  = get_num_parameters ();
      auto population_size = get_population_size();
      std::random_device                rd;
      std::mt19937_64                   gen(rd());
      std::uniform_real_distribution<>  uniform(0,1);

      bool1d mutate("mutate",population_size);
      for (int iens = 0; iens < population_size; iens++) {
        mutate(iens) = uniform(gen) < mutation_rate ? true : false;
      }

      uniform = std::uniform_real_distribution<>(-1,1);
      for (int iparam = 0; iparam < num_parameters; iparam++) {
        for (int iens = 0; iens < population_size; iens++) {
          if (mutate(iens)) {
            // Mutate the parameter
            parameters(iparam,iens) += uniform(gen)*mutation_magnitude*(ubounds(iparam)-lbounds(iparam));
            // Keep the mutated parameters in bounds
            parameters(iparam,iens) = std::max(lbounds(iparam),std::min(ubounds(iparam),parameters(iparam,iens)));
          }
        }
      }
    }

    protected:

    real2d parameters        ;
    real1d lbounds           ;
    real1d ubounds           ;
    real1d best_parameters   ;
    real   best_cost         ;
    real   mutation_rate     ;
    real   mutation_magnitude;
    int    tournament_size   ;
    int    generation        ;

    void copy(Genetic_Algorithm const &rhs) {
      this->parameters         = rhs.parameters        ;
      this->lbounds            = rhs.lbounds           ;
      this->ubounds            = rhs.ubounds           ;
      this->best_parameters    = rhs.best_parameters   ;
      this->best_cost          = rhs.best_cost         ;
      this->mutation_rate      = rhs.mutation_rate     ;
      this->mutation_magnitude = rhs.mutation_magnitude;
      this->tournament_size    = rhs.tournament_size   ;
      this->generation         = rhs.generation        ;
    }
  };

}


