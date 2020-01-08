/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_statevector_controller_hpp_
#define _aer_statevector_controller_hpp_

#include "base/controller.hpp"
#include "statevector_state.hpp"
#include "framework/json.hpp"
#include <chrono>
#include <ctime>    
#include <sstream>
#include <boost/lexical_cast.hpp>


#include <limits>
#include "dtoa_milo.h"

extern "C" {
  #include "Simulator/sim.h"
}

#include <Python.h>
//#include <stdlib.h>
#include <cmath>

namespace AER {
namespace Simulator {

enum class Gates {
  id, h, s, sdg, t, tdg, // single qubit
  // multi-qubit controlled (including single-qubit non-controlled)
  mcx, mcy, mcz, mcu1, mcu2, mcu3, mcswap
};
//=========================================================================
// StatevectorController class
//=========================================================================

/**************************************************************************
 * Config settings:
 * 
 * From Statevector::State class
 * 
 * - "initial_statevector" (json complex vector): Use a custom initial
 *      statevector for the simulation [Default: null].
 * - "zero_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-10]
 * - "statevector_parallel_threshold" (int): Threshold that number of qubits
 *      must be greater than to enable OpenMP parallelization at State
 *      level [Default: 13]
 * - "statevector_sample_measure_opt" (int): Threshold that number of qubits
 *      must be greater than to enable indexing optimization during
 *      measure sampling [Default: 10]
 * - "statevector_hpc_gate_opt" (bool): Enable large qubit gate optimizations.
 *      [Default: False]
 * 
 * From BaseController Class
 *
 * - "max_parallel_threads" (int): Set the maximum OpenMP threads that may
 *      be used across all levels of parallelization. Set to 0 for maximum
 *      available. [Default : 0]
 * - "max_parallel_experiments" (int): Set number of circuits that may be
 *      executed in parallel. Set to 0 to use the number of max parallel
 *      threads [Default: 1]
 * - "counts" (bool): Return counts object in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 * 
 **************************************************************************/

const stringmap_t<Gates> ma_gateset_({
  // Single qubit gates
  {"id", Gates::id},     // Pauli-Identity gate
  {"x", Gates::mcx},     // Pauli-X gate
  {"y", Gates::mcy},     // Pauli-Y gate
  {"z", Gates::mcz},     // Pauli-Z gate
  {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
  {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
  {"t", Gates::t},       // T-gate (sqrt(S))
  {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
  // Waltz Gates
  {"u1", Gates::mcu1},   // zero-X90 pulse waltz gate
  {"u2", Gates::mcu2},   // single-X90 pulse waltz gate
  {"u3", Gates::mcu3},   // two X90 pulse waltz gate
  // Two-qubit gates
  {"cx", Gates::mcx},        // Controlled-X gate (CNOT)
  {"cy", Gates::mcy},        // Controlled-Y gate
  {"cz", Gates::mcz},        // Controlled-Z gate
  /*{"cu1", Gates::mcu1},      // Controlled-u1 gate
  {"cu2", Gates::mcu2},      // Controlled-u2 gate
  {"cu3", Gates::mcu3},      // Controlled-u3 gate
  */
  {"swap", Gates::mcswap},   // SWAP gate
  // 3-qubit gates
  //{"ccx", Gates::mcx},       // Controlled-CX gate (Toffoli)
  //{"cswap", Gates::mcswap},  // Controlled SWAP gate (Fredkin)
  // Multi-qubit controlled gates
  /*{"mcx", Gates::mcx},      // Multi-controlled-X gate
  {"mcy", Gates::mcy},      // Multi-controlled-Y gate
  {"mcz", Gates::mcz},      // Multi-controlled-Z gate
  {"mcu1", Gates::mcu1},    // Multi-controlled-u1
  {"mcu2", Gates::mcu2},    // Multi-controlled-u2
  {"mcu3", Gates::mcu3},    // Multi-controlled-u3
  {"mcswap", Gates::mcswap} // Multi-controlled SWAP gate
  */
});
class StatevectorController : public Base::Controller {
public:
  //-----------------------------------------------------------------------
  // Base class config override
  //-----------------------------------------------------------------------
  StatevectorController();

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  // Allowed config options:
  // - "initial_statevector: complex_vector"
  // Plus Base Controller config options
  virtual void set_config(const json_t &config) override;

  // Clear the current config
  void virtual clear_config() override;

protected:

  virtual size_t required_memory_mb(const Circuit& circuit,
                                    const Noise::NoiseModel& noise) const override;

private:

  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // This simulator will only return a single shot, regardless of the
  // input shot number

  virtual void ds_apply_ops(ds_Register ds_reg, const Circuit &circ, const std::unordered_set<int> & x_errors, const std::unordered_set<int> & y_errors, const std::unordered_set<int> & z_errors) const;
  virtual void check_and_inject(ds_Register ds_reg, long position, int qubit, const std::unordered_set<int> & x_errors, const std::unordered_set<int> & y_errors, const std::unordered_set<int> & z_errors) const;

  virtual ExperimentData run_circuit(const Circuit &circ,
                                 const Noise::NoiseModel& noise,
                                 const json_t &config,
                                 uint_t shots,
                                 uint_t rng_seed) const override;

  virtual ExperimentData direct_ds_sim(const Circuit &circ,
                                 const Noise::NoiseModel& noise,
                                 const json_t &config,
                                 uint_t shots,
                                 uint_t rng_seed) const;

  virtual ExperimentData inject_n_errors(const Circuit &circ,
                                 const Noise::NoiseModel& noise,
                                 const json_t &config,
                                 uint_t shots,
                                 uint_t rng_seed) const;

  virtual ExperimentData get_hotspots(const Circuit &circ,
                                 const Noise::NoiseModel& noise,
                                 const json_t &config,
                                 uint_t shots,
                                 uint_t rng_seed) const;
  //-----------------------------------------------------------------------
  // Custom initial state
  //-----------------------------------------------------------------------        
  cvector_t initial_state_;
};

//=========================================================================
// Implementations
//=========================================================================

StatevectorController::StatevectorController() : Base::Controller() {
  // Disable qubit truncation by default
  Base::Controller::truncate_qubits_ = false;
}

//-------------------------------------------------------------------------
// Config
//-------------------------------------------------------------------------

void StatevectorController::set_config(const json_t &config) {
  // Set base controller config
  Base::Controller::set_config(config);

  //Add custom initial state
  if (JSON::get_value(initial_state_, "initial_statevector", config)) {
    // Check initial state is normalized
    if (!Utils::is_unit_vector(initial_state_, validation_threshold_))
      throw std::runtime_error("StatevectorController: initial_statevector is not a unit vector");
  }
}

void StatevectorController::clear_config() {
  Base::Controller::clear_config();
  initial_state_ = cvector_t();
}

size_t StatevectorController::required_memory_mb(const Circuit& circ,
                                                 const Noise::NoiseModel& noise) const {
  //Statevector::State<> state;
  //return state.required_memory_mb(circ.num_qubits, circ.ops);
  return 32;
}

//-------------------------------------------------------------------------
// Run circuit
//-------------------------------------------------------------------------

void StatevectorController::check_and_inject(ds_Register ds_reg, long position, int qubit, const std::unordered_set<int> & x_errors, const std::unordered_set<int> & y_errors, const std::unordered_set<int> & z_errors) const{
  if(x_errors.count(position) != 0){
    ds_X_no_error(ds_reg, qubit, 1);
    ds_reg.n_errors[0] += 1;
  }
  if(y_errors.count(position) != 0){   
    ds_XZ_no_error(ds_reg, qubit, 1);
    ds_reg.n_errors[2] += 1;
  }
  if(z_errors.count(position) != 0){   
    ds_Z_no_error(ds_reg, qubit, 1);
    ds_reg.n_errors[1] += 1;
  }
}
void StatevectorController::ds_apply_ops(ds_Register ds_reg, const Circuit &circ,const std::unordered_set<int> & x_errors, const std::unordered_set<int> & y_errors, const std::unordered_set<int> & z_errors) const {
  int gate_time = 1;
  long gate_position = 0;
  std::unordered_set<int> qubit_init_done;
  //std::cout<<"Add code for errors directly after initialization!" << std::endl;

  for(int i = 0; i < circ.num_qubits; i++){
    check_and_inject(ds_reg, i, i, x_errors, y_errors, z_errors);
  }
  gate_position += circ.num_qubits;
  bool do_init_errors = true;
  for (const auto & op: circ.ops) {
    reg_t oqs = op.qubits;
    const size_t Nqs = op.qubits.size();
    switch (op.type) {
      case Operations::OpType::gate:
        auto it = ma_gateset_.find(op.name);
        if (it == ma_gateset_.end())
          throw std::invalid_argument("QubitVectorState::invalid gate instruction \'" + 
                                      op.name + "\'.");
        switch (it -> second) {
	  case Gates::mcx:
	    // Includes X, CX, CCX, etc
	    //BaseState::qreg_.apply_mcx(op.qubits);
            if (Nqs == 1){
              ds_X(ds_reg,oqs[0],gate_time);
            }
            else if(Nqs == 2){
              ds_cnot(ds_reg,oqs[0],oqs[1], gate_time);
            }
            else {
                throw std::invalid_argument("QubitVectorState::invalid gate instruction \'" +
                                            op.name + "\'. X with more than 2 argument qubits");
            }
	    break;            
          case Gates::mcy:
	    // Includes Y, CY, CCY, etc
	    //BaseState::qreg_.apply_mcy(op.qubits);
            if (Nqs == 1){
              ds_XZ(ds_reg,oqs[0],gate_time);
            }
            else {
              throw std::invalid_argument("QubitVectorState::invalid gate instruction \'" +
                                          op.name + "\'. Y with more than 2 argument qubits");
            }
	    break;
	  case Gates::mcz:
	    // Includes Z, CZ, CCZ, etc
	    //BaseState::qreg_.apply_mcphase(op.qubits, -1);
            if (Nqs == 1){
              ds_Z(ds_reg,oqs[0],gate_time);
            }
            else {
                throw std::invalid_argument("QubitVectorState::invalid gate instruction \'" +
                                            op.name + "\'. Z with more than 2 argument qubits");
            }
	    break;
            
	  case Gates::id:
	    break;
	  case Gates::h:
	    //apply_gate_mcu3(op.qubits, M_PI / 2., 0., M_PI);
            if (Nqs == 1){
              ds_Hadamard(ds_reg,oqs[0],gate_time);
            }
            else {
                throw std::invalid_argument("QubitVectorState::invalid gate instruction \'" +
                                            op.name + "\'. H with more than 2 argument qubits");
            }
	    break;
          /*
          case Gates::s:
            apply_gate_phase(op.qubits[0], complex_t(0., 1.));
            break;
          case Gates::sdg:
            apply_gate_phase(op.qubits[0], complex_t(0., -1.));
            break;
          case Gates::t: {
            const double isqrt2{1. / std::sqrt(2)};
            apply_gate_phase(op.qubits[0], complex_t(isqrt2, isqrt2));
          } break;
          case Gates::tdg: {
            const double isqrt2{1. / std::sqrt(2)};
            apply_gate_phase(op.qubits[0], complex_t(isqrt2, -isqrt2));
          } break;
          case Gates::mcswap:
            // Includes SWAP, CSWAP, etc
            BaseState::qreg_.apply_mcswap(op.qubits);
            break;
          case Gates::mcu3:
            // Includes u3, cu3, etc
            apply_gate_mcu3(op.qubits,
			      std::real(op.params[0]),
			      std::real(op.params[1]),
			      std::real(op.params[2]));
            break;
          case Gates::mcu2:
	    // Includes u2, cu2, etc
	    apply_gate_mcu3(op.qubits,
		    apply_gate(op);
		    break;
          */
          default:
            throw std::invalid_argument("QubitVector::State::invalid instruction \'" +
                                        op.name + "\'.");
        }
     }
    
    for(int i = 0; i < Nqs; i++){
      check_and_inject(ds_reg, gate_position, oqs[i], x_errors, y_errors, z_errors);
      gate_position += 1;
    }

  }
  /*for(i=0;i<qs;i++){
    ds_Hadamard(ds_reg, i, 1);		
  }
  for(i=0;i<qs;i++){
    ds_Z(ds_reg, i, 1);		
  }
  for(i=0;i<qs;i++){
    ds_Hadamard(ds_reg, i, 1);		
  }*/
}

ExperimentData StatevectorController::direct_ds_sim(const Circuit &circ,
                                 const Noise::NoiseModel& noise,
                                 const json_t &config,
                                 uint_t shots,
                                 uint_t rng_seed) const {
 {

    Statevector::State<> state;
    // Set config
    state.set_config(config);
    double error_rate=-1.0;
    JSON::get_value(error_rate, "ds_error_rate",config);
    // Rng engine
    //RngEngine rng;
    //rng.set_seed(rng_seed);

    // Output data container
    ExperimentData data;
    data.set_config(config);   
    
    int i;
    int qs = circ.num_qubits;

    int gate_time=1;
    long ds_seed_offset = 0;
    JSON::get_value(ds_seed_offset, "ds_seed_offset",config);

    long ds_seed_py = rng_seed+ds_seed_offset;
    ds_initialize_simulator(ds_seed_offset);
    
    ds_Register ds_reg;
    ds_reg = ds_create_register(qs, error_rate, 0);
    ds_set_state(ds_reg, 0, 1, 0);
    // apply ops	
    //state.apply_ops(circ.ops, data, rng);
    std::unordered_set <int> x_errors, y_errors, z_errors;
    ds_apply_ops(ds_reg, circ, x_errors, y_errors, z_errors);
    //std::vector<std::complex<double>> state_from_sim;
    //cvector_t state_from_sim;

    std::time_t done_sim = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    //std::cout << "done sim at: "<<std::ctime(&done_sim) <<"nc: " <<ds_reg.nc<<std::endl;

    //std::vector<complex_t> state_from_sim;
    //data.sv_vec.reserve(ds_reg.nc);

    //PyObject* resList = PyList_New(ds_reg.nc);
    data.sv_list = PyList_New(ds_reg.nc);

    //std::cout<<"injected "<<ds_reg.n_errors[0]<<" X-errors "<<ds_reg.n_errors[1]<<" Z-errors and "<<ds_reg.n_errors[2]<<" Y-Errors" << std::endl;
    

    data.add_additional_data("x_errors", ds_reg.n_errors[0]);
    data.add_additional_data("z_errors", ds_reg.n_errors[1]);
    data.add_additional_data("y_errors", ds_reg.n_errors[2]);
    for(i=0;i<ds_reg.nc;i++){
      //std::cout<<" i: "<<i<<" x: "<<ds_reg.state[i].x<<" <: "<<ds_reg.state[i].y<<std::endl;
      //data.sv_vec.emplace_back(std::complex<double>{ds_reg.state[i].x, ds_reg.state[i].y});

      //PyList_SetItem(data.sv_list, i, PyComplex_FromDoubles(0.1,-0.3));
      PyList_SetItem(data.sv_list, i, PyComplex_FromDoubles(ds_reg.state[i].x, ds_reg.state[i].y));
    }

    /*
    for(i=0;i<ds_reg.nc;i++){
      //std::ostringstream strs;
      //strs <<std::setprecision(std::numeric_limits<double>::max_digits10)<< ds_reg.state[i].x<<"+"<<ds_reg.state[i].y<<"j";
      //state_from_sim.emplace_back(std::complex<double>{ds_reg.state[i].x, ds_reg.state[i].y});
      char buffer1[256] = { '\0' };
      char buffer2[256] = { '\0' };
      dtoa_milo(ds_reg.state[i].x, buffer1);
      dtoa_milo(ds_reg.state[i].y, buffer2);

      std::string strbuffer1(buffer1);
      std::string strbuffer2(buffer2);
      //strbuffer+= strbuffer1 + "+"+strbuffer2+"j";
      state_from_sim.emplace_back(strbuffer1+"+"+strbuffer2+"j");
      //state_from_sim.emplace_back(boost::lexical_cast<std::string>(ds_reg.state[i].x)+"+"+boost::lexical_cast<std::string>(ds_reg.state[i].y)+"j");
    }
    */
    //ds_print(ds_reg);
    // translate ds_reg to std::vector<std::complex<RealType>> ret;
    //

    std::time_t bfddone_sim = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    //std::cout << "bafore destroy at: "<<std::ctime(&bfddone_sim) <<std::endl;
    //:wqds_reg = &ds_reg;
   //std::cout<<"injected "<<ds_reg.n_errors<<" errors" << std::endl;
    ds_destroy_register(ds_reg);
    //state.add_creg_to_data(data);
    //json_t jdata = state_from_sim; 
    // Add final state to the data
    //json_t datRes = state_from_sim;
    //data.add_additional_data("statevector",datRes);
    //data.sv_vec=state_from_sim;
    //data.sv_list = resList;
    //ExperimentData data2;
    //data2.set_config(config);   
    //data.sim_reg = ds_reg;
    std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    //std::cout << "done sim at: "<<std::ctime(&end_time) <<std::endl;
    return data;
  }
}

ExperimentData StatevectorController::inject_n_errors(const Circuit &circ,
                                 const Noise::NoiseModel& noise,
                                 const json_t &config,
                                 uint_t shots,
                                 uint_t rng_seed) const {
    Statevector::State<> state;
    // Set config
    state.set_config(config);
    double error_rate=-1.0;
    JSON::get_value(error_rate, "ds_error_rate",config);
    // Rng engine
    RngEngine rng;
    rng.set_seed(rng_seed);

    // Output data container
    ExperimentData data;
    data.set_config(config);   
    
    int i;
    int qs = circ.num_qubits;

    int gate_time=1;
    long ds_seed_offset = 1;
    JSON::get_value(ds_seed_offset, "ds_seed_offset",config);

    long ds_seed_py = rng_seed+ds_seed_offset;
    ds_initialize_simulator(ds_seed_offset);
    
    ds_Register ds_reg;
    //error-free, errors are injected using x_/y_/z_errors in ds_apply_ops
    ds_reg = ds_create_register(qs, 0, 0);
    ds_set_state(ds_reg, 0, 1, 0);
    // apply ops	
    //state.apply_ops(circ.ops, data, rng);
    std::unordered_set <int> x_errors, y_errors, z_errors;

    //std::cout << "ds::before first apply_ops" <<std::endl;
    ds_apply_ops(ds_reg, circ, x_errors, y_errors, z_errors);
    //std::vector<std::complex<double>> state_from_sim;
    //cvector_t state_from_sim;

    //std::cout << "ds::after first apply_ops" <<std::endl;
    int *qubits_to_measure = new int[qs];
    for(int j=0;j<qs;j++){
      qubits_to_measure[j] = qs - j - 1;
    }

    //std::cout << "ds::before measure" <<std::endl;
    double correct_result = ds_set_measure(ds_reg, qs, qubits_to_measure, (1 << qs) - 1);
    int locations = 0;
    // locations gives how many positions there are to inject an error
    for(int j = 0; j<qs; j++){
      locations += ds_reg.steps[j];
    }
    // allow for init errors
    locations += qs;
    //std::cout << " number of locations in circuit: " << locations << std::endl;
    //get number of errors
    long num_x_errors, num_y_errors, num_z_errors;  
    num_x_errors = 0;
    num_y_errors = 0;
    num_z_errors = 0;

    JSON::get_value(num_x_errors, "ds_num_x_errors", config);
    JSON::get_value(num_y_errors, "ds_num_y_errors", config);
    JSON::get_value(num_z_errors, "ds_num_z_errors", config);
    
    // create arrays that store the location of the errors
    std::vector<int> x_error_locations, y_error_locations, z_error_locations;

    x_error_locations.resize(num_x_errors);
    y_error_locations.resize(num_y_errors);
    z_error_locations.resize(num_z_errors);

    // repeat simulations num_iterations times
    int num_iterations;  
    num_iterations = 1;
    JSON::get_value(num_iterations, "ds_num_iterations", config);
    std::vector<double> result_ar;
    result_ar.reserve(num_iterations);

    //std::cout << "ds::before iterations" <<std::endl;
    for(int iteration=0;iteration<num_iterations;iteration++){

      x_errors.clear();
      y_errors.clear();
      z_errors.clear();

      for(int j=0; j<num_x_errors; j++)
        x_error_locations.at(j) = rng.rand_int((int_t) 0, (int_t) (locations-1)); //interval excluding #locations

      for(int j=0; j<num_y_errors; j++)
        y_error_locations.at(j) = rng.rand_int((int_t)0, (int_t) locations-1); //interval excluding #locations

      for(int j=0; j<num_z_errors; j++)
        z_error_locations.at(j) = rng.rand_int((int_t)0, (int_t) locations-1); //interval excluding #locations

      // only inject if there is an odd number of errors at a location
      for(int j=0; j<num_x_errors; j++)
        if(std::count(x_error_locations.begin(), x_error_locations.end(), x_error_locations[j]) % 2 == 1)
          x_errors.insert(x_error_locations[j]);
 
      for(int j=0; j<num_y_errors; j++)
        if(std::count(y_error_locations.begin(), y_error_locations.end(), y_error_locations[j]) % 2 == 1)
          y_errors.insert(y_error_locations[j]);

      for(int j=0; j<num_z_errors; j++)
        if(std::count(z_error_locations.begin(), z_error_locations.end(), z_error_locations[j]) % 2 == 1)
          z_errors.insert(z_error_locations[j]);
      // reset ds_reg and start new simulation with errors injected at locations given by unordered sets x_/y_/z_errors
      ds_clearreg(ds_reg);
      ds_set_state(ds_reg, 0, 1, 0);   

      //std::cout << "ds::before iteration apply ops" <<std::endl;
      ds_apply_ops(ds_reg, circ, x_errors, y_errors, z_errors);

      //std::cout << "ds::after iteration apply ops" <<std::endl;
      double result_with_error = ds_set_measure(ds_reg, qs, qubits_to_measure, (1<<qs) - 1);

      //std::cout << "ds::after iteration apply op measurement: "<< result_with_error <<std::endl;
      result_ar.push_back(result_with_error);
    }
    //TODO: return result_ar
    //

    //std::cout << "ds::done adding info to return object" <<std::endl;
    data.add_additional_data("result-ar", result_ar);
    data.add_additional_data("correct-result", correct_result);

    //data.add_additional_data("x_errors", ds_reg.n_errors[0]);
    //data.add_additional_data("z_errors", ds_reg.n_errors[1]);
    //data.add_additional_data("y_errors", ds_reg.n_errors[2]);

    data.sv_list = PyList_New(ds_reg.nc);

    //std::cout<<"injected "<<ds_reg.n_errors[0]<<" X-errors "<<ds_reg.n_errors[1]<<" Z-errors and "<<ds_reg.n_errors[2]<<" Y-Errors" << std::endl;
    

    data.add_additional_data("x_errors", ds_reg.n_errors[0]);
    data.add_additional_data("z_errors", ds_reg.n_errors[1]);
    data.add_additional_data("y_errors", ds_reg.n_errors[2]);
    for(i=0;i<ds_reg.nc;i++){
      //std::cout<<" i: "<<i<<" x: "<<ds_reg.state[i].x<<" <: "<<ds_reg.state[i].y<<std::endl;
      //data.sv_vec.emplace_back(std::complex<double>{ds_reg.state[i].x, ds_reg.state[i].y});

      //PyList_SetItem(data.sv_list, i, PyComplex_FromDoubles(0.1,-0.3));
      PyList_SetItem(data.sv_list, i, PyComplex_FromDoubles(ds_reg.state[i].x, ds_reg.state[i].y));
    }

    data.add_additional_data("x_error_locs", x_errors);
    data.add_additional_data("z_error_locs", z_errors);
    data.add_additional_data("y_error_locs", y_errors);

    ds_destroy_register(ds_reg);

    delete[] qubits_to_measure;

    //std::cout << "ds::returning data" <<std::endl;
    //state.add_creg_to_data(data);
    //json_t jdata = state_from_sim; 
    // Add final state to the data
    //json_t datRes = state_from_sim;
    //data.add_additional_data("statevector",datRes);
    //data.sv_vec=state_from_sim;
    //data.sv_list = resList;
    //ExperimentData data2;
    //data2.set_config(config);   
    //data.sim_reg = ds_reg;
    //std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    //std::cout << "done sim at: "<<std::ctime(&end_time) <<std::endl;
    return data;
}

ExperimentData StatevectorController::get_hotspots(const Circuit &circ,
                                 const Noise::NoiseModel& noise,
                                 const json_t &config,
                                 uint_t shots,
                                 uint_t rng_seed) const {
/* TODO
 * similar to inject_n_errors: iterate over every location, do simulation with error at that location, divide and get probiablity for noisy result by correct result? or by 1/2^Qubits
 *
 * */

    Statevector::State<> state;
    // Set config
    state.set_config(config);
    double error_rate=-1.0;
    JSON::get_value(error_rate, "ds_error_rate",config);
    // Rng engine
    RngEngine rng;
    rng.set_seed(rng_seed);

    // Output data container
    ExperimentData data;
    data.set_config(config);   
    
    int i;
    int qs = circ.num_qubits;

    int gate_time=1;
    long ds_seed_offset = 1;
    JSON::get_value(ds_seed_offset, "ds_seed_offset",config);

    long ds_seed_py = rng_seed+ds_seed_offset;
    ds_initialize_simulator(ds_seed_offset);
    
    ds_Register ds_reg;
    //error-free, errors are injected using x_/y_/z_errors in ds_apply_ops
    ds_reg = ds_create_register(qs, 0, 0);
    ds_set_state(ds_reg, 0, 1, 0);
    // apply ops	
    //state.apply_ops(circ.ops, data, rng);
    std::unordered_set <int> x_errors, y_errors, z_errors;
    ds_apply_ops(ds_reg, circ, x_errors, y_errors, z_errors);
    //std::vector<std::complex<double>> state_from_sim;
    //cvector_t state_from_sim;
    int *qubits_to_measure = new int[qs];
    for(int j=0;j<qs;j++){
      qubits_to_measure[j] = qs - j - 1;
    }
    double correct_result = ds_set_measure(ds_reg, qs, qubits_to_measure, (1<<qs) - 1);
    int locations = 0;
    // locations gives how many positions there are to inject an error
    for(int j = 0; j<qs; j++){
      locations += ds_reg.steps[j];
    }
    // allow for init errors
    locations += qs;

    std::vector<double> result_ar_x;
    std::vector<double> result_ar_y;
    std::vector<double> result_ar_z;

    result_ar_x.reserve(locations);
    result_ar_y.reserve(locations);
    result_ar_z.reserve(locations);

    for(int location=0; location<locations;location++){
      x_errors.insert(location);      
      ds_clearreg(ds_reg);
      ds_set_state(ds_reg, 0, 1, 0);   
      ds_apply_ops(ds_reg, circ, x_errors, y_errors, z_errors);
      double result_with_error = ds_set_measure(ds_reg, qs, qubits_to_measure, (1<<qs) - 1);
      result_ar_x[location] = result_with_error;
      x_errors.clear(); 
    }

    for(int location=0; location<locations;location++){
      y_errors.insert(location);      
      ds_clearreg(ds_reg);
      ds_set_state(ds_reg, 0, 1, 0);   
      ds_apply_ops(ds_reg, circ, x_errors, y_errors, z_errors);
      double result_with_error = ds_set_measure(ds_reg, qs, qubits_to_measure, (1<<qs) - 1);
      result_ar_y[location] = result_with_error;
      y_errors.clear(); 
    }

    for(int location=0; location<locations;location++){
      z_errors.insert(location);      
      ds_clearreg(ds_reg);
      ds_set_state(ds_reg, 0, 1, 0);   
      ds_apply_ops(ds_reg, circ, x_errors, y_errors, z_errors);
      double result_with_error = ds_set_measure(ds_reg, qs, qubits_to_measure, (1<<qs) - 1);
      result_ar_z[location] = result_with_error;
      z_errors.clear(); 
    }

    data.add_additional_data("result-ar-x",result_ar_x);
    data.add_additional_data("result-ar-y",result_ar_y);
    data.add_additional_data("result-ar-z",result_ar_z);

    data.add_additional_data("correct-result", correct_result);
}
ExperimentData StatevectorController::run_circuit(const Circuit &circ,
                                              const Noise::NoiseModel& noise,
                                              const json_t &config,
                                              uint_t shots,
                                              uint_t rng_seed) const {
  // Initialize  state
  Statevector::State<> state;

  //std::time_t sstart_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  //std::cout << "start sim at: "<<std::ctime(&sstart_time) <<std::endl;
  bool is_devitt_sim = false;
  bool is_inject_n_errors = false;
  bool is_hotspots = false;

  JSON::get_value(is_devitt_sim, "ds_direct_sim",config);
  JSON::get_value(is_inject_n_errors, "ds_inject_n_errors",config);
  JSON::get_value(is_hotspots, "ds_hotspots",config);
  if(is_devitt_sim){
    return direct_ds_sim(circ, noise, config, shots, rng_seed);
  }
  else if(is_inject_n_errors){  
    std::cout << "start running inject n errors" <<std::endl;
    return inject_n_errors(circ, noise, config, shots, rng_seed);
  }
  else if(is_hotspots){
    return get_hotspots(circ, noise, config, shots, rng_seed);
  }
  else {
    std::cout << "Old Code! Did not check functionality."<<std::endl;
    // Set config
    state.set_config(config);
    state.set_parallalization(parallel_state_update_);
    
    // Rng engine
    RngEngine rng;
    rng.set_seed(rng_seed);
 
    // Output data container
    ExperimentData data;
    data.set_config(config);
    
    // Run single shot collecting measure data or snapshots
    if (initial_state_.empty())
      state.initialize_qreg(circ.num_qubits);
    else
      state.initialize_qreg(circ.num_qubits, initial_state_);
    state.initialize_creg(circ.num_memory, circ.num_registers);
    //state.apply_ops(circ.ops, data, rng);
    state.add_creg_to_data(data);
    
    // Add final state to the data
    data.add_additional_data("statevector", state.qreg().vector());
  
    return data;
  }
}

//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
