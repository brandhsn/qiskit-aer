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

#ifndef _aer_controller_execute_hpp_
#define _aer_controller_execute_hpp_

#include <string>
#include "framework/json.hpp"
#include "misc/hacks.hpp"
#include <Python.h>
#include "framework/results/result.hpp"
#include "framework/results/experiment_data.hpp"
//=========================================================================
// Controller Execute interface
//=========================================================================


// This is used to make wrapping Controller classes in Cython easier
// by handling the parsing of std::string input into JSON objects.
namespace AER { 
template <class controller_t>
PyObject* controller_execute(const std::string &qobj_str) {
  controller_t controller;
  auto qobj_js = json_t::parse(qobj_str);

  // Fix for MacOS and OpenMP library double initialization crash.
  // Issue: https://github.com/Qiskit/qiskit-aer/issues/1
  if (JSON::check_key("config", qobj_js)) {
    std::string path;
    JSON::get_value(path, "library_dir", qobj_js["config"]);
    Hacks::maybe_load_openmp(path);
  }

  Result sim_result = controller.execute(qobj_js);
  PyObject* return_variables = PyList_New(2);
  PyList_SetItem(return_variables, 0, PyUnicode_FromString(sim_result.json().dump(-1).c_str()));
  PyList_SetItem(return_variables, 1, sim_result.results[0].data.sv_list);
  return return_variables;
}
} // end namespace AER
#endif
