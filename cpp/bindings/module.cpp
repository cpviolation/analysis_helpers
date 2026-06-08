#include <pybind11/pybind11.h>

#include "RooArgSet.h"
#include "RooRealVar.h"

#include "RooThreshold.h"
#include "RooThresholdDAcp.h"

namespace py = pybind11;

double evaluate_threshold(double x_value, double kin_end_value, double alpha_value) {
  RooRealVar x("x", "x", x_value, kin_end_value, x_value + 10.0);
  RooRealVar kinEnd("kinEnd", "kinEnd", kin_end_value);
  RooRealVar alpha("alpha", "alpha", alpha_value);

  RooThreshold threshold("threshold", "threshold", x, kinEnd, alpha);
  RooArgSet normSet(x);
  return threshold.getVal(&normSet);
}

double evaluate_threshold_dacp(double x_value, double kin_end_value, double alpha_value, double beta_value) {
  RooRealVar x("x", "x", x_value, kin_end_value, x_value + 10.0);
  RooRealVar kinEnd("kinEnd", "kinEnd", kin_end_value);
  RooRealVar alpha("alpha", "alpha", alpha_value);
  RooRealVar beta("beta", "beta", beta_value);

  RooThresholdDAcp threshold("threshold_dacp", "threshold_dacp", x, kinEnd, alpha, beta);
  RooArgSet normSet(x);
  return threshold.getVal(&normSet);
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python bindings for ROOT-based analysis_helpers classes";

  m.def("evaluate_threshold", &evaluate_threshold,
        py::arg("x"), py::arg("kin_end"), py::arg("alpha"),
        "Evaluate RooThreshold for scalar parameters.");

  m.def("evaluate_threshold_dacp", &evaluate_threshold_dacp,
        py::arg("x"), py::arg("kin_end"), py::arg("alpha"), py::arg("beta"),
        "Evaluate RooThresholdDAcp for scalar parameters.");
}
