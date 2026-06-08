#include <cmath>
#include <iostream>

#include "RooArgSet.h"
#include "RooRealVar.h"

#include "RooThreshold.h"
#include "RooThresholdDAcp.h"

int main() {
  RooRealVar x("x", "x", 2.0, 0.5, 10.0);
  RooRealVar kinEnd("kinEnd", "kinEnd", 1.0);
  RooRealVar alpha("alpha", "alpha", 1.5);

  RooThreshold threshold("threshold", "threshold", x, kinEnd, alpha);

  RooArgSet normSet(x);
  const double threshold_value = threshold.getVal(&normSet);
  if (!std::isfinite(threshold_value) || threshold_value <= 0.0) {
    std::cerr << "RooThreshold returned invalid value: " << threshold_value << "\n";
    return 1;
  }

  RooRealVar beta("beta", "beta", 2.0);
  RooThresholdDAcp threshold_dacp("threshold_dacp", "threshold_dacp", x, kinEnd, alpha, beta);

  const double dacp_value = threshold_dacp.getVal(&normSet);
  if (!std::isfinite(dacp_value) || dacp_value <= 0.0) {
    std::cerr << "RooThresholdDAcp returned invalid value: " << dacp_value << "\n";
    return 1;
  }

  return 0;
}
