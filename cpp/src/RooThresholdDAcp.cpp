#include "RooThresholdDAcp.h"

/*
  Author: Maurizio Martinelli
  Institute: CERN
  Email: maurizio.martinelli@nikhef.nl

  Name: RooThresholdDAcp
  Description: this class defines the threshold function
               used to parametrize the soft pion combinatorial
	       background.

  Date: 24/5/2019
 */

// ROOT
#include "TMath.h"
#include "Math/SpecFuncMathCore.h"
// RooFit
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooMath.h"

ClassImp(RooThresholdDAcp)

RooThresholdDAcp::RooThresholdDAcp(const char *name, const char *title,
			   RooAbsReal& _x,
				 //RooAbsReal& _x0,
			   RooAbsReal& _kinEnd,
			   RooAbsReal& _alpha,
			   RooAbsReal& _beta) :
RooAbsPdf(name,title),
  x("x","x",this,_x),
	//x0("x0","x0",this,_x0),
  kinEnd("kinEnd","kinEnd",this,_kinEnd),
  alpha("alpha","alpha",this,_alpha),
  beta("beta","beta",this,_beta)
{
}


RooThresholdDAcp::RooThresholdDAcp(const RooThresholdDAcp& other, const char* name) :
  RooAbsPdf(other,name),
  x("x",this,other.x),
  //x0("x0",this,other.x0),
  kinEnd("kinEnd",this,other.kinEnd),
  alpha("alpha",this,other.alpha),
  beta("beta",this,other.beta)
{
}

// Methods
Double_t
RooThresholdDAcp::evaluate() const
{
	// the function
	//  latex: f(x) = \Theta(x-x_{\text{th}}) (x-x_0)^a e^{-b (x-x_0)}
  if (x-kinEnd<=0.) return 1e-10;
  double u = x-kinEnd;
  return TMath::Power(u,alpha)*TMath::Exp(-beta*u);
}

// Integrals
Int_t
RooThresholdDAcp::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}

Double_t
RooThresholdDAcp::analyticalIntegral(Int_t code, const char* rangeName) const
{
  assert(code==1) ;
	// calculate some values
	//Double_t u = x-x0;
  Double_t xmin = (x.min(rangeName) > kinEnd) ? x.min(rangeName) : kinEnd;
  Double_t xmax = (x.max(rangeName) > kinEnd) ? x.max(rangeName) : kinEnd;
	Double_t umin = xmin-kinEnd;
	Double_t umax = xmax-kinEnd ;
	// avoid denominator being null
	if (beta==0) return (TMath::Power(umax,alpha+1)-TMath::Power(umin,alpha+1))/(alpha+1);

	// calculate integral
	//   latex: \int u^a\exp(-b u)du = -\frac{u^a (bu)^{-a}\Gamma(a+1,b u)}{b} + \text{const}
	//  Gamma(a,x) is the incomplete Gamma function
	//
	Double_t num_max(1.), num_min(1.);
	Double_t den(TMath::Power(beta,alpha+1));
	// calculate integral at maximum
	// num_max *= TMath::Power(umax,alpha);
	// num_max *= TMath::Power(beta*umax,-alpha);
	//num_max *= TMath::Power(beta,-alpha);
	num_max *= ROOT::Math::inc_gamma_c(alpha+1,beta*umax)*ROOT::Math::tgamma(alpha+1);// oss: inc_gamma_c is normalised upon Gamma(a)
	// calculate integral at minimum
	// num_min *= TMath::Power(umin,alpha);
	// num_min *= TMath::Power(beta*umin,-alpha);
	//num_min *= TMath::Power(beta,-alpha);
	num_min *= ROOT::Math::inc_gamma_c(alpha+1,beta*umin)*ROOT::Math::tgamma(alpha+1);
	// return definite integral
	return -(num_max-num_min)/den;
	//return 1.;

}
