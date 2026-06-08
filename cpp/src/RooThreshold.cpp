#include "RooThreshold.h"

/*
  Author: Maurizio Martinelli
  Institute: Nikhef
  Email: maurizio.martinelli@nikhef.nl

  Name: RooThreshold
  Description: this class defines the threshold function
               used to parametrize the soft pion combinatorial
	       background.

  Date: 5/6/2013
 */

// ROOT
#include "TMath.h"
// RooFit
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooMath.h"

ClassImp(RooThreshold)

RooThreshold::RooThreshold(const char *name, const char *title,
			   RooAbsReal& _x,
			   RooAbsReal& _kinEnd,
			   RooAbsReal& _alpha) :
RooAbsPdf(name,title),
  x("x","x",this,_x),
  kinEnd("kinEnd","kinEnd",this,_kinEnd),
  alpha("alpha","alpha",this,_alpha)
{
}


RooThreshold::RooThreshold(const RooThreshold& other, const char* name) :
  RooAbsPdf(other,name),
  x("x",this,other.x),
  kinEnd("kinEnd",this,other.kinEnd),
  alpha("alpha",this,other.alpha)
{
}

// Methods
Double_t
RooThreshold::evaluate() const
{
  if (x-kinEnd<=0.) return 1e-10;
  double xFrac = x/kinEnd;
  double u = xFrac*xFrac-1;
  return x*TMath::Sqrt(u)*TMath::Exp(-alpha*u);
}

// Integrals
Int_t
RooThreshold::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}

Double_t
RooThreshold::analyticalIntegral(Int_t code, const char* rangeName) const
{
  assert(code==1) ;

  static const Double_t sqrtPi = TMath::Sqrt(atan2(0.0,-1.0));
  Double_t min = (x.min(rangeName) > kinEnd) ? x.min(rangeName) : kinEnd;
  Double_t max = (x.max(rangeName) > kinEnd) ? x.max(rangeName) : kinEnd;
  Double_t minFrac2 = (min*min)/(kinEnd*kinEnd);
  Double_t maxFrac2 = (max*max)/(kinEnd*kinEnd);
  Double_t fmin = TMath::Sqrt(minFrac2-1.);
  Double_t fmax = TMath::Sqrt(maxFrac2-1.);
  Double_t sqrtAlpha = TMath::Sqrt(alpha);
  Double_t a1, a2 ;
  a1 = sqrtPi*(RooMath::erf(sqrtAlpha*fmax)-RooMath::erf(sqrtAlpha*fmin));
  a2 = 2.*sqrtAlpha*(TMath::Exp(alpha*(1-minFrac2))*fmin-TMath::Exp(alpha*(1-maxFrac2))*fmax);

  return kinEnd*kinEnd*(a1+a2)*0.25/(alpha*sqrtAlpha);

}
