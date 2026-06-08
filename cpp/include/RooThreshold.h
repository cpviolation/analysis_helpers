#ifndef ROOTHRESHOLD_H
#define ROOTHRESHOLD_H

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

// RooFit
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"

class RooThreshold : public RooAbsPdf {
public:
  RooThreshold() {} ;
  RooThreshold(const char *name, const char *title,
	      RooAbsReal& _x,
	      RooAbsReal& _kinEnd,
	      RooAbsReal& _alpha);
  RooThreshold(const RooThreshold& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooThreshold(*this,newname); }
  inline virtual ~RooThreshold() { }

protected:
  // Variables
  RooRealProxy x ;
  RooRealProxy kinEnd ;
  RooRealProxy alpha ;
  // Functions
  Double_t evaluate() const ;
  // Integrals
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

private:
  ClassDef(RooThreshold,1)
};

#endif
