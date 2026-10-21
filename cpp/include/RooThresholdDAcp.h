#ifndef ROOTHRESHOLDDACP_H
#define ROOTHRESHOLDDACP_H

/*
Author: Maurizio Martinelli
Institute: CERN
Email: maurizio.martinelli@cern.ch

Name: RooThresholdDAcp
Description: this class defines the threshold function
             used to parametrize the soft pion combinatorial
       background.

Date: 24/5/2019
 */

// RooFit
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"

class RooThresholdDAcp : public RooAbsPdf {
public:
  RooThresholdDAcp() {} ;
  RooThresholdDAcp(const char *name, const char *title,
	      RooAbsReal& _x,
      	//RooAbsReal& _x0,
	      RooAbsReal& _kinEnd,
	      RooAbsReal& _alpha,
	      RooAbsReal& _beta);
  RooThresholdDAcp(const RooThresholdDAcp& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooThresholdDAcp(*this,newname); }
  inline virtual ~RooThresholdDAcp() { }

protected:
  // Variables
  RooRealProxy x ;
  //RooRealProxy x0 ;
  RooRealProxy kinEnd ;
  RooRealProxy alpha ;
  RooRealProxy beta ;
  // Functions
  Double_t evaluate() const ;
  // Integrals
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

private:
  ClassDef(RooThresholdDAcp,1)
};

#endif
