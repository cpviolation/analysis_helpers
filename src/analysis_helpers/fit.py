try:
    import ROOT as r  # type: ignore[import-not-found]
    _HAS_ROOT = True
except ModuleNotFoundError:
    _HAS_ROOT = False

    class _MissingROOT:
        """Proxy that raises a clear error only when ROOT-backed code is used."""

        def __getattr__(self, _name):
            raise ImportError(
                "PyROOT is not available. Install and configure ROOT to use ROOT-dependent fit tools."
            )

    r = _MissingROOT()
import math
import re
import os
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import mplhep
mplhep.style.use('LHCb2')
from array import array
from .root_helpers import DefineTree, LoadCompiledLibraries, ROOT2MPLLineStyle, ROOT2MPLColor, ROOT2MPLText
from .utils import get_temporary_file_name
from .plotting import configure_plot_layout
import copy


def WS(ws, obj, opts=None):
    """A RooWorkspace helper for python

    Args:
        ws (ROOT.RooWorkspace): The RooWorkspace to import the object into
        obj (ROOT.TObject): The ROOT object to import
        opts (list, optional): options. Defaults to [r.RooFit.RecycleConflictNodes()].

    Raises:
        TypeError: The object is not of the same class as the one in the workspace

    Returns:
        ROOT.TObject: The object added to the workspace
    """
    if opts is None:
        opts = [r.RooFit.RecycleConflictNodes()]

    name = obj.GetName()
    wsobj = ws.obj(name)
    if obj.InheritsFrom('RooAbsArg') or obj.InheritsFrom('RooAbsData'):
        if not wsobj:
            if len(opts) > 0:
                ws.__getattribute__('import')(obj, *opts)
            else:
                ws.__getattribute__('import')(obj)
            wsobj = ws.obj(name)
        else:
            if wsobj.Class() != obj.Class():
                wsobj.Print()
                obj.Print()
                raise TypeError()
    else:
        if not wsobj:
            ws.__getattribute__('import')(obj, name)
            wsobj = ws.obj(name)
        else:
            if wsobj.Class() != obj.Class():
                raise TypeError()
    return wsobj


class FitModels:
    """This class contains a few RooFit models of general use

    Returns:
        class: An object to control the functions
    """
    def __init__(self, _name='FitModels'):
        """
        Initial call, define what's needed and what's missing
        """
        self.name = _name
        return


    def SimpleWorkspace(self, obsname='x', xmin=-10, xmax=10, title=None, unit=''):
        """Creates a simple workspace

        Args:
            obsname (str, optional): the observable name. Defaults to 'x'.
            xmin (float, optional): the minimum value of the observable
            xmax (float, optional): the maximum value of the observable
            title (str, optional): the title of the observable. Defaults to None.
            unit (str, optional): the unit of the observable. Defaults to ''.

        Returns:
            RooWorkspace: a RooWorkspace with the created observable
        """
        ws = r.RooWorkspace('ws','ws')
        WS(ws, r.RooRealVar(obsname, obsname if title is None else title,
                            xmin, xmax, unit))
        return ws


    def GetObservable(self, ws, obsname):
        """Retrieve the observable from the workspace

        Args:
            ws (RooWorkspace): The workspace
            obsname (str): observable name

        Returns:
            r.RooRealVar: the observable variable
        """
        x = ws.var(obsname)
        if x is None:
            print('Please choose a valid name for the observable')
            return None
        return x


    def Gaussian(self, ws, name, obsname):
        """Add a gaussian PDF to the workspace and returns it

        Args:
            ws (RooWorkspace): the workspace
            name (str): name of the gaussian PDF
            obsname (str): name of the observable

        Returns:
            RooGaussian: the Gaussian PDF
        """
        # A model of a Gaussian
        x = self.GetObservable(ws,obsname)
        xmin, xmax = x.getMin(), x.getMax()
        mean = 0.5*(xmin+xmax)
        unit = x.getUnit()
        # G1
        m = WS(ws, r.RooRealVar(name+'_m','#mu_{{1}} ({})'.format(name),mean,xmin,xmax,unit))
        s = WS(ws, r.RooRealVar(name+'_s','#sigma_{{1}} ({})'.format(name),0.1*(xmax-xmin),0,xmax-xmin,unit))
        g = WS(ws, r.RooGaussian(name+'_gau','G ({})'.format(name),x,m,s))
        return g


    def TwoGaussians(self, ws, name, obsname):
        """A PDF made of two gaussians

        Args:
            ws (RooWorkspace): the workspace
            name (str): name of the gaussian PDF
            obsname (str): name of the observable

        Returns:
            RooAddPdf: the two Gaussian PDF
        """
        # A model sum of 2 Gaussians
        x = self.GetObservable(ws,obsname)
        xmin, xmax = x.getMin(), x.getMax()
        mean = 0.5*(xmin+xmax)
        unit = x.getUnit()
        # G1
        m1 = WS(ws, r.RooRealVar(name+'_m1','#mu_{{1}} ({})'.format(name),mean,xmin,xmax,unit))
        s1 = WS(ws, r.RooRealVar(name+'_s1','#sigma_{{1}} ({})'.format(name),0.1*(xmax-xmin),0,xmax-xmin,unit))
        g1 = WS(ws, r.RooGaussian(name+'_g1','G_{{1}} ({})'.format(name),x,m1,s1))
        # G2
        dm2= WS(ws, r.RooRealVar(name+'_dm2','#mu_{{2}}-#mu_{{1}} ({})'.format(name),xmin-mean,xmax-mean,unit))
        m2 = WS(ws, r.RooFormulaVar(name+'_m2','@0+@1',r.RooArgList(m1,dm2)))
        rs2= WS(ws, r.RooRealVar(name+'_rs2','#sigma_{{2}}/#sigma_{{1}} ({})'.format(name),1,0,10))
        s2 = WS(ws, r.RooFormulaVar(name+'_s2','@0*@1',r.RooArgList(rs2,s1)))
        g2 = WS(ws, r.RooGaussian(name+'_g2','G_{{2}} ({})'.format(name),x,m2,s2))
        # Sum gaussians
        f1 = WS(ws, r.RooRealVar(name+'_fg1','f_{{g1}} ({})'.format(name),0.9,0,1))
        g12= WS(ws, r.RooAddPdf(name+'_g12','g_{{12}} ({})'.format(name),g1,g2,f1))
        return g12


    def ThreeGaussians(self, ws, name, obsname):
        """A PDF made of three gaussians

        Args:
            ws (RooWorkspace): the workspace
            name (str): name of the gaussian PDF
            obsname (str): name of the observable

        Returns:
            RooAddPdf: the three Gaussian PDF
        """
        # A model sum of 3 Gaussians
        x = self.GetObservable(ws,obsname)
        xmin, xmax = x.getMin(), x.getMax()
        mean = 0.5*(xmin+xmax)
        unit = x.getUnit()
        # G1
        m1 = WS(ws, r.RooRealVar(name+'_m1','#mu_{{1}} ({})'.format(name),mean,xmin,xmax,unit))
        s1 = WS(ws, r.RooRealVar(name+'_s1','#sigma_{{1}} ({})'.format(name),0.1*(xmax-xmin),0,xmax-xmin,unit))
        g1 = WS(ws, r.RooGaussian(name+'_g1','G_{{1}} ({})'.format(name),x,m1,s1))
        # G2
        dm2= WS(ws, r.RooRealVar(name+'_dm2','#mu_{{2}}-#mu_{{1}} ({})'.format(name),xmin-mean,xmax-mean,unit))
        m2 = WS(ws, r.RooFormulaVar(name+'_m2','@0+@1',r.RooArgList(m1,dm2)))
        rs2= WS(ws, r.RooRealVar(name+'_rs2','#sigma_{{2}}/#sigma_{{1}} ({})'.format(name),1,0,10))
        s2 = WS(ws, r.RooFormulaVar(name+'_s2','@0*@1',r.RooArgList(rs2,s1)))
        g2 = WS(ws, r.RooGaussian(name+'_g2','G_{{2}} ({})'.format(name),x,m2,s2))
        # G3
        dm3= WS(ws, r.RooRealVar(name+'_dm3','#mu_{{3}}-#mu_{{1}} ({})'.format(name),xmin-mean,xmax-mean,unit))
        m3 = WS(ws, r.RooFormulaVar(name+'_m3','@0+@1',r.RooArgList(m1,dm3)))
        rs3= WS(ws, r.RooRealVar(name+'_rs3','#sigma_{{3}}/#sigma_{{1}} ({})'.format(name),1,0,10))
        s3 = WS(ws, r.RooFormulaVar(name+'_s3','@0*@1',r.RooArgList(rs3,s1)))
        g3 = WS(ws, r.RooGaussian(name+'_g3','G_{{3}} ({})'.format(name),x,m3,s3))
        # Sum gaussians
        f1 = WS(ws, r.RooRealVar(name+'_fg1','f_{{g1}} ({})'.format(name),0.9,0,1))
        f2 = WS(ws, r.RooRealVar(name+'_fg2','f_{{g2}} ({})'.format(name),0.9,0,1))
        g23= WS(ws, r.RooAddPdf(name+'_g23','g_{{23}} ({})'.format(name),g2,g3,f2))
        g123=WS(ws, r.RooAddPdf(name+'_g123','g_{{123}} ({})'.format(name),g1,g23,f1))
        return g123


    def Johnson(self, ws, name, obsname):
        """A Johnson PDF

        Args:
            ws (RooWorkspace): the workspace
            name (str): name of the gaussian PDF
            obsname (str): name of the observable

        Returns:
            RooJohnson: the Johnson PDF
        """
        # A model of a JohnsonSU
        x = self.GetObservable(ws,obsname)
        xmin, xmax = x.getMin(), x.getMax()
        mean = 0.5*(xmin+xmax)
        unit = x.getUnit()
        m  = WS(ws, r.RooRealVar(name+'_mJsu','#mu_{{JSU}} ({})'.format(name),mean,xmin,xmax,unit))
        s  = WS(ws, r.RooRealVar(name+'_sJsu','#sigma_{{JSU}} ({})'.format(name),0.1*(xmax-xmin),0,xmax-xmin,unit))
        gamma = WS(ws, r.RooRealVar(name+'_gammaJsu','#nu_{{JSU}} ({})'.format(name),-1,+1))
        delta= WS(ws, r.RooRealVar(name+'_deltaJsu','#tau_{{JSU}} ({})'.format(name),0,5))
        jsu= WS(ws, r.RooJohnson(name+'_jsu','J_{{SU}} ({})'.format(name),x,m,s,gamma,delta))
        return jsu
    

    def JohnsonPlusGaussian(self, ws, name, obsname):
        """A PDF made of a Johnson and a Gaussian

        Args:
            ws (RooWorkspace): the workspace
            name (str): name of the gaussian PDF
            obsname (str): name of the observable

        Returns:
            RooAddPdf: the Jonhnson + Gaussian PDF
        """
        # A model sum of 2 Gaussians
        x = self.GetObservable(ws,obsname)
        xmin, xmax = x.getMin(), x.getMax()
        mean = 0.5*(xmin+xmax)
        unit = x.getUnit()
        # G1
        m1 = WS(ws, r.RooRealVar(name+'_m1','#mu_{{1}} ({})'.format(name),mean,xmin,xmax,unit))
        s1 = WS(ws, r.RooRealVar(name+'_s1','#sigma_{{1}} ({})'.format(name),0.1*(xmax-xmin),0,xmax-xmin,unit))
        g1 = WS(ws, r.RooGaussian(name+'_g1','G_{{1}} ({})'.format(name),x,m1,s1))
        # JSU
        dm2= WS(ws, r.RooRealVar(name+'_dm','#mu_{{JSU}}-#mu_{{1}} ({})'.format(name),xmin-mean,xmax-mean,unit))
        m2 = WS(ws, r.RooFormulaVar(name+'_mJsu','@0+@1',r.RooArgList(m1,dm2)))
        rs2= WS(ws, r.RooRealVar(name+'_rs2','#sigma_{{JSU}}/#sigma_{{1}} ({})'.format(name),1,0,10))
        s2 = WS(ws, r.RooFormulaVar(name+'_s2','@0*@1',r.RooArgList(rs2,s1)))
        gamma = WS(ws, r.RooRealVar(name+'_gammaJsu','#gamma_{{JSU}} ({})'.format(name),-1,+1))
        delta = WS(ws, r.RooRealVar(name+'_deltaJsu','#delta_{{JSU}} ({})'.format(name),0,5))
        jsu= WS(ws, r.RooJohnson(name+'_jsu','J_{{SU}} ({})'.format(name),x,m2,s2,gamma,delta))
        # Sum PDFs
        f1 = WS(ws, r.RooRealVar(name+'_fg1','f_{{g1}} ({})'.format(name),0.9,0,1))
        g1_jsu= WS(ws, r.RooAddPdf(name+'_g1_jsu','g_{{1}} + J_{{SU}} ({})'.format(name),g1,jsu,f1))
        return g1_jsu


    def DmSignal(self, ws, name, obsname, sig_type=0):
        """Define a fit model with different layers of complexity
            sig_type== 0 : base line, JSU + G1 + G2
                       1 : simpler, JSU + G1
                       2 : simplest, JSU
                       3 : complex, JSU + G1 + G2 + G3
                       4 : classic, G1 + G2
                       5 : classic complex, G1 + G2 + G3
                       6 : Johnson + 1 Gaussian low correlations
                
        Args:
            ws (RooWorkspace): the workspace
            name (str): the name of the signal pdf
            obsname (str): the observable name
            sig_type (int, optional): the model choice. Defaults to 0.

        Returns:
            RooAbsPdf: The resulting PDF
        """
        sig_pdf = None
        if sig_type<4:
            # Core: 1 Johnson
            jsu= self.Johnson(ws, name, obsname)
            if sig_type==1:
                # Add 1 gaussian
                fjsu = WS(ws, r.RooRealVar(name+'_fjsu','f_{{JSU}} ({})'.format(name),0.8,0,1))
                sig_pdf = WS(ws, r.RooAddPdf(name+'_sig','sig ({})'.format(name),jsu,self.Gaussian(ws, name, obsname),fjsu))
            elif sig_type==2:
                sig_pdf = jsu
            elif sig_type==3:
                # Add 3 gaussians
                fjsu = WS(ws, r.RooRealVar(name+'_fjsu','f_{{JSU}} ({})'.format(name),0.8,0,1))
                sig_pdf = WS(ws, r.RooAddPdf(name+'_sig','sig ({})'.format(name),jsu,self.ThreeGaussians(ws, name, obsname),fjsu))
            else:
                # default: JSU+2 Gaussians
                fjsu = WS(ws, r.RooRealVar(name+'_fjsu','f_{{JSU}} ({})'.format(name),0.8,0,1))
                sig_pdf = WS(ws, r.RooAddPdf(name+'_sig','sig ({})'.format(name),jsu,self.TwoGaussians(ws, name, obsname),fjsu))
        if sig_type==4:
            sig_pdf = self.TwoGaussians(ws, name, obsname)
        if sig_type==5:
            sig_pdf = self.ThreeGaussians(ws, name, obsname)
        if sig_type==6:
            sig_pdf = self.JohnsonPlusGaussian(ws, name, obsname)
        if sig_pdf is None:
            print('Cannot define the signal pdf, please choose a valid value in (0-5)')
        sig_pdf.SetName(name+'_sig')
        return sig_pdf


    def DmBackground(self, ws, name, obsname, xth=None, bkg_type=0):
        """Define a fit model with different layers of complexity
            bkg_type == 0 : RooThresholdDAcp
                        1 : RooThreshold
                        2 : RooDstD0BG

        Args:
            ws (RooWorkspace): the workspace
            name (str): the name of the signal pdf
            obsname (str): the observable name
            xth (float, optional): the threshold value. Defaults to None.
            bkg_type (int, optional): the background model choice. Defaults to 0.

        Returns:
            RooAbsPdf: the background model
        """
        x = self.GetObservable(ws,obsname)
        unit = x.getUnit()
        if xth is None: xth = x.getMin()

        # Background - threshold function
        th = WS(ws, r.RooRealVar(name+'_th','th ({})'.format(name),xth,unit))
        bkg_pdf = None
        a = WS(ws, r.RooRealVar(name+'_a','a ({})'.format(name),0.6,0,3.))
        a.setError(0.1)
        if bkg_type==1:
            try:
                r.RooThreshold
            except AttributeError:
                LoadCompiledLibraries()
            a.setVal(2) 
            a.setMin(-10)
            a.setMax(10)
            bkg_pdf = WS(ws, r.RooThreshold(name+'_bkg','bkg ({})'.format(name),x,th,a) )
        elif bkg_type==2:
            a.setVal(0.029050); a.setMin(-10); a.setMax(10);
            b = WS(ws, r.RooRealVar(name+'_b','b ({})'.format(name),2.3769,-10,10))
            b.setError(0.03)
            c = WS(ws, r.RooRealVar(name+'_c','c ({})'.format(name),0,-10,10))
            c.setError(0.03)
            bkg_pdf = WS(ws, r.RooDstD0BG(name+'_bkg','bkg ({})'.format(name),x,th,c,a,b))
        else:
            try:
                r.RooThresholdDAcp
            except AttributeError:
            #raise ValueError('RooThresholdDAcp still to be implemented')
                LoadCompiledLibraries()
            b = WS(ws, r.RooRealVar(name+'_b','b ({})'.format(name),0.06,0,0.1))
            b.setError(0.03)
            bkg_pdf = WS(ws, r.RooThresholdDAcp(name+'_bkg','bkg ({})'.format(name),x,th,a,b))
        if bkg_pdf is None:
            print('Cannot define the background pdf, please check your options')
            return None
        return bkg_pdf


    def DmModel(self, ws, name, obsname, xth=None, mtype=0, nevs=0):
        """Define a fit model with different layers of complexity
         mtype== 0 : base line, JSU + G1 + G2
                 1 : simpler, JSU + G1
                 2 : simplest, JSU
                 3 : complex, JSU + G1 + G2 + G3
                 4 : classic, G1 + G2
                 5 : classic complex, G1 + G2 + G3
                 6 : Johnson + 1 Gaussian low correlations
          mtype+ 0 : RooThresholdDAcp
          mtype+10 : RooThreshold
          mtype+20 : RooDstD0BG

        Args:
            ws (RooWorkspace): the workspace
            name (str): the name of the signal pdf
            obsname (str): the observable name
            xth (float, optional): the threshold value. Defaults to None.
            mtype (int, optional): the model choice. Defaults to 0.
            nevs (int, optional): number of events of the extended model. Defaults to 0.

        Returns:
            RooAbsPdf: the signal+background model
        """
        x = self.GetObservable(ws,obsname)
        if xth is None: xth = x.getMin()

        sig_type = mtype%10
        sig_pdf = self.DmSignal(ws, name, obsname, sig_type)

        bkg_type = int(mtype/10)
        bkg_pdf = self.DmBackground(ws, name, obsname, xth, bkg_type)

        if nevs != 0:
            # create an extended model
            ns = WS(ws, r.RooRealVar(name+'_nsig','nsig ({})'.format(name),0.5*nevs,0,nevs))
            ns.setError(0.05*nevs)
            nb = WS(ws, r.RooRealVar(name+'_nbkg','nbkg ({})'.format(name),0.5*nevs,0,nevs))
            nb.setError(0.05*nevs)
            return WS(ws, r.RooAddPdf(name+'_model','model ({})'.format(name), r.RooArgList(sig_pdf,bkg_pdf),r.RooArgList(ns,nb)))
        else:
            fsig = WS(ws, r.RooRealVar(name+'_fsig','f_{{sig}} ({})'.format(name),0.5,0,1))
            return WS(ws, r.RooAddPdf(name+'_model','model ({})'.format(name),sig_pdf,bkg_pdf,fsig))
        return None # just in case


    def DmModelBinFlip(self, ws, name, obsname, nevs, xth=None, mtype=0, ratios=True):
        """Define a fit model with different layers of complexity
         mtype== 0 : base line, JSU + G1 + G2
                 1 : simpler, JSU + G1
                 2 : simplest, JSU
                 3 : complex, JSU + G1 + G2 + G3
                 4 : classic, G1 + G2
                 5 : classic complex, G1 + G2 + G3
                 6 : Johnson + 1 Gaussian low correlations
          mtype+ 0 : RooThresholdDAcp
          mtype+10 : RooThreshold
          mtype+20 : RooDstD0BG

        Args:
            ws (RooWorkspace): the workspace
            name (str): the name of the signal pdf
            obsname (str): the observable name
            nevs (int): number of events of the extended model.
            xth (float, optional): the threshold value. Defaults to None.
            mtype (int, optional): the model choice. Defaults to 0.
            ratios (bool, optional): whether the model is used to calculate bin-flip ratios. Defaults to True.

        Returns:
            RooSimultaneousPdf: the signal+background model
        """
        x = self.GetObservable(ws,obsname)
        if xth is None: xth = x.getMin()

        sig_type = mtype%10
        sig_pdf = self.DmSignal(ws, name, obsname, sig_type)

        bkg_type = int(mtype/10)
        bkg_pdf = self.DmBackground(ws, name, obsname, xth, bkg_type)

        # define ratios for bin-flip
        Rb_sig, ns, ns_minus_b, ns_plus_b, nb_minus_b, nb_plus_b = [], [], [], [], [], []
        model_minus_b, model_plus_b = [], []
        for i in range(1, 9):
            if ratios:
                # signal ratios
                Rb_sig.append(WS(ws, r.RooRealVar(name+f'_R{i}_sig','R_{{{}}} (sig) ({})'.format(i, name),0.5,0.0,1.0)))
                # signal yields
                ns.append(WS(ws, r.RooRealVar(name+f'_nsig_{i}','nsig_{{{}}} ({})'.format(i, name),0.5*nevs,0,nevs)))
                ns[-1].setError(0.05*nevs)
                ns_minus_b.append(r.RooFormulaVar(name+f'_nsig_minus_b_{i}','@0*@1/(1+@1)',r.RooArgList(ns[-1],Rb_sig[-1])))
                ns_plus_b.append(r.RooFormulaVar(name+f'_nsig_plus_b_{i}','@0/(1+@1)',r.RooArgList(ns[-1],Rb_sig[-1])))
            else:
                # for sweights keep yields the same
                ns_minus_b.append(WS(ws, r.RooRealVar(name+f'_nsig_minus_b_{i}','nsig (-{}) ({})'.format(i, name),0.5*nevs,0,nevs)))
                ns_minus_b[-1].setError(0.05*nevs)
                ns_plus_b.append(WS(ws, r.RooRealVar(name+f'_nsig_plus_b_{i}','nsig (+{}) ({})'.format(i, name),0.5*nevs,0,nevs)))
                ns_plus_b[-1].setError(0.05*nevs)
            # background yields
            nb_minus_b.append(WS(ws, r.RooRealVar(name+f'_nbkg_minus_b_{i}','nbkg (-{}) ({})'.format(i, name),0.5*nevs,0,nevs)))
            nb_minus_b[-1].setError(0.05*nevs)
            nb_plus_b.append(WS(ws, r.RooRealVar(name+f'_nbkg_plus_b_{i}','nbkg (+{}) ({})'.format(i, name),0.5*nevs,0,nevs)))
            nb_plus_b[-1].setError(0.05*nevs)
            # models for bin-flip
            model_minus_b.append(WS(ws, r.RooAddPdf(name+f'_model_minus_b_{i}','model (-{}) ({})'.format(i, name),
                                r.RooArgList(sig_pdf,bkg_pdf),r.RooArgList(ns_minus_b[-1],nb_minus_b[-1]))))
            model_plus_b.append(WS(ws, r.RooAddPdf(name+f'_model_plus_b_{i}','model (+{}) ({})'.format(i, name),
                                r.RooArgList(sig_pdf,bkg_pdf),r.RooArgList(ns_plus_b[-1],nb_plus_b[-1]))))
        # create a simultaneous pdf
        category = WS(ws, r.RooCategory(name+'_bin_cat','category ({})'.format(name)))
        for i in range(1, 9):
            category.defineType('minus_{}'.format(i))
            category.defineType('plus_{}'.format(i))
        simpdf = WS(ws, r.RooSimultaneous(name+'_model','simpdf ({})'.format(name),category))
        for i in range(1, 9):
            simpdf.addPdf(model_minus_b[i-1],'minus_{}'.format(i))
            simpdf.addPdf(model_plus_b[i-1],'plus_{}'.format(i))
        return simpdf


class FitUtils:
    """
    This class contains a few methods to extend RooFit capabilities

    Call: FitUtils(name)
    
    Functions:
        - getResiduals(frame, ranges)
        - plotFitAndResiduals(frame, name, title, tag, xSize, ySize)
        - getLegend({name: {index, color, style}})
        - getIntegralsAndEvents(model, xmin, xmax, varName, sigName, bkgName, sigEvName, bkgEvName)
        - getSignificance(model, xmin, xmax, varName, sigName, bkgName, sigEvName, bkgEvName)
        - getPurity(model, xmin, xmax, varName, sigName, bkgName, sigEvName, bkgEvName)
        - fitToDataAlt(model, data, outname, kNLL, kChi, kMinos, kExtended, nthreads, ranges, poissonError, constraints, kSilent)
        - fitToData(model, data, outname, kNLL, kChi, kMinos, kExtended, nthreads, ranges, poissonError, constraints, kSilent)
        - combDataset(name, title, vars, category, dsets)
        - SaveFitResults(res, oname, verbose)
        - FixVariables(pdf, fix=True, verbose=True)
        - GetVarsList(ws, vlist)
        - GetVarsSet(ws, vlist)
        - SetVarsPrintDigits(ws, ndig)
        - Sweights(ws, data, model, pdfs, nevs, verbose)
        - resultsToDictionary(resfile)
        - searchForFitProblems(resfile)
        - GetWorkspaceFromFile(ws_name, file_name)
    """
    def __init__( self, _name='FitUtils' ):
        """
        Initial call, define what's needed and what's missing
        """
        self.name = _name

    # This function returns a RooPlot with the residuals
    def getResiduals(self, frame, ranges=None):
        """Returns a RooPlot with the residuals

        Args:
            frame (RooFrame): a RooFit frame where the residuals will be plotted
            ranges (list, optional): A list with the defined fit ranges to plot for the RooFrame variable. Defaults to None.

        Returns:
            dict( frame, objs): a dictionary with the RooPlot with the residuals (frame) and the list of objects to be added to the canvas (objs)
        """
        # create a list of extra object to return with the canvas
        extra = []
        # the actual frame with residuals
        frameRes = frame.getPlotVar().frame()
        if ranges is not None:
            dataHist = frame.getHist(frame.getObject(0).GetName())
            curves, hresid = [], []
            for i in range(1, len(ranges)+1):
                curves += [frame.getObject(i)]
                hresid += [dataHist.makePullHist(curves[i-1], True)]
                for j in range(hresid[i-1].GetN()):
                    hresid[i-1].SetPointError(j, 0., 0., 0., 0.)
                frameRes.addObject(hresid[i-1], "PZ")
        else:
            resHist = frame.pullHist()
            for i in range(0, resHist.GetN()):
                resHist.SetPointError(i, 0., 0., 0., 0.)
            frameRes.addObject(resHist, "PZ")
        frameRes.SetMinimum(-5.)
        frameRes.SetMaximum(+5.)
        frameRes.SetTitle("")
        frameRes.GetYaxis().SetNdivisions(110)
        frameRes.GetYaxis().SetLabelSize(0.)
        frameRes.GetYaxis().SetTitle("Pull")
        frameRes.GetYaxis().CenterTitle()
        frameRes.GetYaxis().SetTitleSize(0.17)
        frameRes.GetYaxis().SetTitleOffset(0.38)
        frameRes.GetXaxis().SetNdivisions(frame.GetXaxis().GetNdivisions())

        # axis on top of the frame
        xup = r.TGaxis(-0.05, 5., 0.05, 5., -0.05, 0.05, 510, "-")
        xup.SetLabelSize(0.)
        xup.SetNdivisions(frame.GetXaxis().GetNdivisions())
        # axis right of the frame
        #        yright = TGaxis(0.05,-5.,0.05,5.,-5.,5.,510,"+L")
        #        yright.SetLabelSize(0.)
        #        yright.SetNdivisions(110)
        frameRes.addObject(xup)
        #        frameRes.addObject(yright)
        extra.append(xup)
        #        extra.append(yright)

        # line to show residuals +3
        lResup = r.TLine(frameRes.getPlotVar().getMin(),3., frameRes.getPlotVar().getMax(),3.)
        lResup.SetLineColor(r.kRed)
        # line to show residuals -3
        lResdw = r.TLine(frameRes.getPlotVar().getMin(),-3., frameRes.getPlotVar().getMax(),-3.)
        lResdw.SetLineColor(r.kRed)
        frameRes.addObject(lResup)
        frameRes.addObject(lResdw)
        extra.append(lResup)
        extra.append(lResdw)

        # axis labels
        xLabel = frameRes.getPlotVar().getMin() - 0.01*(frameRes.getPlotVar().getMax()-frameRes.getPlotVar().getMin())
        labdw = r.TLatex(xLabel,-3.,"-3")
        labdw.SetTextSize(0.16)
        labdw.SetTextAlign(32)
        labmid = r.TLatex(xLabel, 0.," 0")
        labmid.SetTextSize(0.16)
        labmid.SetTextAlign(32)
        labup = r.TLatex(xLabel, 3.,"+3")
        labup.SetTextSize(0.16)
        labup.SetTextAlign(32)
        frameRes.addObject(labdw)
        frameRes.addObject(labmid)
        frameRes.addObject(labup)
        extra.append(labdw)
        extra.append(labmid)
        extra.append(labup)
        return dict(frame=frameRes, objs=extra)

    # This function plots the fit results with the residuals
    def plotFitAndResiduals(self, frame, name = 'cFit', title = 'Fit Results', 
                            tagLeft = [''], tagRight = [''], xSize = 600, ySize = 640, 
                            Blind=False, semilog=False, ranges=None):
        """This function plots the fit results with the residuals

        Args:
            frame (RooPlot): the Roofit frame where the fit results will be plotted
            name (str, optional): Name of the canvas. Defaults to 'cFit'.
            title (str, optional): Title of the canvas. Defaults to 'Fit Results'.
            tagLeft (list, optional): Labels added on the top left of the canvas. Defaults to [''].
            tagRight (list, optional): Labels added on the top right of the canvas. Defaults to [''].
            xSize (int, optional): Width of the canvas. Defaults to 600.
            ySize (int, optional): Height of the canvas. Defaults to 640.
            Blind (bool, optional): Do not plot the y axis labels. Defaults to False.
            semilog (bool, optional): Set the y-axis to log scale. Defaults to False.
            ranges (list, optional): A list with the defined fit ranges to plot for the RooFrame variable. Defaults to None.

        Returns:
            dict(canvas, extra): A dictionary with the canvas and the list of objects to be added to the canvas (extra)
        """
        # create a list of extra object to return with the canvas
        extra = []
        lat = r.TLatex()
        lat.SetTextSize(0.06)
        lat.SetTextAlign(13)
        xLatLeft = frame.GetXaxis().GetXmin() + 0.05*( frame.GetXaxis().GetXmax()- frame.GetXaxis().GetXmin() )
        xLatRight = frame.GetXaxis().GetXmin() + 0.95*( frame.GetXaxis().GetXmax()- frame.GetXaxis().GetXmin() )
        yLat = (0.7 if semilog else 0.96)*frame.GetMaximum()
        extra.append(lat)

        # Blind
        if Blind:
            frame.GetYaxis().SetLabelSize(0)
            frame.GetYaxis().SetTickLength(0)

        c = r.TCanvas(name,title,xSize,ySize)
        c.Divide(2)
        c.GetPad(1).SetPad(0.0,0.0,1.0,0.8)
        c.GetPad(2).SetPad(0.0,0.8,1.0,1.0)
        c.GetPad(1).SetLeftMargin(0.15)
        c.GetPad(2).SetLeftMargin(0.15)
        c.GetPad(2).SetBottomMargin(0.0)
        c.GetPad(1).SetTopMargin(0.0)
        c.cd(1)
        frame.Draw()
        if semilog: c.GetPad(1).SetLogy()
        if type(tagLeft)==str:
            lat.DrawLatex(xLatLeft,yLat,tagLeft)
        else:
            for i in range(len(tagLeft)):
                lat.DrawLatex(xLatLeft,yLat*(1-float(i)/10),tagLeft[i])
        lat.SetTextAlign(33)
        if type(tagRight)==str:
            lat.DrawLatex(xLatRight,yLat,tagRight)
        else:
            for i in range(len(tagRight)):
                lat.DrawLatex(xLatRight,yLat*(1-float(i)/10),tagRight[i])
        c.cd(2)
        frameRes = self.getResiduals(frame,ranges)
        frameRes['frame'].Draw()
        c.Update()
        extra.append(frameRes)

        return dict(canvas=c, extra=extra)


    def plotFitAndResidualsPad(self, frame, pad, tagLeft='', tagRight='', semilog=False, Blind=False, ranges=None):
        """This function plots the fit results with the residuals in two pads

        Args:
            frame (RooPlot): The Roofit frame where the fit results will be plotted
            pad (TPad): The pad where the fit results will be plotted
            tagLeft (str, optional): Labels added on the top left of the canvas. Defaults to [''].
            tagRight (list, optional): Labels added on the top right of the canvas. Defaults to [''].
            semilog (bool, optional): Set the y-axis to log scale. Defaults to False.
            Blind (bool, optional): Do not plot the y axis labels. Defaults to False.
            ranges (list, optional): A list with the defined fit ranges to plot for the RooFrame variable. Defaults to None.

        Returns:
            (list): the list with the extra objects created. Should be added to the `canvas['extra']` list.
        """
        # create a list of extra object to return with the canvas
        extra = []
        lat = r.TLatex()
        lat.SetTextSize(0.06)
        lat.SetTextAlign(13)
        xLatLeft = frame.GetXaxis().GetXmin() + 0.05*( frame.GetXaxis().GetXmax()- frame.GetXaxis().GetXmin() )
        xLatRight = frame.GetXaxis().GetXmin() + 0.95*( frame.GetXaxis().GetXmax()- frame.GetXaxis().GetXmin() )
        yLat = (0.7 if semilog else 0.96)*frame.GetMaximum()
        extra.append(lat)

        # Blind
        if Blind:
            frame.GetYaxis().SetLabelSize(0)
            frame.GetYaxis().SetTickLength(0)

        pad.Divide(2)
        pad.GetPad(1).SetPad(0.0,0.0,1.0,0.8)
        pad.GetPad(2).SetPad(0.0,0.8,1.0,1.0)
        pad.GetPad(1).SetLeftMargin(0.15)
        pad.GetPad(2).SetLeftMargin(0.15)
        pad.GetPad(2).SetBottomMargin(0.0)
        pad.GetPad(1).SetTopMargin(0.0)
        pad.GetPad(1).cd()
        frame.Draw()
        if semilog: pad.GetPad(1).SetLogy()
        lat.DrawLatex(xLatLeft,yLat,tagLeft)
        lat.SetTextAlign(33)
        lat.DrawLatex(xLatRight,yLat,tagRight)
        pad.GetPad(2).cd()
        frameRes = self.getResiduals(frame,ranges)
        frameRes['frame'].Draw()
        extra.append(frameRes)

        return extra
    
    def pull_array(self, hist, model, scale_factor=None, poisson=True):
        """Calculate pulls of data and model

        Args:
            hist (tuple): tuple containing (data_counts, data_bins) arrays from histogram
            model (array): model values to compare against data
            scale_factor (float, optional): scaling factor for model values. If None, uses bin width. Defaults to None.
            poisson (bool, optional): Use Poisson error for the counts. Defaults to True.

        Returns:
            tuple: the normalised pulls and the bin centers
        """
        # get histogram from data
        data_counts, data_bins = hist
        # get values of model in bin centers
        bins_centers = np.array([0.5*(data_bins[i]+data_bins[i+1]) for i in range(data_bins.size-1)])
        # integral of model is 1, need to scale to data
        sf = (data_bins[1]-data_bins[0]) if scale_factor is None else scale_factor
        model_counts = model * sf
        # calculate pulls
        pull = data_counts - model_counts
        data_err = np.sqrt(data_counts) if poisson else np.sqrt(data_counts) # to be fixed
        for i in range(len(data_err)):
            if data_err[i] == 0: data_err[i] = 1.
        norm_pull = pull / data_err
        return norm_pull, bins_centers
    

    def plotFitAndResidualsMPL(self, frame, axes=None, labels=None):
        """Plot the fit results with the residuals using matplotlib

        Args:
            frame (RooPlot): the Roofit frame where the fit results will be plotted
            axes (axes, optional): axes where to plot. Defaults to None.
            labels (list, optional): labels of the plotted components. Defaults to None.

        Returns:
            tuple: the figure and the axes
        """
        # convert the RooFit objects to numpy arrays
        nItems = int(frame.numItems())
        tmp_file_name = get_temporary_file_name()
        rf = r.TFile(tmp_file_name,'recreate')
        rf.cd()
        for i in range(nItems):
            frame.getObject(i).Write()
        rf.Close()
        uprf= uproot.open(tmp_file_name)
        his = uprf[frame.getObject(0).GetName()].to_numpy()
        grs = []
        for i in range(1,nItems):
            frObj = frame.getObject(i)
            _gr = uprf[frObj.GetName()].to_hist(his[1]).to_numpy()
            grs.append( {
                'x': [0.5*(_gr[1][i]+_gr[1][i+1]) for i in range(len(_gr[1])-1)],
                'y': _gr[0],
                'title': frObj.GetTitle(),
                'linestyle': ROOT2MPLLineStyle(frObj.GetLineStyle()),
                'color': ROOT2MPLColor(r.gROOT.GetColor(frObj.GetLineColor()))
            } )
        # Plotting
        fig = None
        if axes is None:
        #     fig, ax = plt.subplots(2,1,figsize=(8,6),sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            fig = plt.figure()
            gs = fig.add_gridspec(ncols=1,nrows=2,height_ratios=[1, 3],hspace=None)
            # Create the axes and impose the sharing of the x axis
            (ax1, ax2) = gs.subplots(sharex='col')
        else:
            ax1, ax2 = axes
        # plot the histogram of the data
        mplhep.histplot(his,yerr=np.sqrt(his[0]),histtype='errorbar',color='black',label='data' if labels is None else labels[0],ax=ax2)
        ax2.set_xlabel( ROOT2MPLText(frame.GetXaxis().GetTitle()) )
        ax2.set_ylabel( ROOT2MPLText(frame.GetYaxis().GetTitle()) )
        ax2.set_xlim(his[1][0],his[1][-1])
        ax2.set_ylim(bottom=0)
        # plot the model
        for i, gr in enumerate(grs):
            ax2.plot(gr['x'],gr['y'],
                     linestyle=gr['linestyle'],
                     color=gr['color'],
                     label=gr['title'] if labels is None else (labels[i+1] if i+1<len(labels) else None))
        ax2.legend()
        # create the array of the pulls
        pullA, bin_centers = self.pull_array(his, grs[-1]['y'], scale_factor=1)
        # set limits for the pull plot and label
        ax1.set_xlim(his[1][0],his[1][-1])
        ax1.set_ylim(-5,5)
        ax1.set_ylabel('Pull')
        # plot the pulls
        ax1.bar(bin_centers, pullA, width=his[1][1]-his[1][0], edgecolor="white", linewidth=1)
        # plot the lines at 0, +/- 3
        ax1.plot(ax1.get_xlim(),[0,0],'k', lw=1)
        ax1.plot(ax1.get_xlim(),[3,3]  ,'r', lw=1)
        ax1.plot(ax1.get_xlim(),[-3,-3],'r', lw=1)
        # remove x-axis labels from ax1 (residuals plot)
        ax1.set_xticklabels([])
        # add legend
        ax2.legend()
        # add chi2 with automatic positioning
        ax2.annotate(r'$\chi^2/n_{dof}$ = '+f'{np.square(pullA).sum():.1f}/{his[1].size-2}', 
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=24, ha='right', va='top')
        if fig is None:
            return (ax1, ax2)
        return fig, (ax1, ax2)
    

    # Draw a legend
    def getLegend( self, elements, name = 'cLegend', xlow = 0.6, ylow = 0.4, xmax = 0.95, ymax = 0.95 ):
        """Draw a legend

        Args:
            elements (dict): A dictionary of the elements to be added to the legend in this format `{ name: {index, color, style} }`.
            name (str, optional): The name of the r.TPad containing the legend. Defaults to 'cLegend'.
            xlow (float, optional): Minimum x (NDC) of the legend box. Defaults to 0.6.
            ylow (float, optional): Minimum y (NDC) of the legend box. Defaults to 0.4.
            xmax (float, optional): Maximum x (NDC) of the legend box. Defaults to 0.95.
            ymax (float, optional): Maximum y (NDC) of the legend box. Defaults to 0.95.

        Returns:
            (dict): a dictionary with the r.TPad (`legend`) containing the legend and the extra objects created (`extra`).
        """
        # elements = { name: {position, color, style} }
        #  r.TPad
        # Create a r.TPad containing the legend
        leg = r.TPad(name,name,xlow,ylow,xmax,ymax)
        leg.Draw()
        leg.cd()
        leg.SetFillStyle(4000)
        # for each element draw a box and write some text in the order specified by element's order
        t = r.TLatex()
        t.SetTextAlign(12)
        t.SetTextSize(0.08)
        # n+1: step = 1/n
        n = len(elements)
        boxes = {}
        for var, args in elements.items():
            boxes.update( { var+'_border': r.TBox(0.1, 1./n*(0.1+args['Index']), 0.28, 1./n*(0.9+args['Index'])) } )
            boxes.update( { var: r.TBox(0.105, 1./n*(0.1+args['Index'])+0.01, 0.275, 1./n*(0.9+args['Index'])-0.01) } )
            boxes[var+'_border'].SetFillColor(r.kBlack)
            if 'FillColor' in args.keys() is not None: boxes[var].SetFillColor(args['FillColor'])
            if 'LineColor' in args.keys() is not None: boxes[var].SetLineColor(args['LineColor'])
            if 'FillStyle' in args.keys() is not None: boxes[var].SetFillStyle(args['FillStyle'])
            if 'LineStyle' in args.keys() is not None: boxes[var].SetLineStyle(args['LineStyle'])
            if 'DrawOption' in args.keys() is not None: boxes[var].SetDrawOption(args['DrawOption'])
            boxes[var+'_border'].Draw()
            boxes[var].Draw()
            t.DrawLatexNDC(0.3, 1./n*(0.5+args['Index']), args['Latex'] if 'Latex' in args.keys() else var)
        return dict(legend=leg, extra=[t,boxes])

    # Get Integrals and number of events
    def getIntegralsAndEvents(self, model, xmin, xmax, varName, sigName, bkgName, sigEvName, bkgEvName):
        """Get integrals and number of events for signal and background

        Args:
            model (RooAbsPdf): the fit model
            xmin (float): minimum x value
            xmax (float): maximum x value
            varName (str): name of the variable
            sigName (str): name of the signal component
            bkgName (str): name of the background component
            sigEvName (str): name of the signal event variable
            bkgEvName (str): name of the background event variable

        Returns:
            dict: a dictionary with the integrals and number of events for signal, background and total
        """
        # get signal and background distributions:
        sig = model.getComponents().find(sigName)
        bkg = model.getComponents().find(bkgName)
        # get number of events
        ns = model.getVariables().find(sigEvName)
        nb = model.getVariables().find(bkgEvName)
        # get main variable
        x = model.getVariables().find(varName)
        x.setRange("signal",xmin,xmax)
        # evaluate integrals
        sigFullInt = sig.createIntegral(r.RooArgSet(x),r.RooArgSet(x),"Full")
        bkgFullInt = bkg.createIntegral(r.RooArgSet(x),r.RooArgSet(x),"Full")
        totFullInt = model.createIntegral(r.RooArgSet(x),r.RooArgSet(x),"Full")
        sigInt = sig.createIntegral(r.RooArgSet(x),r.RooArgSet(x),"signal")
        bkgInt = bkg.createIntegral(r.RooArgSet(x),r.RooArgSet(x),"signal")
        totInt = model.createIntegral(r.RooArgSet(x),r.RooArgSet(x),"signal")
        # calculate significance and its error
        # first store number of events and error in defined signal region
        Sig, Bkg, Tot, SigErr, BkgErr, TotErr = 0., 0., 0., 0., 0., 0.
        Sig = sigInt.getVal()/sigFullInt.getVal() * ns.getVal()
        SigErr = sigInt.getVal()/sigFullInt.getVal() * ns.getError()
        Bkg = bkgInt.getVal()/bkgFullInt.getVal() * nb.getVal()
        BkgErr = bkgInt.getVal()/bkgFullInt.getVal() * nb.getError()
        Tot = totInt.getVal()/totFullInt.getVal() * (ns.getVal() + nb.getVal())
        TotErr = totInt.getVal()/totFullInt.getVal() * (math.sqrt( ns.getError() * ns.getError() + nb.getError() * nb.getError() ) )
        # store in a dictionary
        values = {'Signal': {'value':Sig, 'error':SigErr,'integral': sigInt.getVal()},
                  'Background': {'value':Bkg, 'error':BkgErr,'integral': bkgInt.getVal()},
                  'Total': {'value':Tot, 'error':TotErr,'integral': totInt.getVal()} }
        return values

    # Integrate a pdf within two values
    def getIntegral(self, pdf, x, xmin, xmax, rangeName='intRange'):
        """Get the integral of a pdf over a specified range

        Args:
            pdf (RooAbsPdf): the pdf to integrate
            x (RooRealVar): the variable to integrate over
            xmin (float): the minimum x value
            xmax (float): the maximum x value
            rangeName (str, optional): the name of the integration range. Defaults to 'intRange'.

        Returns:
            float: the value of the integral
        """
        x.setRange(rangeName,xmin,xmax)
        ival = pdf.createIntegral(r.RooArgSet(x),r.RooArgSet(x),rangeName).getVal()
        return ival

    # Get Integrals and number of events
    def getIntegralsAndEventsBis(self, data, model, xmin, xmax, varName, pdfNames = [{'name': '', 'nevName': ''}]):

        pdfs, nevVars = {}, {}
        for pdf in pdfNames:
          # get signal and background distributions:
          pdfs[pdf['name']] = model.getComponents().find(pdf['name'])
          # get number of events
          nevVars[pdf['name']] = model.getVariables().find(pdf['nevName'])

        # get main variable
        x = model.getVariables().find(varName)
        x.setRange("Signal",xmin,xmax)

        # evaluate integrals
        integrals = {}
        for pdf in pdfNames:
          pname = pdf['name']
          integrals[pname] = { 'Full': pdfs[pname].createIntegral(r.RooArgSet(x),r.RooArgSet(x),"Full"),
                               'Signal': pdfs[pname].createIntegral(r.RooArgSet(x),r.RooArgSet(x),"Signal") }
        integrals['total'] = { 'Full': model.createIntegral(r.RooArgSet(x),r.RooArgSet(x),"Full"),
                               'Signal': model.createIntegral(r.RooArgSet(x),r.RooArgSet(x),"Signal") }

        # calculate significance and its error
        # first store number of events and error in defined signal region
        Nevs = {}
        Nevs['total'] = {'Full': {'val': data.sumEntries(), 'err': math.sqrt(data.sumEntries())},
                         'Signal': {'val': integrals['total']['Signal'].getVal()/integrals['total']['Full'].getVal()*data.sumEntries(),
                                    'err': integrals['total']['Signal'].getVal()/integrals['total']['Full'].getVal()*sqrt(data.sumEntries())} }
        for pdf in pdfNames:
          pname = pdf['name']
          Nevs[pname] = {'Full': {'val': nevVars[pname].getVal(),
                                  'err': nevVars[pname].getError()},
                         'Signal': {'val': integrals[pname]['Signal'].getVal()/integrals[pname]['Full'].getVal()*nevVars[pname].getVal(),
                                    'err': integrals[pname]['Signal'].getVal()/integrals[pname]['Full'].getVal()*nevVars[pname].getError()} }

        return { 'Integrals': integrals, 'Nevs': Nevs }



    # Significance
    def getSignificance(self, model, xmin, xmax, varName, sigName, bkgName, sigEvName, bkgEvName):
        """Get the significance of the signal, defined as $$\\frac{S}{\\sigma_B}$$ where $$\\sigma_B$$ is the error in the total number of events

        Args:
            model (RooAbsPdf): the model
            xmin (float): the minimum of the variable in the required range
            xmax (float): the maximum of the variable in the required range
            varName (str): the variable name
            sigName (str): the signal PDF name
            bkgName (str): the background PDF name
            sigEvName (str): the signal yield variable name
            bkgEvName (str): the background yield variable name

        Returns:
            RooRealVar: the significance stored in a RooRealVar
        """        

        # Get Number Events from fit
        events =  self.getIntegralsAndEvents(model, xmin, xmax, varName, sigName, bkgName, sigEvName, bkgEvName)

        # then evaluate significance
        S =  events['Signal']['value']/ events['Total']['error']
        Serr = events['Signal']['error']/ events['Total']['error']

        # store in r.RooRealVar
        significance = r.RooRealVar("significance", "significance",0.,1.e4)
        significance.setVal(S)
        significance.setError(Serr)
        return significance

    # Purity
    def getPurity(self, model, xmin, xmax, varName, sigName, bkgName, sigEvName, bkgEvName):
        """Get the purity of the signal, defined as $$\\frac{S}{S+B}$$

        Args:
            model (RooAbsPdf): the model
            xmin (float): the minimum of the variable in the required range
            xmax (float): the maximum of the variable in the required range
            varName (str): the variable name
            sigName (str): the signal PDF name
            bkgName (str): the background PDF name
            sigEvName (str): the signal yield variable name
            bkgEvName (str): the background yield variable name
        
        Returns:
            RooRealVar: the purity stored in a RooRealVar
        """

        # Get Number Events from fit
        events =  self.getIntegralsAndEvents(model, xmin, xmax, varName, sigName, bkgName, sigEvName, bkgEvName)

        # Evaluate purity
        Sig, SigErr, Bkg, BkgErr = events['Signal']['value'], events['Signal']['error'], events['Background']['value'], events['Background']['error']
#        print(Sig, SigErr, Bkg, BkgErr)
        P =  Sig/ (Sig+ Bkg )
#        Perr = 1./(Sig+Bkg) * sqrt( SigErr*SigErr + (Sig*Sig/(Sig+Bkg)/(Sig+Bkg))*(SigErr*SigErr + BkgErr*BkgErr) );
        Perr = 1./pow(Sig+Bkg,2) * math.sqrt( SigErr*SigErr*Bkg*Bkg + Sig*Sig*BkgErr*BkgErr)

        # store in r.RooRealVar
        purity = r.RooRealVar("purity", "purity",0.,1.)
        purity.setVal(P)
        purity.setError(Perr)
        return purity

    def fitToData(self, model, data, roofit_options=None, outname="",
                    kNLL=True, kChi=False, kMinos=False, kSilent=False ):
        """Fit the model to the data

        Args:
            model (RooAbsPdf): the model to fit
            data (RooDataSet): the data to fit
            roofit_options (list, optional): options for the fit. Defaults to r.RooLinkedList().
            outname (str, optional): the filename where to store the fit results. Defaults to "".
            kNLL (bool, optional): whether to use the negative log-likelihood for the fit. Defaults to True.
            kChi (bool, optional): whether to use the chi-squared for the fit. Defaults to False.
            kMinos (bool, optional): whether to use the MINOS algorithm for the fit. Defaults to False.
            kSilent (bool, optional): whether to run the fit in silent mode. Defaults to False.

        Raises:
            e: exception 

        Returns:
            RooFitResult: the result of the fit
        """        

        optList = r.RooLinkedList()
        if roofit_options is not None:
            for opt in roofit_options:
                optList.Add(opt)

        if not(kNLL or kChi): return 0

        res = r.RooFitResult()
        timer = r.TStopwatch()
        
        try:
            # Timer
            timer.Start()

            # Define the fit variables
            fitStatus = -1

            # Minimization
            # Set error type
            optList.Add(r.RooFit.Save())
            if kSilent:
                print('Running fit in "Silent" mode')
                r.RooMsgService.instance().setGlobalKillBelow(r.RooFit.ERROR)
                optList.Add(r.RooFit.PrintLevel(-1))
            
            res = model.chi2FitTo(data,optList) if kChi else model.fitTo(data,optList)

            # Check the status of the fit and perform again if needed
            covQual = res.covQual()
            fitStatus = res.status()
            nFit=1
            while (fitStatus or covQual!=3) and nFit<20:
                res = model.chi2FitTo(data,optList) if kChi else model.fitTo(data,optList)
                covQual = res.covQual()
                fitStatus = res.status()
                nFit = nFit+1

            # run the fit one more time to check stability
            res = model.chi2FitTo(data,optList) if kChi else model.fitTo(data,optList)
            covQual = res.covQual()
            fitStatus = res.status()
            nFit = nFit+1

            # Keep searching if fit not stable
            if (fitStatus or covQual!=3):
                while (fitStatus or covQual!=3) and nFit<20:
                    res = model.chi2FitTo(data,optList) if kChi else model.fitTo(data,optList)
                    covQual = res.covQual()
                    fitStatus = res.status()
                    nFit = nFit+1
            # Then we stop searching...

            # Run MINOS if required
            if fitStatus == 0 and covQual==3 and kMinos:
                optList.Add(r.RooFit.Minos(True))
                res = model.chi2FitTo(data,optList) if kChi else model.fitTo(data,optList)
            
            # Stop the timer
            timer.Stop()

            # Print and Store results
            # save to file
            if len(outname):
                self.SaveFitResults(res,outname+'.res',True)
                print("Fit results saved in %s.res\n\n" % outname)

            if not kSilent:
                print("Fit Result")
                res.Print("V")
                # Minuit Exit Status
                print("Fit status =", fitStatus)
                print("Minuit status =", res.covQual())

                # Fit Informations
                print("Number of Loops =", nFit)
                print("Real Time  =", timer.RealTime())
                print("CPU Time  =", timer.CpuTime())

            return res

        except Exception as e:
            # Cleanup in case of exception
            timer.Stop()
            if kSilent: 
                r.RooMsgService.instance().reset()
            raise e
        
        finally:
            # Always reset RooMsgService and clear optList if needed
            if kSilent: 
                r.RooMsgService.instance().reset()
            
            # Clear the options list to prevent memory accumulation
            if optList:
                optList.Clear()

    def Migrad(self, minuit):
        """Run migrad on the given minuit object

        Args:
            minuit (RooMinimizer): the minuit object to run migrad on

        Returns:
            (r.RooFitResult): the output of migrad
        """
        minuit.migrad()
        minuit.hesse()
        return minuit.save()

    def Fit(self, model, data,
            kChi=False, kExtended=False, kMinos=False,
            nthreads=1, range=0, poissonError=False,
            fname=None, kSilent=False):
        """
        A function to run fits using RooMinimizer. Likelihood by default.
        
        Args:
            model (RooAbsPdf): the model to fit
            data (RooAbsData): the data to fit
            kChi (bool, optional): whether to use chi-squared fit. Defaults to False.
            kExtended (bool, optional): whether to use extended maximum likelihood. Defaults to False.
            kMinos (bool, optional): whether to run MINOS algorithm. Defaults to False.
            nthreads (int, optional): number of CPU threads (currently not used). Defaults to 1.
            range (int, optional): fit range (currently not used). Defaults to 0.
            poissonError (bool, optional): whether to use Poisson errors. Defaults to False.
            fname (str, optional): name for the fit function. Defaults to None.
            kSilent (bool, optional): whether to run in silent mode. Defaults to False.
            
        Raises:
            Exception: any exception that occurs during fitting
            
        Returns:
            RooFitResult: the result of the fit, or None if chi2 fit requested with wrong data type
        """
        
        timer = r.TStopwatch()
        fun = None
        m = None
        
        try:
            # Check if data is weighted
            kWeighted = data.isWeighted()
            if kWeighted: print('Data ',data.GetName(),' is weighted!')
            if poissonError: kWeighted = False
            
            # Set error type (currently not used but kept for future enhancement)
            # errType = r.RooAbsData.SumW2 if kWeighted else r.RooDataHist.Poisson
            
            # Create fit function
            if kChi:
                if type(data) != r.RooDataHist:
                    print('chi2 fit option is possible only with r.RooDataHist data sets')
                    return None
                funname = 'chi2' if fname is None else fname
                fun = r.RooChi2Var(funname, funname, model, data, kExtended, r.RooFit.PrintLevel(-1)) if kSilent else \
                      r.RooChi2Var(funname, funname, model, data, kExtended)
            else:
                fun = r.RooNLLVar("nll", "nll", model, data, kExtended, r.RooFit.PrintLevel(-1)) if kSilent else \
                      r.RooNLLVar("nll", "nll", model, data, kExtended)
            
            # Start timer
            timer.Start()
            
            # Setup minuit
            m = r.RooMinimizer(fun)
            if kSilent:
                print('Running fit in "Silent" mode')
                r.RooMsgService.instance().setGlobalKillBelow(r.RooFit.ERROR)
            
            # Run fit
            res = self.Migrad(m)
            
            # Check the status of the fit and perform again if needed
            goodFit = res.status() == 0 and res.covQual() == 3
            nFit = 1
            while not goodFit and nFit < 20:
                res = self.Migrad(m)
                goodFit = res.status() == 0 and res.covQual() == 3
                nFit = nFit + 1

            # Run the fit one more time to check stability
            res = self.Migrad(m)
            goodFit = res.status() == 0 and res.covQual() == 3
            nFit = nFit + 1

            # Keep searching if fit not stable
            if not goodFit:
                while not goodFit and nFit < 20:
                    res = self.Migrad(m)
                    goodFit = res.status() == 0 and res.covQual() == 3
                    nFit = nFit + 1
            # Then we stop searching...

            # Run MINOS if required
            if goodFit and kMinos:
                m.minos()
                res = m.save()

            # Stop the timer
            timer.Stop()

            if not kSilent:
                print("Fit Result")
                res.Print("V")
                # Minuit Exit Status
                print("Fit status =", res.status())
                print("CovQuality =", res.covQual())

                # Fit Informations
                print("Number of Loops =", nFit)
                print("Real Time  = {:.2f}s".format(timer.RealTime()))
                print("CPU Time   = {:.2f}s".format(timer.CpuTime()))

            return res

        except Exception as e:
            # Cleanup in case of exception
            timer.Stop()
            if kSilent:
                r.RooMsgService.instance().reset()
            raise e
        
        finally:
            # Always reset RooMsgService if it was modified
            if kSilent:
                r.RooMsgService.instance().reset()
            
            # Clean up objects to prevent memory leaks
            # Note: ROOT objects are managed by ROOT's memory management,
            # but we ensure proper cleanup of references

    def AddFitInfo(self, fname, res):
        """Add fit information to the fit results file

        Args:
            fname (str): the file name
            res (r.RooFitResult): the fit results object
        """
        f = open(fname,'a')
        f.write("#-- Fit Info --#\n")
        #f.write("Iterations: "+str(nFit)+"\n")
        f.write("Fit Status: "+str(res.status())+"\n")
        f.write("CovQuality: "+str(res.covQual())+" (3 is Good)\n")
        f.write(f"EDM: {res.edm():.2e}\n")
        f.close()
        return

    def fitToDataAlt(self, model, data, outname="",
                    kNLL=True, kChi=False, kMinos=False, kExtended = False,
                    nthreads=1, range=None, poissonError = False, constraints=None,
                    kSilent=False, kAsymptotic=False ):
        """Fit the model to the data (alternative implementation with enhanced options)

        Args:
            model (RooAbsPdf): the model to fit
            data (RooDataSet): the data to fit
            outname (str, optional): the filename where to store the fit results. Defaults to "".
            kNLL (bool, optional): whether to use the negative log-likelihood for the fit. Defaults to True.
            kChi (bool, optional): whether to use the chi-squared for the fit. Defaults to False.
            kMinos (bool, optional): whether to use the MINOS algorithm for the fit. Defaults to False.
            kExtended (bool, optional): whether to use extended maximum likelihood. Defaults to False.
            nthreads (int, optional): number of CPU threads for parallel processing. Defaults to 1.
            range (str, optional): fit range specification. Defaults to None.
            poissonError (bool, optional): whether to use Poisson error handling. Defaults to False.
            constraints (RooArgSet, optional): constraints to apply during fitting. Defaults to empty RooArgSet.
            kSilent (bool, optional): whether to run the fit in silent mode. Defaults to False.
            kAsymptotic (bool, optional): whether to use asymptotic error calculation. Defaults to False.

        Raises:
            Exception: any exception that occurs during fitting

        Returns:
            RooFitResult: the result of the fit
        """

        if not(kNLL or kChi): return 0

        res = r.RooFitResult()
        timer = r.TStopwatch()
        optionsList = None

        if constraints is None:
            constraints = r.RooArgSet()

        try:
            # Check if data is weighted
            kWeighted = data.isWeighted()
            if kWeighted: print('Data ',data.GetName(),' is weighted!')
            if poissonError: kWeighted = False

            # Timer
            timer.Start()

            # Define the fit variables
            fitStatus = -1

            # Minimization
            if kSilent:
                print('Running fit in "Silent" mode')
                r.RooMsgService.instance().setGlobalKillBelow(r.RooFit.ERROR)

            # Set error type and build options list
            optionsList = r.RooLinkedList()
            if kWeighted: optionsList.Add(r.RooFit.AsymptoticError(True) if kAsymptotic else r.RooFit.SumW2Error(kWeighted))
            optionsList.Add(r.RooFit.Extended(kExtended))
            optionsList.Add(r.RooFit.NumCPU(nthreads))
            optionsList.Add(r.RooFit.Constrain(constraints))
            optionsList.Add(r.RooFit.Save())
            if range is not None: optionsList.Add(r.RooFit.Range(range))
            if kSilent: optionsList.Add(r.RooFit.PrintLevel(-1))
            
            if not kSilent: optionsList.Print("V")
            res = model.chi2FitTo(data,optionsList) if kChi else model.fitTo(data,optionsList)

            # Check the status of the fit and perform again if needed
            fitStatus = res.status()
            nFit=1
            while fitStatus and nFit<20:
                res = model.chi2FitTo(data,optionsList) if kChi else model.fitTo(data,optionsList)
                fitStatus = res.status()
                nFit = nFit+1

            # run the fit one more time to check stability
            res = model.chi2FitTo(data,optionsList) if kChi else model.fitTo(data,optionsList)
            fitStatus = res.status()
            nFit = nFit+1

            # Keep searching if fit not stable
            if fitStatus:
                while fitStatus and nFit<20:
                    res = model.chi2FitTo(data,optionsList) if kChi else model.fitTo(data,optionsList)
                    fitStatus = res.status()
                    nFit = nFit+1
            # Then we stop searching...

            # Run MINOS if required
            if fitStatus == 0 and kMinos:
                optionsList.Add(r.RooFit.Minos(True))
                res = model.chi2FitTo(data,optionsList) if kChi else model.fitTo(data,optionsList)
            
            # Stop the timer
            timer.Stop()

            # Print and Store results
            # save to file
            if len(outname):
                self.SaveFitResults(res,outname+'.res',True)
                print("Fit results saved in %s.res\n\n" % outname)

            if not kSilent:
                print("Fit Result")
                res.Print("V")
                # Minuit Exit Status
                print("Fit status =", fitStatus)

                # Fit Informations
                print("Number of Loops =", nFit)
                print("Real Time  =", timer.RealTime())
                print("CPU Time  =", timer.CpuTime())

            return res

        except Exception as e:
            # Cleanup in case of exception
            timer.Stop()
            if kSilent: 
                r.RooMsgService.instance().reset()
            raise e
        
        finally:
            # Always reset RooMsgService and clear optionsList if needed
            if kSilent: 
                r.RooMsgService.instance().reset()
            
            # Clear the options list to prevent memory accumulation
            if optionsList:
                optionsList.Clear()
    

    def PlotResiduals2D(self, model, data, obsNames=[], cname='2Dresiduals', title='residuals2D', tagLeft = '', tagRight = '', xbins=50, ybins=50):
        """Plot the fit residuals in a 2D histogram

        Args:
            model (RooAbsPdf): the model used in the fit
            data (RooAbsData): the data used in the fit
            obsNames (list, optional): the list of observables to project (should be 2). Defaults to [].
            cname (str, optional): name of the r.TCanvas. Defaults to '2Dresiduals'.
            title (str, optional): title of the r.TCanvas. Defaults to 'residuals2D'.
            tagLeft (str, optional): text to be added on the top left of the canvas. Defaults to ''.
            tagRight (str, optional): text to be added on the top right of the canvas. Defaults to ''.
            xbins (int, optional): number of bins on the x axis. Defaults to 50.
            ybins (int, optional): number of bins on the y axis. Defaults to 50.

        Returns:
            dict(canvas, extra): a canvas and a list of extra objects to be added to the canvas
        """
        # Check the number of variables
        if len(obsNames)!=2:
            print('Could not plot 2D residuals. The number of projection variables is not 2', obsNames)
            return 0

        # Check data has the observables
        obs = r.RooArgSet()
        obsList = r.RooArgList()
        dataObs = data.get()
        for name in obsNames:
            if dataObs.find(name) is not None:
                obs.add(dataObs.find(name))
                obsList.add(dataObs.find(name))
        if len(obs)!=2:
            print('Could not plot 2D residuals. The number of projection variables found in dataset is not 2', obsNames)
            return 0

        # Create an histogram out of the data
        x, y = obs.find(obsNames[0]), obs.find(obsNames[1])
        cut = '%.2f < %s && %s < %.2f && %.2f < %s && %s < %.2f' % (x.getMin(), x.GetName(), x.GetName(), x.getMax(),
                                                                    y.getMin(), y.GetName(), y.GetName(), y.getMax())
        hData = data.createHistogram(x, y, xbins, ybins, cut, "hData_"+data.GetName())

        # Create an histogram out of the model and scale it to data
        hModel = hData.Clone("hModel_"+model.GetName())
        hModel.Scale(0.)
        model.fillHistogram(hModel, obsList, data.numEntries())

        # Build residuals histogram
        hRes = hData.Clone('hRes_'+cname)
        hRes.Sumw2()
        hDist = r.TH1D('hDist_'+cname,'hDist_'+cname,21,-5,5)
        hDist.GetXaxis().SetTitle('Residuals')
        for x in range(1,xbins+1):
            for y in range(1,ybins+1):
                err= 1 if hData.GetBinError(x,y) == 0 else hData.GetBinError(x,y)
                #                    val = (hData.GetBinContent(x,y) - hModel.GetBinContent(x,y) )/ hData.GetBinError(x,y)
                val = (hData.GetBinContent(x,y) - hModel.GetBinContent(x,y) )/ err
                hDist.Fill(val)
                hRes.SetBinContent( x,y, val )
                hRes.SetBinError(x,y,0)
                #                print(hModel.GetBinContent(x,y), hModel.GetBinError(x,y), hData.GetBinContent(x,y), hData.GetBinError(x,y))
        hRes.SetMaximum(5)
        hRes.SetMinimum(-5)

        lat = r.TLatex()
        lat.SetTextSize(0.06)
        lat.SetTextAlign(13)

        can = {'canvas': r.TCanvas(cname, title, 800, 400), 'extra':[]}
        can['canvas'].Divide(2)
        can['canvas'].cd(1)
        can['canvas'].GetPad(1).SetRightMargin(0.1)
        hRes.Draw('col4z')
        lat.DrawLatexNDC(0.2,0.95,tagLeft)
        lat.DrawLatexNDC(0.7,0.95,tagRight)
        can['canvas'].cd(2)
        hDist.Draw()
        lat.DrawLatexNDC(0.2,0.95,tagLeft)
        lat.DrawLatexNDC(0.75,0.95,tagRight)
        can['canvas'].Update()
        can['extra']+=[hRes, hDist, hData, hModel, lat]

        return can
    
    def DataHistFromNumpy(self, x_arr, observables):
        """Build a RooDataHist from numpy data (1D or multi-dimensional).

        Args:
            x_arr (array): Input data. Supported formats:
                - 1D array with shape (n_samples,)
                - 2D array with shape (n_samples, n_observables)
                - 2D array with shape (n_observables, n_samples) (auto-transposed)
            observables (RooRealVar, RooArgSet, RooArgList, list): observables used to define binning/ranges.

        Returns:
            RooDataHist: the RooDataHist object
        """
        # Normalise observables to a python list first
        if hasattr(observables, "InheritsFrom") and observables.InheritsFrom("RooAbsArg"):
            obs_list = [observables]
        else:
            obs_list = [obs for obs in observables]

        if len(obs_list) == 0:
            raise ValueError("observables cannot be empty")

        # Convert inputs to numpy and align shape to (n_samples, n_observables)
        arr = np.asarray(x_arr)
        ndim_obs = len(obs_list)

        if arr.ndim == 1:
            if ndim_obs != 1:
                raise ValueError(f"1D input is compatible only with 1 observable, got {ndim_obs}")
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            if arr.shape[1] != ndim_obs:
                # Accept common transposed layout: (n_observables, n_samples)
                if arr.shape[0] == ndim_obs:
                    arr = arr.T
                else:
                    raise ValueError(
                        f"Input shape {arr.shape} is incompatible with {ndim_obs} observables. "
                        "Expected (n_samples, n_observables) or (n_observables, n_samples)."
                    )
        else:
            raise ValueError(f"x_arr must be 1D or 2D, got {arr.ndim}D")

        # Build bins/ranges from RooRealVar definitions
        bins = [obs.getBins() for obs in obs_list]
        ranges = [(obs.getMin(), obs.getMax()) for obs in obs_list]

        nphist, edges = np.histogramdd(arr, bins=bins, range=ranges)

        # Build a RooArgSet expected by RooDataHist.from_numpy
        obs_set = r.RooArgSet()
        for obs in obs_list:
            obs_set.add(obs)

        hist_ranges = [(edge[0], edge[-1]) for edge in edges]
        data = r.RooDataHist.from_numpy(nphist, obs_set, bins=bins, ranges=hist_ranges)
        return data
    
    def DatasetFromNumpy(self, x_arr, obs):
        """Build a RooDataSet from a numpy array

        Args:
            x_arr (array): the numpy array with the data
            obs (RooRealVar): the observable

        Returns:
            RooDataSet: the RooDataSet object
        """        
        data = r.RooDataSet.from_numpy({obs.GetName(): x_arr}, [obs])
        return data

    def CombineDatasets(self, name, title, variables, category, dsets):
        """Combine datasets for simultaneous fitting

        Args:
            name (str): name of the combined dataset
            title (str): description of the combined dataset
            variables (list): list of variables to include in the combined dataset
            category (RooCategory): the RooCategory object for associating the dataset to the model
            dsets (dict): a dictionary in the form {key: r.RooDataSet/RooDataHist} of the datasets to combine

        Returns:
            (RooDataSet/RooDataHist): the combined dataset
        """
        # check the datasets are all the same type
        dset_type = type(dsets[list(dsets.keys())[0]])
        if not all(type(dsets[key]) is dset_type for key in dsets.keys()):
            raise ValueError("All datasets must be of the same type")
        # crate a list with the variables to include in the combined dataset
        #vset = r.RooArgSet()
        vlist = r.RooArgList()
        for v in variables:
            #vset.add(v)
            vlist.add(v)
        # create the combined dataset
        dataType = str(type(dsets[list(dsets.keys())[0]]))
        data = copy.copy(dsets[list(dsets.keys())[0]])
        data.SetName(name)
        data.SetTitle(title)
        data.reset()
        # fill the combined datasets
        for key, args in dsets.items():
            if 'RooDataSet' in dataType:
                data.add(r.RooDataSet(name+key, title+key, vlist, r.RooFit.Index(category), r.RooFit.Import(key,args)))
                print('added RooDataSet', name+key, ': (',key,',',args,')')
            else:
                data.add(r.RooDataHist(name+key, title+key, vlist, r.RooFit.Index(category), r.RooFit.Import(key,args)))
                print('added RooDataHist', name+key, ': (',key,',',args,')')
        return data

    def SaveFitResults(self, res, oname, verbose = False):
        """A function to save the fit results to a file in a verbose format

        Args:
            res (r.RooFitResult): the output of the fit
            oname (str): the name of the output file
            verbose (bool, optional): True if more detailed, False otherwise. Defaults to False.
        """
        covQualOutput = { -1  : "Unknown, matrix was externally provided",
                           0  : "Not calculated at all",
                           1  : "Approximation only, not accurate",
                           2  : "Full matrix, but forced positive-definite",
                           3  : "Full, accurate covariance matrix" }

        out = open(oname,'w')
        out.write('\n')
        out.write("\t  r.RooFitResult: minimized FCN value: %f, estimated distance to minimum: %f\n" \
                  "\t                covariance matrix quality: %s\n" % (res.minNll(), res.edm(), covQualOutput[res.covQual()]) )
        out.write("\t                Status : ")
        for icycle in range(res.numStatusHistory()):
            out.write("%s = %d " % (res.statusLabelHistory(icycle), res.statusCodeHistory(icycle) ))
        out.write("\n\n")

        if verbose:
          if res.constPars().getSize()>0:
            out.write("\t    Constant Parameter    Value     \n"+
                      "\t  --------------------  ------------\n")

          for i in range(res.constPars().getSize()):
            out.write("\t  "+"{:>20s}".format(res.constPars().at(i).GetName())+
                      "  "+"{:12s}".format("%12.4e" % res.constPars().at(i).getVal())+"\n")
          out.write("\n")

          # Has any parameter asymmetric errors?
          doAsymErr = False
          for i in range(res.floatParsFinal().getSize()):
            if res.floatParsFinal().at(i).hasAsymError():
              doAsymErr=True
              break

          if doAsymErr:
            out.write("\t    Floating Parameter  InitialValue    FinalValue (+HiError,-LoError)    GblCorr.\n"+
                      "\t  --------------------  ------------  ----------------------------------  --------\n")
          else:
            out.write("\t    Floating Parameter  InitialValue    FinalValue +/-  Error     GblCorr.\n"+
                      "\t  --------------------  ------------  --------------------------  --------\n")

          for i in range(res.floatParsFinal().getSize()):
            out.write("\t  "+"{:>20s}".format(res.floatParsFinal().at(i).GetName()))
            out.write("\t  "+"{:12s}".format("%12.4e" % res.floatParsInit().at(i).getVal())+
                      "\t  "+"{:12s}".format("%12.4e" % res.floatParsFinal().at(i).getVal()) )

            if res.floatParsFinal().at(i).hasAsymError():
              out.write(+"{:21s}".format("+%8.2e, -%8.2e" % (res.floatParsFinal().at(i).getAsymErrorHi(), -1*res.floatParsFinal().at(i).getAsymErrorLo())))
            else:
              err = res.floatParsFinal().at(i).getError()
              out.write( ("        " if doAsymErr else "")+" +/- "+"{:9s}".format("%9.2e" % err))

            if res.globalCorr():
              out.write("  "+"{:8s}".format("%8.6f" % res.globalCorr(res.floatParsFinal().at(i).GetName())))
            else:
              out.write("  <none>")
            
            out.write("\n")  # Add newline after each parameter

          out.write("\n")

        else:
          out.write("\t    Floating Parameter    FinalValue +/-  Error   \n"+
                    "\t  --------------------  --------------------------\n")

          for i in range(res.floatParsFinal().getSize()):
            err = res.floatParsFinal().at(i).getError()
            out.write("\t  "+"{:>20s}".format(res.floatParsFinal().at(i).GetName())+
                      "  "+"{:12s}".format("%12.4e" % res.floatParsFinal().at(i).getVal())+
                      " +/- "+"{:9s}".format("%9.2e" % err)+"\n")

        out.write("\n")
        out.close()

    def FixVariables(self, ws, pdf, fix = True, verbose = True):
        """Fix or release all the parameters of a PDF in a RooWorkspace

        Args:
            ws (RooWorkspace): the workspace
            pdf (RooAbsPdf): the pdf to fix/release the parameters of
            fix (bool, optional): True to fix, False to release. Defaults to True.
        """
        # pars= ws.pdf(pdf).getVariables()
        pars= ws.pdf(pdf).getVariables()
        pars.Print()
        for par in pars:
            if par.GetName() == 'sample':
                continue
            ws.var(par.GetName()).setConstant(fix)
            if verbose: print('variable %s set to constant' % par)
        return

    def GetVarsList(self, ws, vlist):
        """Create a list of r.RooRealVariables from a RooWorkspace given a list of strings

        Args:
            ws (RooWorkspace): the workspace
            vlist (list): the list of variables

        Returns:
            (RooArgList): the list of r.RooRealVariables
        """
        alist = r.RooArgList()
        allvars = ws.allVars()
        for v in vlist:
            if not ws.var(v) in allvars: raise ValueError(f'Variable {v} not found in workspace')
            alist.add(ws.var(v))
        return alist

    def GetVarsSet(self, ws, vlist):
        """Create a set of r.RooRealVariables from a RooWorkspace given a list of strings

        Args:
            ws (RooWorkspace): the workspace
            vlist (list): the list of variables

        Returns:
            (RooArgSet): the set of r.RooRealVariables
        """
        alist = r.RooArgSet()
        allvars = ws.allVars()
        for v in vlist:
            if not ws.var(v) in allvars: raise ValueError(f'Variable {v} not found in workspace')
            alist.add(ws.var(v))
        return alist

    def SetVarsPrintDigits(self, ws, ndig):
        """Set the number of digits for the precision with which the variables are printed

        Args:
            ws (RooWorkspace): the workspace
            ndig (int): the number of digits
        """
        vlist = ws.allVars()
        it = vlist.createIterator()
        var = it.Next()
        while var:
            var.printSigDigits(ndig)
            var = it.Next()
        return
    

    def Sweights(self, ws, obs, data, model_name, pdfs = [''], nevs = [''], verbose=False):
        """Calculate s-weights

        Args:
            ws (RooWorkspace): RooFit Workspace
            obs (str, optional): Name of the observable in the workspace. Defaults to 'x'.
            data (str, optional): Name of the dataset in the workspace. Defaults to 'data'.
            model_name (str, optional): Name of the model in the workspace. Defaults to 'model'.
            pdfs (list, optional): Names of the probability density functions in the workspace to fix for the calculation. Defaults to [''].
            nevs (list, optional): Names of the yields in the workspace to use for the calculation. Defaults to [''].

        Returns:
            dict(tree, vars): A dictionary of trees and variables
        """
        for pdf in pdfs: 
             self.FixVariables(ws, pdf, True, verbose)
        ws.var(obs).setConstant(False)
        # Re fitting
        r.RooMsgService.instance().setSilentMode(True)
        self.fitToData(ws.pdf(model_name), data, kNLL=True, kChi=False, kSilent=True)
        #ws.pdf(model_name).fitTo(data,r.RooFit.Extended(True))
        # Make splot!
        splot = r.RooStats.SPlot("sData","An SPlot", data, ws.pdf(model_name), self.GetVarsList(ws, nevs) )
        r.RooMsgService.instance().reset()
        # Declare a tree
        tVars = {}
        for nev in nevs: tVars['w_'+nev] = 'F'
        t = DefineTree(tVars, 'ntp_sweights', 'tree with s-weights')
        # Fill the tree
        for entry in range(0,data.numEntries()):
            Vars = data.get(entry)
            for nev in nevs:
                t['vars']['w_'+nev][0] = Vars.find(nev+'_sw').getVal()
            t['tree'].Fill()
        # Cleanup
        if splot:
            del splot
        return t

    def decodeResLine(self,line):
        """Given a line from the output of r.RooArgSet.writeToFile(), decode the line and return a dictionary of the variables

        Args:
            line (str): the line written to file

        Returns:
            (dict): a dictionary in the format `{'name': vname, 'val': value, 'err': err, 'attributes': attributes, 'range': vrng}`, 
            where `attributes` is a dictionary containing specific information on the variable (binning/unit/constant)
        """
        if not( '=' in line ): return None
        ll = line.split('=')
        vname = ll[0].replace(' ','')
        # variable range
        rreg = re.compile(r'L\((.*?)\)')#(r'L\([+-]?([0-9]*[.])?[0-9]+\\s+-\s+[+-]?([0-9]*[.])?[0-9]+\)')
        rfnd = rreg.search(line).group() if rreg.search(line) is not None else ''
        vrng = [float(x) for x in rfnd.split('L(')[1].split(')')[0].split(' - ')] if rfnd!='' else None
        attributes = {'constant': 'C' in ll[1]}
        if '//' in ll[1]: attributes['unit'] = ll[1].split('// ')[1].replace('[','').replace(']','')
        if not( '+/-' in ll[1] ):
            lv = ll[1].split(' ')
            while '' in lv: lv.remove('')
            value = float(lv[0])
            breg = re.compile(r'B\([0-9]*\)')
            bfnd = breg.search(ll[1])
            if bfnd: attributes['bins'] = int(bfnd.group()[2:-1])
            return {'name': vname, 'val': value, 'attributes': attributes, 'range': vrng}
        else:
            lv = ll[1].split('+/-')
            value = float(lv[0])
            areg = re.compile(r'\([+-]?([0-9]*[.])?[0-9]+\,\s+[+-]?([0-9]*[.])?[0-9]+\)')
            afnd = areg.search(lv[1]).group() if areg.search(lv[1]) is not None else ''
            if len(afnd): # asymmetric errors
                errs = [float(x) for x in afnd.replace('(','').replace(')','').split(',')]
                return {'name': vname, 'val': value, 'err': errs, 'attributes': attributes, 'range': vrng}
            else:
                le = lv[1].split(' ')
                while '' in le: le.remove('')
                err = float(le[0])
                return {'name': vname, 'val': value, 'err': err, 'attributes': attributes, 'range': vrng}
        return None

    def resultsToDictionary(self,resfile):
        """Decode the output of r.RooArgSet.writeToFile() and return a dictionary of all the variables

        Args:
            resfile (str): the name of the file containing the output of r.RooArgSet.writeToFile()

        Returns:
            (dict): A dictionary in the format `{'variables':{}, 'status':{}, 'constants':{}, 'observables':{}, 'blind': None}`
        """
        res_dict= {'variables':{}, 'status':{}, 'constants':{}, 'observables':{}, 'blind': None}
        f = open( resfile, 'r')
        for line in f:
            if 'Fit Info' in line: continue
            ldec = self.decodeResLine(line)
            if ldec is None:
                if ':' in line:
                    ll = line.split(':')
                    sname = ll[0].replace(' ','')
                    value = ll[1].split()[0]
                    if sname == 'BlindStr': res_dict['blind']=value
                    else: res_dict['status'].update({sname : value})
                continue
            else:
                if 'bins' in ldec['attributes'].keys():
                    res_dict['observables'].update({ldec['name']: ldec})
                elif 'err' not in ldec.keys() and not ldec['attributes']['constant']:
                    res_dict['observables'].update({ldec['name']: ldec})
                elif ldec['attributes']['constant']:
                    res_dict['constants'].update({ldec['name']: ldec})
                else:
                    res_dict['variables'].update({ldec['name']: ldec})
        return res_dict

    def check_variable(self, vinfo, z=2., verbose=False):
        """Check whether a fitted variable is (too) close to the boundaries

        Args:
            vinfo (dict): Variable information
            z (float, optional): Multiplier for the error. Defaults to 2..
            verbose (bool, optional): If True, print detailed information. Defaults to False.

        Returns:
            int: A code indicating the status of the variable
        """        
        check_boundary = min([abs(vinfo['val']-vinfo['range'][i])/(vinfo['range'][1]-vinfo['range'][0]) for i in range(2)]) < 0.01
        range_with_error = (vinfo['val'] - z*vinfo['err'], vinfo['val'] + z*vinfo['err'])
        check_range = range_with_error[0] < vinfo['range'][0] or range_with_error[1] > vinfo['range'][1]
        code = check_boundary*1 + check_range*2
        if verbose:
            if code == 1:
                print(f"Variable {vinfo['name']} is close to the boundary of its range.")
            elif code == 2:
                print(f"Variable {vinfo['name']} may go outside its range.")
            elif code == 3:
                print(f"Variable {vinfo['name']} is close to the boundary and may go outside its range.")
        return code

    def testVariable(self, vinfo, verbose=False):
        """Test whether a variable is in the range of the allowed values

        Args:
            vinfo (dict): Variable info dictionary
            verbose (bool, optional): verbosity. Defaults to False.

        Returns:
            str: a message if the variable is out of range, otherwise ''
        """
        if not ('err' in vinfo.keys()) or not ('range' in vinfo.keys()) or vinfo['range'] is None: return ''
        if type(vinfo['err'])==list:
            if vinfo['val']-3.*vinfo['err'][0] < vinfo['range'][0] or vinfo['val']+3.*vinfo['err'][1] > vinfo['range'][1]:
                msg = f"{vinfo['name']} may go outside ranges ({vinfo['val']}+-({vinfo['err'][0]},{vinfo['err'][1]}) [{vinfo['range'][0]},{vinfo['range'][1]}])\n"
                if verbose: print(msg)
                return msg
        elif vinfo['val']-3.*vinfo['err'] < vinfo['range'][0] or vinfo['val']+3.*vinfo['err'] > vinfo['range'][1]:
            msg = '{} may go outside ranges ({}+-{} [{},{}])\n'.format(vinfo['name'],vinfo['val'],vinfo['err'],vinfo['range'][0],vinfo['range'][1])
            if verbose: print(msg)
            return msg
        return ''

    def testVariables(self,res_dict,verbose=False):
        """Checks if the variables are within the ranges

        Args:
            res_dict (dict): a dictionary with the results of the fit
            verbose (bool, optional): verbosity. Defaults to False.

        Returns:
            bool: True if all variables are within the ranges, False otherwise
        """
        alerts= ''
        for var in res_dict['variables'].keys():
            vinfo = res_dict['variables'][var]
            alerts+= self.testVariable(vinfo,verbose)
        is_good = len(alerts) == 0
        if not is_good and verbose:
            print(alerts)
        return is_good

    def IsGoodFit(self,resfile,verbose=False):
        """Check if the fit is good

        Args:
            resfile (str): the name of the file containing the output of r.RooArgSet.writeToFile()
            verbose (bool, optional): verbosity. Defaults to False.

        Returns:
            (bool): True if the fit is good, False otherwise
        """
        if not os.path.exists(resfile): return False
        dd = self.resultsToDictionary(resfile)
        is_good = (dd['status']['FitStatus']=='0') and (dd['status']['CovQuality']=='3')
        if not is_good and verbose:
            if dd['status']['FitStatus']!='0':
                print('FitStatus is',dd['status']['FitStatus'])
            if dd['status']['CovQuality']!='3':
                print('CovQuality is',dd['status']['CovQuality'])
        return is_good

    def searchForFitProblems(self,resfile,verbose=False):
        """Searches for problems in the fit results file

        Args:
            resfile (str): the fit results file name
            verbose (bool, optional): verbosity. Defaults to False.

        Returns:
            bool: True if the fit is good, False otherwise
        """
        if not os.path.exists(resfile): return False
        if verbose: print('testing',resfile)
        rd = self.resultsToDictionary(resfile)
        is_good = self.testVariables(rd)
        is_good*= self.IsGoodFit(resfile,verbose)
        if (not is_good) and verbose: print('problem in',resfile)
        return is_good

    # def compareResults(self, res1, res2,
    #                    cname='fitres_comparison', norm=True, ignore=None,
    #                    legend=None, latexNames=None):
    #     """Compare the results of two fits and plot the differences

    #     Args:
    #         res1 (str): the name of the file containing the output of r.RooArgSet.writeToFile() for the first fit
    #         res2 (str): the name of the file containing the output of r.RooArgSet.writeToFile() for the second fit
    #         cname (str, optional): the name of the output r.TCanvas. Defaults to 'fitres_comparison'.
    #         norm (bool, optional): True to plot differences with respect to average, False otherwise. Defaults to True.
    #         ignore (list, optional): list of variables to ignore. Defaults to None.
    #         legend (list, optional): list of names of the fit results for the legend. Defaults to None.
    #         latexNames (dict, optional): a dictionary with the names of the variables and their latex counterpart. Defaults to None.

    #     Returns:
    #         (dict(canvas, extra)): a canvas with the comparison and a list of extra objects to be added to the canvas
    #     """
    #     # load fit results to dictionary
    #     d1, d2 = self.resultsToDictionary(res1), self.resultsToDictionary(res2)
    #     # check if they are compatible
    #     if not CompatibleDicts(d1,d2):
    #         print('Fit results have a different structure and wont be compared.')
    #         return 0
    #     # get x values
    #     vnames = []
    #     x1, x1err, x2, x2err = array('d',[]),array('d',[]),array('d',[]),array('d',[])
    #     for k in d1['variables'].keys():
    #         if ignore is not None and d1['variables'][k]['name'] in ignore: continue
    #         vn = d1['variables'][k]['name']
    #         vnames += [latexNames[vn] if (latexNames is not None and vn in latexNames.keys()) else vn]
    #         a1, a2 = d1['variables'][k], d2['variables'][k]
    #         am = float(a1['val']+a2['val'])/2
    #         #print(a1['val'], am, a1['val']-am, type(a1['val']-am))
    #         x1.append(a1['val']-am)
    #         x2.append(a2['val']-am)
    #         x1err.append(a1['err'])
    #         x2err.append(a2['err'])
    #         if norm:
    #             x1[-1]=x1[-1]/am
    #             x2[-1]=x2[-1]/am
    #             x1err[-1]=x1err[-1]/am
    #             x2err[-1]=x2err[-1]/am
    #     # determinare x range
    #     xmin, xmax = 0,0
    #     for i in range(len(x1)):
    #         xmin = min(x1[i]-x1err[i],x2[i]-x2err[i],xmin)
    #         xmax = max(x1[i]+x1err[i],x2[i]+x2err[i],xmax)
    #     xmin = xmin-0.1*(xmax-xmin)
    #     xmax = xmax+0.1*(xmax-xmin)
    #     # create frame
    #     c = Canvas(cname, 600, 800)
    #     c['canvas'].SetLeftMargin(0.2)
    #     c['canvas'].SetRightMargin(0.05)
    #     c['canvas'].SetTopMargin(0.05)
    #     hf= c['canvas'].DrawFrame(xmin,0.,xmax,1.)
    #     hf.SetXTitle('Fit Results vs Average')
    #     hf.SetYTitle('')
    #     hf.GetYaxis().SetLabelSize(0)
    #     hf.GetYaxis().SetNdivisions(0)
    #     # set y values and add variable names
    #     lat = r.TLatex()
    #     lat.SetTextSize(0.04)
    #     lat.SetTextAlign(32)
    #     xlat = xmin-0.02*(xmax-xmin)
    #     y1, y2, ye = array('d',[]), array('d',[]), array('d',[])
    #     for i in range(len(x1)):
    #         dy = 1./(len(x1)+1)
    #         y1.append((i+1.1)*dy)
    #         y2.append((i+0.9)*dy)
    #         ye.append(0)
    #         lat.DrawLatex(xlat, (i+1)*dy, vnames[i])
    #     # add line at 0
    #     lz= r.TLine(0,0,0,1.)
    #     lz.SetLineStyle(r.kDashed)
    #     lz.Draw()
    #     # create graphs
    #     gr1 = r.TGraphErrors(len(x1),x1,y1,x1err,ye)
    #     gr2 = r.TGraphErrors(len(x2),x2,y2,x2err,ye)
    #     gr2.SetLineColor(r.kRed)
    #     gr2.SetMarkerColor(r.kRed)
    #     gr1.Draw('P')
    #     gr2.Draw('P')
    #     # draw labels
    #     if legend is not None and len(legend)==2:
    #         lat.SetTextAlign(13)
    #         lat.SetTextSize(0.06)
    #         lat.DrawLatex(xmin+0.02*(xmax-xmin),0.98,legend[0])
    #         lat.SetTextAlign(33)
    #         lat.SetTextColor(r.kRed)
    #         lat.DrawLatex(xmax-0.02*(xmax-xmin),0.98,legend[1])
    #     c['extra']+=[hf,gr1,gr2,lat,lz]
    #     c['canvas'].Update()
    #     return c

    def GetGraphVariables(self, vv, yoffset = 1., offsets=None, norm=False):
        """Obtain a r.TGraphErrors object with the variables in vv

        Args:
            vv (dict): Variables in the form of a dictionary with the following structure: {'name':{'val':value,'err':error}}
            yoffset (float, optional): The offset to draw the graph on y (useful when plotting multiple instances). Defaults to 1..
            offsets (dict, optional): Offsets to apply for a fair comparison. Structure {'name':offset}. Defaults to None.
            norm (bool, optional): Normalise the graph. Defaults to False.

        Returns:
            r.TGraphErrors: A r.TGraphErrors object with the variables in vv
        """
        x, xe = array('d',[]),array('d',[])
        for v in vv.keys():
            x.append(vv[v]['val'] if offsets is None else vv[v]['val']-offsets[v])
            xe.append(vv[v]['err'])
            if norm: 
                x[-1]=x[-1]/(x[-1] if offsets is None else offsets[v])
                xe[-1]=xe[-1]/(xe[-1] if offsets is None else offsets[v])
        y, ye = array('d',[]), array('d',[])
        for i in range(len(x)):
            dy = 1./(len(x)+1)
            y.append((i+yoffset)*dy)
            ye.append(0)
        gr = r.TGraphErrors(len(x),x,y,xe,ye)
        return gr

    # def CompareFitResults(self, file_names, cname='fit_results_comparison', subset_vars=None,
    #                     ignore=None, legend=None,colors=hu.colors,xtitle='Fit Results',latexNames=None):
    #     """Compare the results of the same fit stored in different files

    #     Args:
    #         file_names (list): file names with the fit results (as written by r.RooArgSet.WriteToFile)
    #         cname (str, optional): name of the output r.TCanvas. Defaults to 'fit_results_comparison'.
    #         subset_vars (list, optional): list of variables to plot. Defaults to None.
    #         ignore (list, optional): list of variables to ignore. Defaults to None.
    #         legend (list, optional): legend labels. Defaults to None.
    #         colors (list, optional): user-defined colors. Defaults to hu.colors.
    #         xtitle (str, optional): x axis title. Defaults to 'Fit Results'.
    #         latexNames (dict, optional): dictionary of latex names for each variable. Defaults to None.

    #     Returns:
    #         dict(canvas, extra): a canvas and a list of extra objects to be added to the canvas
    #     """
    #     # decode results to dictionaries
    #     results = [self.resultsToDictionary(f) for f in file_names]
    #     # check if they are compatible
    #     for i in range(len(results)-2):
    #         if not CompatibleDicts(results[i],results[-1]):
    #             print('Fit results have a different structure and wont be compared.')
    #             return 0
    #     # get variables' names
    #     varList = []
    #     for v in results[0]['variables'].keys(): 
    #         if ignore is not None and results[0]['variables'][v]['name'] in ignore: continue
    #         if subset_vars is not None and v not in subset_vars: continue
    #         varList += [v]
    #     vnames = []
    #     for v in varList: 
    #         vn = results[0]['variables'][v]['name']
    #         vnames += [latexNames[vn] if (latexNames is not None and vn in latexNames.keys()) else vn]
    #     # get means of the variables
    #     means = {}
    #     for v in varList: 
    #         means[v] = np.mean([r['variables'][v]['val'] for r in results])
    #     # get the graphs
    #     graphs = [self.GetGraphVariables(dict(filter(lambda x: x[0] in varList, results[i]['variables'].items())),
    #                                      yoffset=1.+(2*i-len(results)+1)*0.05, offsets=means) for i in range(len(results))]
    #     xmin, xmax = hu.FindGraphsMinX(graphs), hu.FindGraphsMaxX(graphs)
    #     xmin, xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)
    #     nvars = len(varList)
    #     # create frame
    #     c = Canvas(cname, 600, 800)
    #     c['canvas'].SetLeftMargin(0.2)
    #     c['canvas'].SetRightMargin(0.05)
    #     c['canvas'].SetTopMargin(0.07)
    #     hf= c['canvas'].DrawFrame(xmin,0.,xmax,1.)
    #     hf.SetXTitle(xtitle)
    #     hf.SetYTitle('')
    #     hf.GetYaxis().SetLabelSize(0)
    #     hf.GetYaxis().SetNdivisions(0)
    #     # Add variable names
    #     lat = r.TLatex()
    #     lat.SetTextSize(0.04)
    #     lat.SetTextAlign(32)
    #     xlat = xmin-0.02*(xmax-xmin)
    #     for i in range(nvars):
    #         dy = 1./(nvars+1)
    #         lat.DrawLatex(xlat, (i+1)*dy, vnames[i])
    #     # add line at 0
    #     lz= r.TLine(0,0,0,1.)
    #     lz.SetLineStyle(r.kDashed)
    #     lz.Draw()
    #     # create graphs
    #     for i in range(len(graphs)): 
    #         graphs[i].SetLineColor(colors[i])
    #         graphs[i].SetMarkerColor(colors[i])
    #         graphs[i].Draw('P')
    #     # draw legend
    #     leg = r.TLegend(0.2,0.93,0.95,0.99)
    #     if legend is not None:
    #         leg.SetNColumns(len(legend) if len(legend)<5 else 5)
    #         for i in range(len(legend)):
    #             leg.AddEntry(graphs[i],legend[i],'lp')
    #         leg.Draw()
    #     c['extra']+=graphs+[hf,lat,lz,leg]
    #     c['canvas'].Update()
    #     return c

    def GetWorkspaceFromFile(self, ws_name, file_name):
        """Get workspace from file

        Args:
            ws_name (str): the name of the workspace
            file_name (str): the name of the file

        Returns:
            (RooWorkspace): the RooWorkspace
        """
        return r.TFile(file_name).Get(ws_name)
    

class FitAnalyser:
    """A class to analyse fit results
    """
    def __init__(self, fit_results_files=[], fit_utils=None):
        self.results = fit_results_files
        self.fu = fit_utils if fit_utils is not None else FitUtils()
        self.df = self.fit_results_to_dataframe()
        return
    
    def fit_results_to_dataframe(self, fit_results_files=[]):
        """Convert fit results to a pandas dataframe

        Args:
            fit_results_files (list, optional): list of fit results files. Defaults to [].

        Returns:
            (pandas.DataFrame): a dataframe with the fit results
        """
        if len(fit_results_files)==0:
            fit_results_files = self.results
        records = []
        for file_path in fit_results_files:
            res_dict = self.fu.resultsToDictionary(file_path)
            row = {'file_path': file_path}
            for name in res_dict['variables']:
                    row[name] = res_dict['variables'][name]['val']
                    row[name+'_err'] = res_dict['variables'][name]['err']
            records.append(row)
        df = pd.DataFrame(records)
        return df
    
    def identify_outliers(self, variable, threshold=5):
        """Identify outliers in a given variable based on a threshold.

        Args:
            variable (str): The variable to check for outliers.
            threshold (float): The z-score threshold to identify outliers.

        Returns:
            pd.DataFrame: DataFrame containing the outlier rows.
        """
        if variable not in self.df.columns:
            raise ValueError(f'Variable {variable} not found in DataFrame columns')
        mean = self.df[variable].mean()
        std = self.df[variable].std()
        z_scores = (self.df[variable] - mean) / std
        outliers = self.df[abs(z_scores) > threshold]
        return outliers

    def print_file_with_outliers(self):
        """Print file paths of fits that contain outliers.
        """
        outlier_files = {}
        for column in self.df.columns:
            if column != 'file_path' and '_err' not in column:
                outliers = self.identify_outliers(column)
                for _, row in outliers.iterrows():
                    if row['file_path'] not in outlier_files:
                        outlier_files[row['file_path']] = {}
                    outlier_files[row['file_path']][column] = row[column]
        print("Files with outliers:")
        for file_path, value in outlier_files.items():
            print(f"{file_path}: {value}")
        return
    
    def print_bad_fits(self):
        """Print file paths of fits that are not good.
        """
        bad_fits = []
        for index, row in self.df.iterrows():
            fr = row['file_path']
            if not self.fu.IsGoodFit(fr):
                bad_fits += [fr]
        print("Bad fits:")
        for bf in bad_fits:
            print(bf)
        return

    def search_variables_close_to_boundaries(self, z=2., level=1):
        """Search for variables that are close to their boundaries in the fit results.

        Args:
            z (float, optional): Multiplier for the error to define closeness. Defaults to 2..

        Returns:
            dict: A dictionary with file paths as keys and lists of problematic variables as values.
        """
        problematic_fits = {}
        for index, row in self.df.iterrows():
            fr = row['file_path']
            res_dict = self.fu.resultsToDictionary(fr)
            problematic_vars = []
            for var in res_dict['variables'].keys():
                vinfo = res_dict['variables'][var]
                code = self.fu.check_variable(vinfo, z=z)
                if code > level:
                    problematic_vars.append((vinfo['name'], code))
            if len(problematic_vars) > 0:
                problematic_fits[fr] = problematic_vars
        return problematic_fits
    
    def plot_variable_distributions(self, selected_vars=None, pull=False, prevent_overlap=True):
        """Plot distributions of fit variables with optimal square grid layout.

        Args:
            selected_vars (list, optional): List of variables to plot. If None, plots all non-error variables.
            pull (bool): Whether to plot pulls instead of raw values.
            prevent_overlap (bool): Whether to apply text overlap prevention techniques.

        Returns:
            tuple: (fig, axes) matplotlib figure and axes objects.
        """
        
        variables_to_plot = [column for column in self.df.columns if '_err' not in column]
        variables_to_plot.remove('file_path')
        if selected_vars is not None:
            variables_to_plot = [var for var in selected_vars if var in variables_to_plot]
        
        n_plots = len(variables_to_plot)
        if n_plots == 0:
            return None, None
        
        # Calculate optimal grid dimensions (as square as possible)
        n_cols = int(math.ceil(math.sqrt(n_plots)))
        n_rows = int(math.ceil(n_plots / n_cols))
        
        # Create subplots with optimal layout and padding
        fig = plt.figure(figsize=(4*n_cols, 3*n_rows))  # Smaller individual plot size
        
        # Plot each variable
        axs = []
        for i, column in enumerate(variables_to_plot):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            if pull:
                mean = self.df[column].mean()
                std = self.df[column].std()
                pulls = (self.df[column] - mean) / std
                mplhep.histplot(np.histogram(pulls.dropna(), 30), histtype='fill', alpha=0.6, ax=ax)
                #ax.hist(pulls.dropna(), bins=30, alpha=0.7)
                ax.set_xlabel(f'Pulls of {column}', fontsize=9)
            else:
                mplhep.histplot(np.histogram(self.df[column].dropna(), 30), histtype='fill', alpha=0.6, ax=ax)
                ax.set_xlabel(column, fontsize=9)

            # Set title with wrapping for long variable names
            title = column
            if len(title) > 20:  # Wrap long titles
                words = title.split('_')
                if len(words) > 1:
                    mid = len(words) // 2
                    title = '_'.join(words[:mid]) + '\n' + '_'.join(words[mid:])
            ax.set_title(title, fontsize=10, pad=8)
            
            # Set labels with appropriate font size
            ax.set_ylabel('Frequency', fontsize=9)
            
            # Rotate x-axis labels if needed to prevent overlap
            ax.tick_params(axis='x', labelsize=8, rotation=45)
            ax.tick_params(axis='y', labelsize=8)
            
            ax.grid(True, alpha=0.3)
            axs.append(ax)
        
        # Configure layout to prevent overlapping
        if prevent_overlap:
            configure_plot_layout(fig, method='tight_layout')
        
        return fig, axs