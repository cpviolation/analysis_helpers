import os
from ._root_import import import_root_with_proxy, is_root_available as _is_root_available, require_root as _require_root

r, _HAS_ROOT = import_root_with_proxy()
from array import array
import pandas as pd
import numpy as np
import uproot


def is_root_available():
    """Return whether PyROOT can be imported."""
    return _is_root_available(_HAS_ROOT)


def require_root():
    """Raise a clear exception if PyROOT is not available."""
    _require_root(
        _HAS_ROOT,
        "PyROOT is not available. Install and configure ROOT to use this function.",
    )

##########
# Load libraries
##
ROOTLibraries = ['Core','RIO','Hist','Graf','Graf3d','Postscript','Gpad',
                 'RooFit','RooFitCore','Tree','MathCore','Foam','Physics']

ROOT2ArrayTypes = {
    'C': 'c', # a character string terminated by the 0 character
    'B': 'b', # an 8 bit signed integer (Char_t)
    'b': 'B', # an 8 bit unsigned integer (UChar_t)
    'S': 'h', # a 16 bit signed integer (Short_t)
    's': 'H', # a 16 bit unsigned integer (UShort_t)
    'I': 'i', # a 32 bit signed integer (Int_t)
    'i': 'I', # a 32 bit unsigned integer (UInt_t)
    'F': 'f', # a 32 bit floating point (Float_t)
    'D': 'd', # a 64 bit floating point (Double_t)
    'L': 'l', # a 64 bit signed integer (Long64_t)
    'l': 'L', # a 64 bit unsigned integer (ULong64_t)
    'O': 'i'  # a boolean (Bool_t)
    }

ROOT2ArrayTypes_Long = {
    'C'        : 'c', # a character string terminated by the 0 character
    'Char_t'   : 'b', # an 8 bit signed integer (B)
    'UChar_t'  : 'B', # an 8 bit unsigned integer (b)
    'Short_t'  : 'h', # a 16 bit signed integer (S)
    'UShort_t' : 'H', # a 16 bit unsigned integer (s)
    'Int_t'    : 'i', # a 32 bit signed integer (I)
    'UInt_t'   : 'I', # a 32 bit unsigned integer (i)
    'Float_t'  : 'f', # a 32 bit floating point (F)
    'Double_t' : 'd', # a 64 bit floating point (D)
    'Long64_t' : 'l', # a 64 bit signed integer (L)
    'ULong64_t': 'L', # a 64 bit unsigned integer (l)
    'Bool_t'   : 'i'  # a boolean (O)
    }

Array2ROOTTypes = {
    'c': 'C', # a character string terminated by the 0 character
    'b': 'B', # an 8 bit signed integer (Char_t)
    'B': 'b', # an 8 bit unsigned integer (UChar_t)
    'h': 'S', # a 16 bit signed integer (Short_t)
    'H': 's', # a 16 bit unsigned integer (UShort_t)
    'i': 'I', # a 32 bit signed integer (Int_t)
    'I': 'i', # a 32 bit unsigned integer (UInt_t)
    'f': 'F', # a 32 bit floating point (Float_t)
    'd': 'D', # a 64 bit floating point (Double_t)
    'l': 'L', # a 64 bit signed integer (Long64_t)
    'L': 'l', # a 64 bit unsigned integer (ULong64_t)
    }

Python2ROOTTypes = {
    'character': 'C',
    'int'      : 'I',
    'long'     : 'L',
    'float'    : 'F',
    }


def LoadCompiledLibraries(libraries=None):
    """Load C++ Compiled libraries

    Args:
        libraries (list, optional): A list of C++ compiled library paths. Defaults to None.
    """
    if libraries is None: 
        return
    require_root()
    for lib in libraries:
        r.gSystem.Load(lib)
    return


def DefineTree(variables, name='tree', title= 'a tree for my variable'):
    """Given a set of variables in the format of a dictionary, a ROOT TTree object is created. 
    The `variables` dictionary should have the form
    ```
    variables = { <str>vname: <str>vtype }
    ```
    where 
    * `vname`: the name of the variable,
    * `vtype`: the type of the variable, one of the `ROOT2ArrayTypes.keys()`

    Args:
        variables (dict): a dictionary with variable names and their respective type.
        name (str, optional): the name of the ROOT TTree object. Defaults to 'tree'.
        title (str, optional): the title of the ROOT TTree objects. Defaults to 'a tree for my variable'.
    """    
    require_root()
    t = r.TTree(name,title)
    t.SetDirectory(0)
    Vars = {}
    for vname, vtype in variables.items():
        Vars.update( { vname : array( ROOT2ArrayTypes[vtype],[0]) } )
    for var,args in Vars.items():
        print(var,args,args[0],type(args[0]).__name__,args.typecode)
        t.Branch(var, args, '%s/%s' % (var, Array2ROOTTypes[args.typecode]) )
#        Vars.update( { vname : array('i' if vtype == 'I' else 'f',[0]) })
#    for var,args in Vars.items(): t.Branch(var, args, '%s/%s' % (var, 'I' if type(args[0]) == type(0) else 'F'))
    t.Print()
    return dict(tree=t, vars=Vars)

def GetTreeVariables(t):
    """Returns the dictionary of variables and types from TTree branches

    Args:
        t (ROOT.TTree): The ROOT TTree object
    """
    require_root()
    tvars = {}
    for l in t.GetListOfLeaves():
        tvars.update( {l.GetName(): Array2ROOTTypes[ROOT2ArrayTypes_Long[l.GetTypeName()]] } )
    return tvars
    

def SaveTree(t, fname):
    """Save a TTree into a file

    Args:
        t (ROOT.TTree): A ROOT TTree object
        fname (str): The file name
    """    
    require_root()
    # Save a tree
    f = r.TFile(fname, 'recreate')
    f.cd()
    t.Write()
    t.SetDirectory(0)
    f.Close()
    print('tree '+t.GetName()+' is saved to file '+fname)
    return

def SaveTrees(trees, fname):
    """Save many TTrees into a file

    Args:
        trees (list): a list of TTree objects
        fname (str): the file name
    """    
    require_root()
    # Save trees
    f = r.TFile(fname, 'recreate')
    f.cd()
    for t in trees:
        t.Write()
        t.SetDirectory(0)
        print('tree '+t.GetName()+' is saved to file '+fname)
    f.Close()
    return

def ReduceTree(t, fname, nentries=None, cut='', firstentry=0, options=''):
    """Reduce a TTree to a new one and save it to a file

    Args:
        t (ROOT.TTree): A ROOT TTree object
        fname (str): The file name
        nentries (int, optional): The number of entries to copy. Defaults to r.kMaxEntries.
        cut (str, optional): A selection to be made on the ROOT TTree branches. Defaults to ''.
        firstentry (int, optional): The first entry to copy. Defaults to 0.
        options (str, optional): Options to run the `ROOT.TTree.CopyTree()` function. Defaults to ''.
    """     
    require_root()
    if nentries is None:
        nentries = r.TTree.kMaxEntries

    # Create a new tree
    f = r.TFile(fname,'recreate')
    t_new = t.CopyTree(cut,options,nentries,firstentry)
    t_new.Write()
    t_new.SetDirectory(0)
    f.Close()
    print('tree '+t.GetName()+' is reduced to file '+fname)
    return
    
def ConvertTChain2DataFrame(ch,columns=None):
    """Convert a TChain to pandas.DataFrame

    Args:
        ch (r.TChain): a ROOT TChain object
        columns (list, optional): list of variables to save in the data frame. Defaults to None.

    Returns:
        pandas.DataFrame: the output data frame
    """    
    require_root()
    if columns is None: columns = [i.GetName() for i in ch.GetListOfLeaves()]
    rdf = r.RDataFrame(ch)
    df  = pd.DataFrame(rdf.AsNumpy(columns=columns))
    return df

def Canvas(name='c1', width=600, height=600, title=''):
    """A dictionary with a ROOT TCanvas and extra objects

    Args:
        name (str, optional): the name of the TCanvas. Defaults to 'c1'.
        width (int, optional): the width of the TCanvas. Defaults to 600.
        height (int, optional): the height of the TCanvas. Defaults to 600.
        title (str, optional): the title of the TCanvas. Defaults to ''.

    Returns:
        dict: {'canvas': ROOT.TCanvas, 'extra': <list>}
    """
    require_root()
    if title=='':title=name
    canvas = r.TCanvas(name, title, width, height)
    # Avoid crashes at Python shutdown due to TCanvas double deletion via CPyCppyy.
    r.SetOwnership(canvas, False)
    return {'canvas': canvas, 'extra':[]}

def SaveCansAsPdf(pdfName, cans):
    """Save a list of TCanvas objects into a single PDF file
    """
    require_root()
    # formerly known as SavePdf
    print('Saving',[can.GetName() for can in cans])
    idx,idxmax = 0, len(cans)-1
    for can in cans:
        if idx==0: can.Print(pdfName+".pdf(")
        elif idx==idxmax: can.Print(pdfName+".pdf)")
        else: can.Print(pdfName+'.pdf')
        idx+=1
    return


def TTree2Array(tree, leaves=None):
    """Transform a ROOT TTree into a numpy array

    Args:
        tree (TTree): the ROOT TTree object
        leaves (list, optional): the list of leaves to be transformed. Defaults to all leaves.

    Returns:
        array: The numpy array
    """
    require_root()
    tree_leaves = [leaf.GetName() for leaf in tree.GetListOfLeaves()]
    leaves = tree_leaves if leaves is None else list(leaves)
    if len(leaves) == 0:
        return np.empty((tree.GetEntries(), 0))
    # RDataFrame is fast, but it requires a file-backed tree.
    if getattr(tree, 'GetCurrentFile', lambda: None)():
        r.EnableImplicitMT()  # opzionale: multithreading
        rdf = r.RDataFrame(tree)
        d = rdf.AsNumpy(columns=leaves)
        return np.column_stack([d[leaf] for leaf in leaves])
    # loop on in-memory tree, which is slower but works without a file
    rows = []
    for entry_idx in range(tree.GetEntries()):
        tree.GetEntry(entry_idx)
        rows.append([getattr(tree, leaf) for leaf in leaves])
    return np.asarray(rows)


def ROOT2MPLLineStyle(lineStyle):
    """A function to convert ROOT line styles to Matplotlib line styles

    Args:
        lineStyle (int): ROOT line style index

    Returns:
        str: matplotlib line style
    """
    if lineStyle not in [1,2,3,4,5,6,7,8,9,10]:
        return 'solid'
    mpl_line_styles = {
        1: 'solid',
        2: 'dashed',
        3: 'dotted',
        4: 'densely dashdotted',
        5: 'dashdotted',
        6: 'densely dashdotdotted',
        7: 'densely dashed',
        8: 'dashdotdotted',
        9: 'long dash with offset',
        10: 'dashdot'
    }
    return mpl_line_styles[lineStyle]


def ROOT2MPLColor(color):
    """A function to convert ROOT colors to Matplotlib colors

    Args:
        color (TColor): ROOT TColor object

    Returns:
        str: matplotlib color name
    """    
    require_root()
    mpl_color_name = color.GetName().lower()[1:]
    mpl_color_name = 'tab:{}'.format(mpl_color_name) if mpl_color_name in ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan'] else 'tab:blue'
    return mpl_color_name


def ROOT2MPLText(text):
    """A function to convert ROOT text to Matplotlib text

    Args:
        text (str): the text to convert

    Returns:
        str: matplotlib text
    """
    mpl_text = str(text)
    if '#' not in mpl_text and '^' not in mpl_text and '_' not in mpl_text:
        return mpl_text

    # Convert ROOT text-style commands to Matplotlib mathtext-friendly forms.
    replacements = {
        '#it{': r'\mathit{',
        '#bf{': r'\mathbf{',
        '#rm{': r'\mathrm{',
        '#bar{': r'\overline{',
    }
    for old, new in replacements.items():
        mpl_text = mpl_text.replace(old, new)

    mpl_text = mpl_text.replace('#', '\\')
    mpl_text = mpl_text.replace('%', r'\%')
    return f'${mpl_text}$'


# Create and save histograms using uproot
def save_np_histograms_uproot(histograms_dict, filename):
    """Save multiple numpy histograms to a ROOT file with uproot

    Args:
        histograms_dict (dict): A dictionary of histograms to save.
        filename (str): The name of the output ROOT file.
    """
    with uproot.recreate(filename) as f:
        for name, (hist_values, bin_edges) in histograms_dict.items():
            # Convert numpy histogram to ROOT-compatible format
            f[name] = (hist_values, bin_edges)
    return


def load_np_histograms_uproot(filename):
    """Load numpy histograms from a ROOT file with uproot

    Args:
        filename (str): The name of the input ROOT file.

    Returns:
        dict: A dictionary of histograms.
    """
    require_root()
    histograms = {}
    with uproot.open(filename) as f:
        for key in f.keys(cycle=None):
            hist = f[key]
            histograms[key] = hist.to_numpy()  # Returns (values, edges)
    return histograms