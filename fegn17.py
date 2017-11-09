#!/usr/bin/python2.7
from __future__ import print_function
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import sys, os
from scipy.io import loadmat, savemat
gcf, gca = plt.gcf, plt.gca

"""
fegn17 is the same as egn17 except that the computation has been embeded into a function 
so that it can be easilly called for several models

dfegn17 computes finite difference eigenfunction kernels
=> outputs are still messy
=> the subsequent calls to fegn17 should be parallelized
=> display is only for demo
"""
####################
def execbash(script, tmpdir):
    proc = subprocess.Popen("/bin/bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd = tmpdir)
    stdout, stderr = proc.communicate(script)
    return stdout, stderr
####################
def read_TXTout(fin):
    with open(fin, 'r') as fid:
        out = {}
        while True:
            l = fid.readline()
            if l == "": break
            l = l.split('\n')[0].strip()
            ####################
            if "Model:" in l:
                #first section in the file
                fid.readline() #skip header line like "LAYER     H(km)     Vp(km/s)     Vs(km/s)  Density     QA(inv)     QB(inv)"
                LAYER, H, Vp, Vs, Density, QA, QB = [], [], [], [], [], [], []
                while True:
                    l = fid.readline()
                    if l == "": break
                    layer, h, vp, vs, density, qa, qb = np.asarray(l.split('\n')[0].strip().split(), float)
                    for W, w in zip([LAYER, H, Vp, Vs, Density, QA, QB], [layer, h, vp, vs, density, qa, qb]):
                        W.append(w)
                    if h == 0: break
                del layer, h, vp, vs, density, qa, qb #prevent confusion with captial variables
                Z = np.concatenate(([0.], np.cumsum(H[:-1])))
                out['model'] = {"Z" : Z, "H" : H, "Vp" : Vp, "Vs" : Vs, "Rh" : Density}
            ####################
            elif "WAVE" in l and "MODE #" in l:
                #------------
                wavetype = l.split('WAVE')[0].strip()[0] #first letter R or L
                modenum  = int(l.split('#')[-1].split()[0])
                key = "%s%d" % (wavetype, modenum)
                #------------
                l = fid.readline()
                l = l.split('\n')[0].strip()
                l = l.split()
                T, C, U = np.asarray([l[2], l[5], l[8]], float)
                #------------
                l = fid.readline()
                l = l.split('\n')[0].strip()
                l = l.split()
                AR, GAMMA, ZREF = np.asarray([l[1], l[3], l[5]], float) #dont know what it is
                out[key] = {"T" : T, "C" : C, "U" : U, "AR" : AR, "GAMMA" : GAMMA, "ZREF" : ZREF}
                #------------
                l = fid.readline() #skip header line
                #------------
                if wavetype == "R":
                    UR, TR, UZ, TZ, DCDH, DCDA, DCDB, DCDR = [np.empty(len(H), float) for _ in xrange(8)]
                    for i in xrange(len(H)):
                        l = np.asarray(fid.readline().split('\n')[0].split(), float)
                        M = int(l[0])
                        for W, w in zip([UR, TR, UZ, TZ, DCDH, DCDA, DCDB, DCDR], l[1:]):
                            W[M - 1] = w
                    out[key]['UR'] = UR
                    out[key]['TR'] = TR
                    out[key]['UZ'] = UZ
                    out[key]['TZ'] = TZ
                    out[key]['DCDH'] = DCDH
                    out[key]['DCDA'] = DCDA
                    out[key]['DCDB'] = DCDB
                    out[key]['DCDR'] = DCDR
                elif wavetype == "L":
                    UT, TT, DCDH, DCDB, DCDR = [np.empty(len(H), float) for _ in xrange(5)]
                    for i in xrange(len(H)):
                        l = np.asarray(fid.readline().split('\n')[0].split(), float)
                        M = int(l[0])
                        for W, w in zip([UT, TT, DCDH, DCDB, DCDR], l[1:]):
                            W[M - 1] = w
                    out[key]['UT'] = UT
                    out[key]['TT'] = TT
                    out[key]['DCDH'] = DCDH
                    out[key]['DCDB'] = DCDB
                    out[key]['DCDR'] = DCDR
                else: raise Exception('error')
    return out
   
####################
def readmod96(mod96file):
    """read files at mod96 format (see Herrmann's doc)"""
    with open(mod96file, 'r') as fid:
        while True:
            l = fid.readline()
            if l == "": break
            l = l.split('\n')[0]
            if "H" in l and "VP" in l and "VS" in l:
                H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS = [[] for _ in xrange(10)]
                while True:
                    l = fid.readline()
                    l = l.split('\n')[0]
                    l = np.asarray(l.split(), float)
                    for W, w in zip([H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS], l):
                        W.append(w)
                    if l[0] == 0.: #thickness is 0 = ending signal (half space)
                        break
                if l[0] == 0.: break            
    H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS = [np.asarray(_, float) for _ in H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS]
    Z = np.concatenate(([0.], H[:-1].cumsum()))
    return Z, H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS
####################
def writemod96(filename, H, VP, VS, RHO, QP=None, QS=None, ETAP=None, ETAS=None, FREFP=None, FREFS=None):
    header ='''MODEL.01
whatever
ISOTROPIC
KGS
FLAT EARTH
1-D
CONSTANT VELOCITY
LINE08
LINE09
LINE10
LINE11
      H(KM)   VP(KM/S)   VS(KM/S) RHO(GM/CC)     QP         QS       ETAP       ETAS      FREFP      FREFS
'''

    assert filename.endswith('.mod96')
    if QP is None: 
        QP    = np.zeros_like(H)
        ETAP  = np.zeros_like(H)
        FREFP = np.ones_like(H)
    if QS is None: 
        QS    = np.zeros_like(H)
        ETAS  = np.zeros_like(H)
        FREFS = np.ones_like(H)

    with open(filename, 'w') as fid:
        fid.write(header)
        for h, vp, vs, rho, qp, qs, etap, etas, frefp, frefs in \
            zip(H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS):
            fid.write('%f %f %f %f %f %f %f %f %f %f\n' % (h, vp, vs, rho, qp, qs, etap, etas, frefp, frefs))

####################
def checkCPS():
    "verify if CPS is installed"
    stdout, _ = execbash('which sdisp96', ".")
    if stdout == "":
        raise Exception('sdisp96 not found, make sure CPS is installed and added to the path')

####################
def fegn17(mod96file, wavetype, nmod, freq, tmpdir = None, fig = None, cleanup = True):
    """ function to compute eignenfunctions and phase velocity sensitivity kernels
    input 
        mod96file       = ascii file at mod96 format (see Herrmann's documentation)
        wavetype        = R or L for Rayleigh or Love
        nmod            = mode number >= 0
        freq            = frequency (Hz) > 0
        tmpdir          = non existent directory name to create and remove after running
        fig             = figure for display or None
    output
        out             = dictionnary containing the function and phase velocity sensitivity kernels
                          for all modes between 0 and nmod
    """
    #------------------- check inputs
    assert os.path.exists(mod96file)
    assert wavetype in 'RL'
    assert nmod >= 0
    assert freq > 0.0

    #------------------- create temporary directory
    if tmpdir is None: tmpdir = "/tmp/tmpdir_fegn17_%10d" % (np.random.rand() * 1e10)
    while os.path.isdir(tmpdir):
        #make sure tmpdir does not exists
        tmpdir += "_%10d" % (np.random.rand() * 1.0e10)
    os.mkdir(tmpdir) 

    #------------------- write bash script   
    script = """
    #rm -f DISTFILE.dst sdisp96.??? s[l,r]egn96.??? S[L,R]DER.PLT S[R,L]DER.TXT

    cat << END > DISTFILE.dst 
    10. 0.125 256 -1.0 6.0
    END

    ln -s {mod96file} model.mod96
    
    ############################### prepare dispersion
    sprep96 -M model.mod96 -dfile DISTFILE.dst -NMOD {nmodmax} -{wavetype} -FREQ {freq}

    ############################### run dispersion
    sdisp96

    ############################### compute eigenfunctions
    s{minuswavetype}egn96 -DE 

    ############################### ouput eigenfunctions
    sdpder96 -{wavetype} -TXT  # plot and ascii output

    ############################### clean up
    #rm -f sdisp96.dat DISTFILE.dst sdisp96.??? s[l,r]egn96.??? S[L,R]DER.PLT

    """.format(mod96file = os.path.realpath(mod96file), 
               nmodmax = nmod + 1, 
               wavetype = wavetype, 
               minuswavetype = wavetype.lower(),
               freq = freq)
    script = "\n".join([_.strip() for _ in script.split('\n')]) #remove indentation from script

    #------------------- execute bash commands
    stdout, stderr = execbash(script, tmpdir = tmpdir)
    expected_output = "%s/S%sDER.TXT" % (tmpdir, wavetype)
    if not os.path.exists(expected_output):
        raise Exception('output file %s not found, script failed \n%s' % (expected_output, stderr))

    #------------------- read Herrmann's output
    out = read_TXTout(expected_output)
    if cleanup:
        #remove temporary directory
        execbash('rm -rf %s' % tmpdir, ".")

    #------------------- display results for last mode
    if fig is not None:
        """not ready, only for demo"""
        key = '%s%d' % (wavetype, nmod)
        if key not in out.keys():
            raise Exception('key %s not found in output file, is the frequency (%fHz) below the cut-off frequency for mode %d?' % (key, freq, nmod))

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharey = ax1)
        ax1.invert_yaxis()
        #------------------
        z  = np.concatenate((np.repeat(out['model']["Z"], 2)[1:], [sum(out['model']["H"]) * 1.1]))
        vp = np.repeat(out['model']["Vp"], 2)
        vs = np.repeat(out['model']["Vs"], 2)
        rh = np.repeat(out['model']["Rh"], 2)
        ax1.plot(vp, z, label = "Vp")
        ax1.plot(vs, z, label = "Vs")
        ax1.plot(rh, z, label = "Rh")
        ax1.legend()
        ax1.grid(True)

        #------------------
        if wavetype == "R":
            ax2.plot(out[key]['UR'],  out['model']['Z'], label = "UR") #radial displacement
            ax2.plot(out[key]['UZ'],  out['model']['Z'], label = "UZ")
            ax2.plot(out[key]['TR'],  out['model']['Z'], label = "TR") #radial stress
            ax2.plot(out[key]['TZ'],  out['model']['Z'], label = "TZ")
        if wavetype == "L":
            ax2.plot(out[key]['UT'],  out['model']['Z'], label = "UT")
            ax2.plot(out[key]['TT'],  out['model']['Z'], label = "TT")        

        ax2.set_title("%s : T = %fs" % (key, out[key]["T"]))
        ax2.legend()
        ax1.set_ylabel('depth (km)')
        ax2.set_xlabel('eigenfunctions')
        ax2.grid(True)
    #------------------- 
    return out
####################
def dfegn17(mod96file, wavetype, nmod, freq, pertu = 0.05, tmpdir = None, cleanup = True, fig = None):
    """compute eigenfunctions sensitivity kernels using finite differences (serial)
    input : 
        see function egn17
        pertu is the relative perturbation to impose in each layer
    output : 
        dout : dictionnary
            dout['Z']      : 1d depth array, similar for depth profile and eigenfunctions
            dout['DURDVS'] : 3d kernel, first derivative of UR relative to VS in each layer
                 first  index = mode number
                 second index = number of the perturbated layer
                 third  index = sample of the eigenfunction that has been perturbated             
    """
    
    #create the main temporary directory
    if tmpdir is None: 
        tmpdir = "/tmp/tmpdir_dfegn17_%10d" % (np.random.rand() * 1e10)
    while os.path.isdir(tmpdir):
        tmpdir += "_%10d" % (np.random.rand() * 1e10)
    os.mkdir(tmpdir)

    #read starting model
    Z, H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS = readmod96('example.mod96')
    model0 = np.concatenate((H, VP, VS, RHO))
    nlayer = len(H)
    IH  = np.arange(nlayer) #index of thickness parameters
    IVP = np.arange(1 * nlayer, 2 * nlayer) #index of Vp parameters
    IVS = np.arange(2 * nlayer, 3 * nlayer) 
    IRH = np.arange(3 * nlayer, 4 * nlayer) 



    #compute eignefunctions for starting model
    out0 = fegn17(mod96file, wavetype, nmod, freq, tmpdir = "%s/%s" % (tmpdir, "startingmodel"), fig = None, cleanup = False)

    #initiate kernels
    if wavetype == "R":
        DURDN = np.zeros((nmod + 1, nlayer, len(model0)), float) * np.nan
        DUZDN = np.zeros((nmod + 1, nlayer, len(model0)), float) * np.nan
        DTRDN = np.zeros((nmod + 1, nlayer, len(model0)), float) * np.nan
        DTZDN = np.zeros((nmod + 1, nlayer, len(model0)), float) * np.nan
    elif wavetype == "L":
        DUTDN = np.zeros((nmod + 1, nlayer, len(model0)), float) * np.nan
        DTTDN = np.zeros((nmod + 1, nlayer, len(model0)), float) * np.nan


    #perturbate each parameter subsequently
    for nparam in xrange(len(model0)):
        if nparam == nlayer - 1 :
            #thickness of the half space: meaningless
            continue

        #prepare the perturbated model and determine the sub-directory name
        modeln = model0.copy()
        modeln[nparam] *= (1.0 + pertu)
        tmpdir_n = "%s/param%06d" % (tmpdir, nparam)
        mod96file_n = "%s/model_param%06d.mod96" % (tmpdir, nparam) #write the perturbated file in the main temporary directory

        #write perturbated model
        writemod96(mod96file_n, 
            H   = modeln[IH], 
            VP  = modeln[IVP], 
            VS  = modeln[IVS], 
            RHO = modeln[IRH], 
            QP  = QP, QS = QS, ETAP=ETAP, ETAS=ETAS, FREFP=FREFP, FREFS=FREFS) #keep attenuation untouched

        #call fegn17 with perturbated model
        out_n = fegn17(mod96file_n, wavetype, nmod, freq, tmpdir = tmpdir_n, fig = None, cleanup = False)
        dm = (modeln[nparam] - model0[nparam])

        for modnum in xrange(nmod+1):
            key = "%s%d" % (wavetype, modnum)

            if wavetype == "R":
                DURDN[modnum, :, nparam] = (out_n[key]['UR'] - out0[key]['UR']) / dm
                DUZDN[modnum, :, nparam] = (out_n[key]['UZ'] - out0[key]['UZ']) / dm
                DTRDN[modnum, :, nparam] = (out_n[key]['TR'] - out0[key]['TR']) / dm
                DTZDN[modnum, :, nparam] = (out_n[key]['TZ'] - out0[key]['TZ']) / dm
            elif wavetype == "L":

                DUTDN[modnum, :, nparam] = (out_n[key]['UT'] - out0[key]['UT']) / dm
                DTTDN[modnum, :, nparam] = (out_n[key]['TT'] - out0[key]['TT']) / dm


    #-------------------
    dout = {"Z" : out0['model']["Z"], "T" : 1. / freq}
    if wavetype == "R":
        dout['DURDH']  = DURDN[:, :, IH ]
        dout['DURDVP'] = DURDN[:, :, IVP]
        dout['DURDVS'] = DURDN[:, :, IVS]
        dout['DURDRH'] = DURDN[:, :, IRH]

        dout['DUZDH']  = DUZDN[:, :, IH ]
        dout['DUZDVP'] = DUZDN[:, :, IVP]
        dout['DUZDVS'] = DUZDN[:, :, IVS]
        dout['DUZDRH'] = DUZDN[:, :, IRH]

        dout['DTRDH']  = DTRDN[:, :, IH ]
        dout['DTRDVP'] = DTRDN[:, :, IVP]
        dout['DTRDVS'] = DTRDN[:, :, IVS]
        dout['DTRDRH'] = DTRDN[:, :, IRH]

        dout['DTZDH']  = DTZDN[:, :, IH ]
        dout['DTZDVP'] = DTZDN[:, :, IVP]
        dout['DTZDVS'] = DTZDN[:, :, IVS]
        dout['DTZDRH'] = DTZDN[:, :, IRH]
                                        
    elif wavetype == "L":               
        dout['DUTDH']  = DUTDN[:, :, IH ]
        dout['DUTDVP'] = DUTDN[:, :, IVP]
        dout['DUTDVS'] = DUTDN[:, :, IVS]
        dout['DUTDRH'] = DUTDN[:, :, IRH]
                                        
        dout['DTTDH']  = DTTDN[:, :, IH ]
        dout['DTTDVP'] = DTTDN[:, :, IVP]
        dout['DTTDVS'] = DTTDN[:, :, IVS]
        dout['DTTDRH'] = DTTDN[:, :, IRH]

    if cleanup:
        #remove temporary directory
        execbash('rm -rf %s' % tmpdir, ".")

    if fig is not None:

        ax2 = fig.add_subplot(224)
        ax1 = fig.add_subplot(223, sharey = ax2)
        ax3 = fig.add_subplot(222, sharex = ax2)

        #------------------
        ax1.invert_yaxis()
        #------------------
        z  = np.concatenate((np.repeat(out0['model']["Z"], 2)[1:], [sum(out0['model']["H"]) * 1.1]))
        vp = np.repeat(out0['model']["Vp"], 2)
        vs = np.repeat(out0['model']["Vs"], 2)
        rh = np.repeat(out0['model']["Rh"], 2)
        ax1.plot(vp, z, label = "Vp")
        ax1.plot(vs, z, label = "Vs")
        ax1.plot(rh, z, label = "Rh")
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylabel('model depth (km)')

        #------------------
        if wavetype == "R":
            vmax = abs(dout['DUZDVS'][nmod, :, :]).max()
            ax3.plot(dout["Z"], out0["%s%d" % (wavetype, nmod)]['UZ'], label = "UZ")
            Y = dout['DUZDVS'][nmod, :, :]
            Y = np.ma.masked_where(np.isnan(Y), Y)
            ax2.pcolormesh(dout["Z"], dout["Z"], Y, vmin = -vmax, vmax = vmax)
            ax2.set_title("DUZ/DVS, T = %f, mode %d" % (1. / freq, nmod))
            ax3.set_xlabel('eignefunction depth (km)')
        elif wavetype == "L":
            vmax = abs(dout['DUTDVS'][nmod, :, :]).max()
            ax3.plot(dout["Z"], out0["%s%d" % (wavetype, nmod)]['UT'], label = "UT")
            Y = dout['DUTDVS'][nmod, :, :]
            Y = np.ma.masked_where(np.isnan(Y), Y)
            ax2.pcolormesh(dout["Z"], dout["Z"], Y, vmin = -vmax, vmax = vmax)
            ax2.set_title("DUT/DVS, T = %f, mode %d" % (1. / freq, nmod))
            ax3.set_xlabel('eignefunction depth (km)')

        ax3.xaxis.set_label_position("top")    
        ax3.legend()
        ax2.grid(True)
        ax3.grid(True)

    return dout

####################
if __name__ == "__main__":
    mod96file = "./example.mod96"
    wavetype = "R"
    nmod = 1
    freq = 0.5

    dout = dfegn17(mod96file, wavetype, nmod, freq, pertu = 0.05, tmpdir = "./tmptmptmp", fig = gcf())

    gcf().show()
    raw_input('pause')



