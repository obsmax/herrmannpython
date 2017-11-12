#!/usr/bin/python2.7
from __future__ import print_function
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import subprocess
import numpy as np
import sys, os
from scipy.io import loadmat, savemat
gcf, gca = plt.gcf, plt.gca

"""
main_fegn_srfker is based on calls to dfegn17.py and  and launch_surf96

dfegn17 computes sensitivity of the eigenfunctions with respect 
to the model parameters in different layers by finite difference
typical output : out[R0][DUZDVS][(depth_Uz),(depth_perturbed_layer),(frequency)]

launch_surf96 computes sensitivity of Vg and Vphi to the same parameters
(except rho), directly in CPS (new function srfker96)
typical output : out[R0][dbc][(depth_perturbed_layer),(frequency)]
                          |
                 -----dV_phi/d_Vs-------
outputs from both functions are saved seperately

"""
####################
def execbash(script, tmpdir):
    proc = subprocess.Popen("/bin/bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd = tmpdir)
    stdout, stderr = proc.communicate(script)
    return stdout, stderr
####################
def read_TXTout_egn17(fin):
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
                #------------ initialize output for new wavetype-mode
                if wavetype == "R":
                    UR, TR, UZ, TZ, DCDH, DCDA, DCDB, DCDR = [np.zeros(len(H), float) for _ in xrange(8)]
                elif wavetype == "L":
                    UT, TT, DCDH, DCDB, DCDR = [np.zeros(len(H), float) for _ in xrange(5)]
                else: raise Exception('error: unknown wavetype')
                
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
                
                #------------
                l = fid.readline() #skip header line
                #------------
                if wavetype == "R":
                    #UR, TR, UZ, TZ, DCDH, DCDA, DCDB, DCDR = [np.empty(len(H), float) for _ in xrange(8)]
                    for i in xrange(len(H)):
                        l = np.asarray(fid.readline().split('\n')[0].split(), float)
                        M = int(l[0])
                        for W, w in zip([UR, TR, UZ, TZ, DCDH, DCDA, DCDB, DCDR], l[1:]):
                            W[M - 1] = w

                elif wavetype == "L":
                    for i in xrange(len(H)):
                        l = np.asarray(fid.readline().split('\n')[0].split(), float)
                        M = int(l[0])
                        for W, w in zip([UT, TT, DCDH, DCDB, DCDR], l[1:]):
                            W[M - 1] = w

                else: raise Exception('error: unknown wavetype')
                
                if key in out:
                    out[key]['T'] = np.append(out[key]['T'], T)
                    out[key]['C'] = np.append(out[key]['C'], C)
                    out[key]['U'] = np.append(out[key]['U'], U)
                    out[key]['AR'] = np.append(out[key]['AR'], AR)
                    out[key]['GAMMA'] = np.append(out[key]['GAMMA'], GAMMA)
                    out[key]['ZREF'] = np.append(out[key]['ZREF'], ZREF)
                    if wavetype == "R":
                        out[key]['UR'] = np.concatenate((out[key]['UR'], UR), axis=0)
                        out[key]['TR'] = np.concatenate((out[key]['TR'], TR), axis=0)
                        out[key]['UZ'] = np.concatenate((out[key]['UZ'], UZ), axis=0)
                        out[key]['TZ'] = np.concatenate((out[key]['TZ'], TZ), axis=0)
                        out[key]['DCDH'] = np.concatenate((out[key]['DCDH'], DCDH), axis=0)
                        out[key]['DCDA'] = np.concatenate((out[key]['DCDA'], DCDA), axis=0)
                        out[key]['DCDB'] = np.concatenate((out[key]['DCDB'], DCDB), axis=0)
                        out[key]['DCDR'] = np.concatenate((out[key]['DCDR'], DCDR), axis=0)        
                    elif wavetype == "L":
                        out[key]['UT'] = np.concatenate((out[key]['UT'], UT), axis=0)
                        out[key]['TT'] = np.concatenate((out[key]['TT'], TT), axis=0)
                        out[key]['DCDH'] = np.concatenate((out[key]['DCDH'], DCDH), axis=0)
                        out[key]['DCDB'] = np.concatenate((out[key]['DCDB'], DCDB), axis=0)
                        out[key]['DCDR'] = np.concatenate((out[key]['DCDR'], DCDR), axis=0)
                else:
                    out[key] = {"T" : T, "C" : C, "U" : U, "AR" : AR, "GAMMA" : GAMMA, "ZREF" : ZREF}
                    if wavetype == "R":
                        out[key]['UR'] = UR
                        out[key]['TR'] = TR
                        out[key]['UZ'] = UZ
                        out[key]['TZ'] = TZ
                        out[key]['DCDH'] = DCDH
                        out[key]['DCDA'] = DCDA
                        out[key]['DCDB'] = DCDB
                        out[key]['DCDR'] = DCDR                     
                    elif wavetype == "L":
                        out[key]['UT'] = UT
                        out[key]['TT'] = TT
                        out[key]['DCDH'] = DCDH
                        out[key]['DCDB'] = DCDB
                        out[key]['DCDR'] = DCDR                    
                            
                    else: raise Exception('error: unknown wavetype')
    for key, subkey in out.items():
        if key is not 'model':
            for a, b in subkey.items():
                if a not in ["T","C","U","AR","GAMMA","ZREF"]:
                    len_array=len(out[key][a])
                    nb_T=len(out[key]["T"])
                    if nb_T*len(H) != len_array:
                        raise Exception('error: array size not consistent with number of periods')
                    out[key][a]=np.reshape(out[key][a],(len(H),nb_T),order='F')           
                        # to be done : concatenate all frequencies in out[key]
    return out

####################
def read_TXTout_surf96(fin):
    with open(fin, 'r') as fid:
        out = {}
        h_flag = 0 # flag for reading layers only once
        while True:
            l = fid.readline()
            if l == "": break
            l = l.split('\n')[0].strip()
            ######################## read the depths of layers
            if h_flag == 0:
                if "wave" in l:
                    l = fid.readline()
                    LAYER, H = [], []
                    while True:
                        l = fid.readline()
                        if l == "": break
                        layer, h = np.asarray(l.split('\n')[0].strip().split()[:2], float)
                        for W, w in zip([LAYER, H], [layer, h]):
                            W.append(w)
                        if h == 0: break
                    del layer, h #prevent confusion with captial variables
                    Z = np.concatenate(([0.], np.cumsum(H[:-1])))
                    out['model'] = {"Z" : Z, "H" : H}
                    h_flag = 1
                    fid.seek(0)
            ########################
            if False:
                print("hello")
            elif "Elastic" in l and "wave" in l and "Mode =" in l:
                #------------
                wavetype = l.split('wave')[0].strip()[9] #first letter R or L
                modenum  = int(l.split('=')[-3].split()[0])
                key = "%s%d" % (wavetype, modenum)
                if wavetype == "L":
                    dbc, dbU, dhc, dhU = [np.empty(len(H), float) for _ in xrange(4)]
                elif wavetype == "R":
                    dac, dbc, daU, dbU, dhc, dhU = [np.empty(len(H), float) for _ in xrange(6)]
                else: raise Exception('error: unknown wavetype')
                
                #------------
                l = l.split('Period=')[1].strip().split()
                T, C, U = np.asarray([l[0], l[5], l[7]], float)
                #------------
                l = fid.readline() #skip header line
                #------------
                if wavetype == "L":
                    for i in xrange(len(H)):
                        l = np.asarray(fid.readline().split('\n')[0].split(), float)
                        M = int(l[0])
                        for W, w in zip([dbc, dbU, dhc, dhU], l[2:]):
                            W[M - 1] = w
                elif wavetype == "R":
                    for i in xrange(len(H)):
                        l = np.asarray(fid.readline().split('\n')[0].split(), float)
                        M = int(l[0])
                        for W, w in zip([dac, dbc, daU, dbU, dhc, dhU], l[2:]):
                            W[M - 1] = w
                else: raise Exception('error: unknown wavetype')
                
                if key in out:
                    out[key]['T'] = np.append(out[key]['T'], T)
                    out[key]['C'] = np.append(out[key]['C'], C)
                    out[key]['U'] = np.append(out[key]['U'], U)
                    if wavetype == "R":
                        out[key]['dac'] = np.concatenate((out[key]['dac'], dac), axis=0)
                        out[key]['dbc'] = np.concatenate((out[key]['dbc'], dbc), axis=0)
                        out[key]['daU'] = np.concatenate((out[key]['daU'], daU), axis=0)
                        out[key]['dbU'] = np.concatenate((out[key]['dbU'], dbU), axis=0)
                        out[key]['dhc'] = np.concatenate((out[key]['dhc'], dhc), axis=0)
                        out[key]['dhU'] = np.concatenate((out[key]['dhU'], dhU), axis=0)       
                    elif wavetype == "L":
                        out[key]['dbc'] = np.concatenate((out[key]['dbc'], dbc), axis=0)
                        out[key]['dbU'] = np.concatenate((out[key]['dbU'], dbU), axis=0)
                        out[key]['dhc'] = np.concatenate((out[key]['dhc'], dhc), axis=0)
                        out[key]['dhU'] = np.concatenate((out[key]['dhU'], dhU), axis=0) 
                else:
                    out[key] = {"T" : T, "C" : C, "U" : U}
                    if wavetype == "R":
                        out[key]['dac'] = dac
                        out[key]['dbc'] = dbc
                        out[key]['daU'] = daU
                        out[key]['dbU'] = dbU
                        out[key]['dhc'] = dhc
                        out[key]['dhU'] = dhU                 
                    elif wavetype == "L":
                        out[key]['dbc'] = dbc
                        out[key]['dbU'] = dbU
                        out[key]['dhc'] = dhc
                        out[key]['dhU'] = dhU                    
                            
                    else: raise Exception('error: unknown wavetype')
    for key, subkey in out.items():
        if key is not 'model':
            for a, b in subkey.items():
                if a not in ["T","C","U"]:
                    len_array=len(out[key][a])
                    nb_T=len(out[key]["T"])
                    if nb_T*len(H) != len_array:
                        raise Exception('error: array size not consistent with number of periods')
                    out[key][a]=np.reshape(out[key][a],(len(H),nb_T),order='F')           
                        # to be done : concatenate all frequencies in out[key]
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
def initiate_surf96(mod96file, freq, nmod):
    
    script='rm %s %s %s' % ('dispersion_points*','sobs.d','srfker96.txt');
    execbash(script,os.getcwd())
    #----------------- prepare dispersion points file
    filename='dispersion_points.surf96'
    mod_array=range(nmod+1)
    with open(filename, 'w') as fid:
        for wavetype in 'LR':
            for mod in mod_array:
                for parameter in 'CU':
                    for i_f in range(len(freq)):
                        T=1./freq[i_f]
                        fid.write('SURF96 %s %s X %d %f %f %f\n' %
                        (wavetype, parameter, mod, T, 1.0, 1.0))
                    fid.write('\n')
    u_inc=0.005
    hask_inc=0.005
    
    #------------------- write bash script   
    script = """
    surf96 << END 
    {u_inc}
    {hask_inc} {hask_inc}
    1
    0
    {nmod}
    {nmod}
    0
    {nmod}
    {nmod}
    {mod96file}
    {filename}
    39
    END
    """.format(u_inc = u_inc, 
               hask_inc = hask_inc, 
               nmod = nmod+1, 
               mod96file = mod96file,
               filename = filename)
    script = "\n".join([_.strip() for _ in script.split('\n')])
    
    #------------------- execute bash script
    stdout, stderr = execbash(script,os.getcwd())
    expected_output = "sobs.d"
    if not os.path.exists(expected_output):
        raise Exception('output file %s not found, script failed \n%s' % (expected_output, stderr))

def launch_surf96(mod96file,freq,nmod, fig = None):
    #------------------- initiate
    initiate_surf96(mod96file, freq, nmod)
    #------------------- prepare script
    script = """
    surf96 1
    srfker96 > srfker96.txt
    surf96 39
    """
    script = "\n".join([_.strip() for _ in script.split('\n')])
    #------------------- launch script
    stdout, stderr = execbash(script,os.getcwd())
    expected_output = "srfker96.txt"
    if not os.path.exists(expected_output):
        raise Exception('output file %s not found, script failed \n%s' % (expected_output, stderr))
    #------------------- read result
    out=read_TXTout_surf96(expected_output)
    Z, H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS = readmod96(mod96file)
    out['model']['Vp']=VP
    out['model']['Vs']=VS
    out['model']['Rh']=RHO
    if fig is not None:

        ax2 = fig.add_subplot(122)
        ax1 = fig.add_subplot(121, sharey = ax2)

        #------------------
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
        ax1.set_ylabel('model depth (km)')

        #------------------
        key="%s%d" % (wavetype, nmod);
        vmax = abs(out[key]['dbc'][:, 2]).max()
        ax2.plot( out[key]['dbc'][:,2], out['model']["Z"])
        ax2.set_title("dc/dVs, T = %f, mode %d" % (1. / freq[2], nmod))
        ax2.set_ylabel('model depth (km)')
        ax2.set_xlabel('dc/dVs')
        
        gcf().savefig('output_surf96.png')
    return out
    
####################
def fegn17(mod96file, wavetype, nmod, freq, tmpdir, fig, cleanup = True):
    """ function to compute eigenfunctions and phase velocity sensitivity kernels
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
    assert freq.all > 0.0

    #------------------- create temporary directory
    if tmpdir is None: tmpdir = "/tmp/tmpdir_fegn17_%10d" % (np.random.rand() * 1e10)
    while os.path.isdir(tmpdir):
        #make sure tmpdir does not exists
        tmpdir += "_%10d" % (np.random.rand() * 1.0e10)
    os.mkdir(tmpdir)
    
    #------------------- write file with frequencies
    filename_farr=tmpdir+'/FARR.dat'
    with open(filename_farr, 'w') as fid:
        for i_f in range(len(freq)):
            f_value=freq[i_f]
            fid.write('%f\n' %(f_value))
    #------------------- write bash script   
    script = """
    #rm -f DISTFILE.dst sdisp96.??? s[l,r]egn96.??? S[L,R]DER.PLT S[R,L]DER.TXT FARR.dat

    cat << END > DISTFILE.dst 
    10. 0.125 256 -1.0 6.0
    END

    ln -s {mod96file} model.mod96
    
    ############################### prepare dispersion
    sprep96 -M model.mod96 -dfile DISTFILE.dst -NMOD {nmodmax} -{wavetype} -FARR FARR.dat

    ############################### run dispersion
    sdisp96

    ############################### compute eigenfunctions
    s{minuswavetype}egn96 -DE 

    ############################### ouput eigenfunctions
    sdpder96 -{wavetype} -TXT  # plot and ascii output

    ############################### clean up
    #rm -f sdisp96.dat DISTFILE.dst sdisp96.??? s[l,r]egn96.??? S[L,R]DER.PLT FARR.dat

    """.format(mod96file = os.path.realpath(mod96file), 
               nmodmax = nmod + 1, 
               wavetype = wavetype, 
               minuswavetype = wavetype.lower())
    script = "\n".join([_.strip() for _ in script.split('\n')]) #remove indentation from script

    #------------------- execute bash commands
    stdout, stderr = execbash(script, tmpdir = tmpdir)
    expected_output = "%s/S%sDER.TXT" % (tmpdir, wavetype)
    if not os.path.exists(expected_output):
        raise Exception('output file %s not found, script failed \n%s' % (expected_output, stderr))

    #------------------- read Herrmann's output
    out = read_TXTout_egn17(expected_output)
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
            ax2.plot(out[key]['UR'][:,len(freq)-1],  out['model']['Z'], label = "UR") #radial displacement
            ax2.plot(out[key]['UZ'][:,len(freq)-1],  out['model']['Z'], label = "UZ")
            ax2.plot(out[key]['TR'][:,len(freq)-1],  out['model']['Z'], label = "TR") #radial stress
            ax2.plot(out[key]['TZ'][:,len(freq)-1],  out['model']['Z'], label = "TZ")
        if wavetype == "L":
            ax2.plot(out[key]['UT'][:,len(freq)-1],  out['model']['Z'], label = "UT")
            ax2.plot(out[key]['TT'][:,len(freq)-1],  out['model']['Z'], label = "TT")        

        ax2.set_title("%s : T = %fs" % (key, out[key]["T"]))
        ax2.legend()
        ax1.set_ylabel('depth (km)')
        ax2.set_xlabel('eigenfunctions')
        ax2.grid(True)
        
        gcf().savefig('output_fegn17.png')
    #------------------- 
    return out
####################
def dfegn17(mod96file, wavetype, nmod, freq, pertu, tmpdir = None, fig = None, cleanup = True):
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
    Z, H, VP, VS, RHO, QP, QS, ETAP, ETAS, FREFP, FREFS = readmod96(mod96file)
    model0 = np.concatenate((H, VP, VS, RHO))
    nlayer = len(H)
    IH  = np.arange(nlayer) #index of thickness parameters
    IVP = np.arange(1 * nlayer, 2 * nlayer) #index of Vp parameters
    IVS = np.arange(2 * nlayer, 3 * nlayer) 
    IRH = np.arange(3 * nlayer, 4 * nlayer) 

    #compute eigenfunctions for starting model
    out0 = fegn17(mod96file, wavetype, nmod, freq, tmpdir = "%s/%s" % (tmpdir, "startingmodel"), fig = None, cleanup = True)

    #initiate kernels
    if wavetype == "R":
        DURDN = np.zeros((nlayer, len(model0),len(freq)), float) * np.nan
        DUZDN = np.zeros((nlayer, len(model0),len(freq)), float) * np.nan
        DTRDN = np.zeros((nlayer, len(model0),len(freq)), float) * np.nan
        DTZDN = np.zeros((nlayer, len(model0),len(freq)), float) * np.nan
    elif wavetype == "L":
        DUTDN = np.zeros((nlayer, len(model0),len(freq)), float) * np.nan
        DTTDN = np.zeros((nlayer, len(model0),len(freq)), float) * np.nan
    
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
                DURDN[:, nparam, :] = (out_n[key]['UR'] - out0[key]['UR']) / dm
                DUZDN[:, nparam, :] = (out_n[key]['UZ'] - out0[key]['UZ']) / dm
                DTRDN[:, nparam, :] = (out_n[key]['TR'] - out0[key]['TR']) / dm
                DTZDN[:, nparam, :] = (out_n[key]['TZ'] - out0[key]['TZ']) / dm
            elif wavetype == "L":
                DUTDN[:, nparam, :] = (out_n[key]['UT'] - out0[key]['UT']) / dm
                DTTDN[:, nparam, :] = (out_n[key]['TT'] - out0[key]['TT']) / dm

            #-------------------
            dout = {"Z" : out0['model']["Z"]}
            dout[key]={}
            if wavetype == "R":
                dout[key]['DURDH']  = DURDN[:, IH, :]
                dout[key]['DURDVP'] = DURDN[:, IVP, :]
                dout[key]['DURDVS'] = DURDN[:, IVS, :]
                dout[key]['DURDRH'] = DURDN[:, IRH, :]

                dout[key]['DUZDH']  = DUZDN[:, IH, :]
                dout[key]['DUZDVP'] = DUZDN[:, IVP, :]
                dout[key]['DUZDVS'] = DUZDN[:, IVS, :]
                dout[key]['DUZDRH'] = DUZDN[:, IRH, :]

                dout[key]['DTRDH']  = DTRDN[:, IH, :]
                dout[key]['DTRDVP'] = DTRDN[:, IVP, :]
                dout[key]['DTRDVS'] = DTRDN[:, IVS, :]
                dout[key]['DTRDRH'] = DTRDN[:, IRH, :]

                dout[key]['DTZDH']  = DTZDN[:, IH, :]
                dout[key]['DTZDVP'] = DTZDN[:, IVP, :]
                dout[key]['DTZDVS'] = DTZDN[:, IVS, :]
                dout[key]['DTZDRH'] = DTZDN[:, IRH, :]
                                                
            elif wavetype == "L":               
                dout[key]['DUTDH']  = DUTDN[:, IH, :]
                dout[key]['DUTDVP'] = DUTDN[:, IVP, :]
                dout[key]['DUTDVS'] = DUTDN[:, IVS, :]
                dout[key]['DUTDRH'] = DUTDN[:, IRH, :]
                                                
                dout[key]['DTTDH']  = DTTDN[:, IH, :]
                dout[key]['DTTDVP'] = DTTDN[:, IVP, :]
                dout[key]['DTTDVS'] = DTTDN[:, IVS, :]
                dout[key]['DTTDRH'] = DTTDN[:, IRH, :]

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
        key="%s%d" % (wavetype, nmod)
        if wavetype == "R":
            vmax = abs(dout[key]['DUZDVS'][:, :, 0]).max()
            ax3.plot(dout["Z"], out0[key]['UZ'][:,0], label = "UZ")
            Y = dout[key]['DUZDVS'][:, :, 0]
            Y = np.ma.masked_where(np.isnan(Y), Y)
            ax2.pcolormesh(dout["Z"], dout["Z"], Y, vmin = -vmax, vmax = vmax)
            ax2.set_title("DUZ/DVS, T = %f, mode %d" % (1. / freq[0], nmod))
            ax3.set_xlabel('eigenfunction depth (km)')
        elif wavetype == "L":
            vmax = abs(dout[key]['DUTDVS'][:, :, 0]).max()
            ax3.plot(dout["Z"], out0[key]['UT'][:,0], label = "UT")
            Y = dout[key]['DUTDVS'][:, :, 0]
            Y = np.ma.masked_where(np.isnan(Y), Y)
            ax2.pcolormesh(dout["Z"], dout["Z"], Y, vmin = -vmax, vmax = vmax)
            ax2.set_title("DUT/DVS, T = %f, mode %d" % (1. / freq[0], nmod))
            ax3.set_xlabel('eigenfunction depth (km)')

        ax3.xaxis.set_label_position("top")    
        ax3.legend()
        ax2.grid(True)
        ax3.grid(True)
        
        gcf().savefig('output_dfegn17.png')

    return dout

####################
if __name__ == "__main__":
    mod96file = "./example.mod96"
    wavetype = "R"
    nmod = 5
    freq = np.array([0.5, 1, 2])

    dout_eig17 = dfegn17(mod96file, wavetype, nmod, freq, pertu = 0.05, tmpdir = "./tmp", fig = plt.figure())
    #dout_eig17 = dfegn17(mod96file, wavetype, nmod, freq, pertu = 0.05)
    dout_surf96 = launch_surf96(mod96file, freq, nmod, fig=plt.figure())
    
    savemat('out_eig17', dout_eig17)
    savemat('out_surf96', dout_surf96)
    
    #gcf().show()
    
    raw_input('pause')
