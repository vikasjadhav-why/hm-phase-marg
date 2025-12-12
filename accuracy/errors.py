import numpy as np
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedHMPhase
from pycbc.inference.models.gaussian_noise import GaussianNoise
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
import pycbc.psd.analytical
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy.integrate import quad
import argparse

## Helper functions
import scipy.special


df = 1/24
psd = pycbc.psd.analytical.aLIGOZeroDetHighPower(length=int((1024/df)+1),
                                                 delta_f = df,
                                                 low_freq_cutoff = 20)

psd_network = { 'hl' :{'H1': psd,'L1': psd},
        'h': {'H1': psd},
         'l': {'L1': psd} }

det_network = {
    'hl' : {'H1': Detector('H1'),'L1': Detector('L1')},
    'h' : {'H1': Detector('H1')},
    'l' : {'L1': Detector('L1')}
}

low_freq_network = {
    'hl' : {'H1': 20,'L1': 20},
    'h' : {'H1':20},
    'l' : {'L1': 20}
}

##Helper functions

def generate_data(network,**params):
    data = {}
    for ifo in det_network[network]:
        hp,hc = get_fd_waveform(delta_f = df,**params)
        fp, fc = det_network[network][ifo].antenna_pattern(params['ra'],
                             params['dec'],
                             params['polarization'],
                             params['tc'])
        dt = det_network[network][ifo].time_delay_from_earth_center(params['ra'],
                                            params['dec'],
                                            params['tc'])
        tshift = dt + params['tc']
        data[ifo] = pycbc.waveform.utils.apply_fd_time_shift(fp*hp + fc*hc,
                                                             tshift)
        data[ifo].resize(len(psd))
    return data

def log_rel_err(log_approx, log_true):
    """
    Calculates absolute relative error between approx and true values
    |approx-true|/true

    """
    delta = log_approx - log_true
    if delta > 0 :
        return np.log10(np.exp(delta)-1)
    if delta < 0 :
        return np.log10(1- np.exp(delta))

def brute_marg(cls_instance,
               param_min,param_max,nsamples):
    samples = np.linspace(param_min,
                          param_max,
                          nsamples)
    loglr_samples = np.zeros(len(samples))
    for i in range(len(samples)):
        ##Currently hard coded to marginalise coa_phase
        cls_instance.update(coa_phase=samples[i])
        loglr_samples[i] = cls_instance.loglr
    marg_loglr = scipy.special.logsumexp(loglr_samples) - np.log(nsamples) # prior and int volume cancel out
    return marg_loglr

def create_static(param):
    static = param.copy()
    _ = static.pop('coa_phase')
    return static

def lr_surface(phi,cls_instance):
    cls_instance.update(coa_phase=phi)
    return np.exp(cls_instance.loglr)

def rvs_marg(cls_instance,
             param_min,param_max,nsamples):
    rvs = np.random.rand(nsamples)
    samples = param_min + rvs*(param_max - param_min)
    loglr_samples = np.zeros(len(samples))
    for i in range(nsamples):
        cls_instance.update(coa_phase=samples[i])
        loglr_samples[i] = cls_instance.loglr
    marg_loglr = scipy.special.logsumexp(loglr_samples) - np.log(nsamples) # prior and int volume cancel out
    return marg_loglr

def error_lr():
    parser = argparse.ArgumentParser()
    parser.add_argument("injections", type=str,help="path to injection hdf file")
    parser.add_argument("start",type=int,help="start index for parameters")
    parser.add_argument("end",type=int,help="end index for parameters")
    parser.add_argument("network",type=str,help=" 'h', 'l', 'hl")
    parser.add_argument("output_path",type=str,help="path for output hdf file")

    args = parser.parse_args()
    inj_file = args.injections
    injection_samples = h5py.File(inj_file,'r')

    variable_params = {'coa_phase'}
    network = args.network

    log_true = np.zeros(args.end-args.start)
    log_approx = np.zeros(args.end-args.start)

    error_approx = np.zeros(args.end-args.start)

    shm = np.zeros(args.end-args.start)

    for idx, i in enumerate(range(args.start,args.end)):
        test_param = {}
        for p in injection_samples.keys():
            test_param[p] = injection_samples[p][i]
        for sp in injection_samples.attrs['static_args']:
            test_param[sp] = injection_samples.attrs[sp]
        ## Change the modes to 22,33
        tp = test_param.copy()
        tp['mode_array'] = '22'
        ##Create marginalisation model
        marg_analytic = MarginalizedPhaseGaussianNoise(variable_params=variable_params,
                               data=generate_data(network=network,**tp),
                               low_frequency_cutoff=low_freq_network[network],
                               psds=psd_network[network],
                               static_params=create_static(tp))

        marg_appx = MarginalizedHMPhase(variable_params=variable_params,
                               data=generate_data(network=network,**tp),
                               low_frequency_cutoff=low_freq_network[network],
                               psds=psd_network[network],
                               static_params=create_static(tp))
        ##Update models to instantiate them
        marg_analytic.update(coa_phase = 0)
        marg_appx.update(coa_phase = 0)

        ## True value
        log_true[idx] = marg_analytic.loglr

        ## approx value
        log_approx[idx] = marg_appx.loglr
        ## snr 
        shm[idx] = np.abs(marg_appx.shm[2])

        ## Calculate the relative error in lr and save
        error_approx[idx] = log_rel_err(log_approx[idx],log_true[idx])

    out_file = h5py.File(args.output_path,'w')
    loglr = out_file.create_group('loglr')
    loglr.create_dataset('true',data=log_true)
    loglr.create_dataset('approx',data=log_approx)
    out_file.create_dataset('errors',data=error_approx)
    inner_products = out_file.create_group('shm')
    inner_products.create_dataset('sh2',data=shm)
    out_file.attrs['modes'] = '22'
    out_file.attrs['network'] = network
    out_file.close()

if __name__ =="__main__":
    error_lr()

