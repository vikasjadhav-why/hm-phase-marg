import numpy as np
from pycbc.workflow import WorkflowConfigParser
from pycbc.inference.models.gaussian_noise import GaussianNoise
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedHMPhase
from pycbc.inference.models import read_from_config
import h5py
import argparse

def create_parser(
        injection_file,variable_params_file,data_file,model_file,
        mode_array,**kwargs):
    """ 
    Create a Workflowconfigparser that updates static_params by reading it from 
    the injection file. 
    Inputs : 
    "injection file" : an hdf file containing only one injection
    "variable_params_file" : a config file containing section of variable_params
    and must include the priors of each parameter.
    "data_file" : a config file containing common data settings.The trigger-time and 
    injection-file options will be updated using the provided injection.
    "model_file" : config file with the model settings.
    "mode_array" : mode array to use in the signal model.
    "kwargs" : additional options for the [model] section.
    Output : 
    "cp" : a workflow config parser
    """
    cp = WorkflowConfigParser([data_file,variable_params_file,model_file])
    
    
    ## Read the injection file to gather all the params
    inj = h5py.File(injection_file,'r')
    all_params = {}
    for p in inj.keys():
        all_params[p] = inj[p][0]
    for sp in inj.attrs['static_args']:
        all_params[sp] = inj.attrs[sp]
    static_params = all_params.copy()
    ## Remove the params included in the variable params
    for vp in cp['variable_params'].keys():
        _ = static_params.pop(vp)
    
    ## Add the mode array to be used in the model
    static_params['mode_array'] = mode_array
    ## Create and add options to static_params section
    cp.add_section('static_params')
    for sp in static_params:
        cp.add_options_to_section('static_params',
                                  [(sp,str(static_params[sp]))])
    
    ## Update options to data section
    cp.add_options_to_section('data',[('trigger-time',str(static_params['tc'])),
                                      ('injection-file',injection_file)])
    for key, value in kwargs.items():
        cp.add_options_to_section('model', [(key, str(value))])
    return cp

def store_results(out_file, idx, loglr, peaks, shm, injection_folder):
    """
    Store results for injection idx into the output HDF file.
    Each injection gets its own group: 'injection_{idx}'.
    Datasets within the group mirror the variable names:
      - loglr : scalar float
      - peaks  : numpy array
      - shm/<key> : one dataset per key in the shm dictionary
    The root of the file has an attribute 'injection_folder' with the source path.
    """
    with h5py.File(out_file, 'a') as f:
        # Write the injection folder as a root-level attribute
        # Only set it on the first write to avoid redundant updates
        if 'injection_folder' not in f.attrs:
            f.attrs['injection_folder'] = injection_folder

        grp = f.create_group(f'injection_{idx}')
        grp.create_dataset('loglr', data=loglr)
        grp.create_dataset('peaks', data=peaks)
        shm_grp = grp.create_group('shm')
        for key, value in shm.items():
            shm_grp.create_dataset(key, data=value)


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("injection_folder",help="path to the folder containing injection files")
    parser.add_argument("variable_params_file",help='var_par file')
    parser.add_argument("data_file", help = "path to data config file")
    parser.add_argument("model_file", help = "path to model config file")
    parser.add_argument("mode_array", help="mode array for signal model")
    parser.add_argument("output_file", help="path for outut file")
    parser.add_argument("start", help="index of injection to start")
    parser.add_argument("end", help= "index of injection to end")
    parser.add_argument("--kwargs", nargs='*', default=[],
                        metavar="KEY=VALUE",
                        help="additional keyword arguments for the model section")
    
    args = parser.parse_args()
    extra_kwargs = parse_kwargs(args.kwargs)
    inj_path = args.injection_folder
    var_par = args.variable_params_file
    data_file = args.data_file
    model_file = args.model_file
    mode_array = args.mode_array
    start = args.start
    end = args.end
    out_file = args.output_file

    for idx, i in enumerate(range(start,end)):
        cp = create_parser(injection_file=f'{inj_path}/injection_{i}.hdf',
                       variable_params_file=var_par,
                       data_file=data_file,
                       model_file=model_file,
                       mode_array=mode_array,
                       **extra_kwargs)
        model = read_from_config(cp)
        model.update(coa_phase = 0)
        loglr = model.loglr  ## numpy float
        peaks = model.peaks  ## numpy array
        shm = model.shm      ## dictionary
        
        store_results(out_file,i,loglr,peaks,shm,inj_path)

if __name__=="__main__":
    run_model()

    


