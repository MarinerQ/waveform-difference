import h5py
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (14,8.6)
import numpy as np
import bilby
from multiprocessing import Pool
import multiprocessing
import sys
from functools import partial
import time
from gstlal import chirptime
from bilby.gw import utils as gwutils
import argparse

def get_inj_paras(parameter_values, parameter_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl', 'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']):
    inj_paras = dict()
    for i in range(len(parameter_names)):
        inj_paras[parameter_names[i]] = parameter_values[i]
    return inj_paras

def get_fmin(Mtot):
    '''
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_spin_prec_e_o_bv4_p_8c_source.html
    int XLALEOBHighestInitialFreq
    '''
    a = np.power(10.5,-1.5) / np.pi
    msun_geo = 4.925491e-6
    a /= msun_geo
    return a/Mtot

def get_chirptime(postsample,fstart):
    GMsun = 1.32712442099e20 # heliocentric gravitational constant, m^2 s^-2
    G = 6.67384e-11 # Newtonian constant of gravitation, m^3 kg^-1 s^-2
    c = 299792458 # speed of light in vacuum (exact), m s^-1
    Msun = GMsun / G # solar mass, kg
    m1 = postsample["mass_1"]
    m2 = postsample["mass_2"]
    a1 = postsample["a_1"]
    a2 = postsample["a_2"]
    tt = []
    for i in range(len(m1)):
        tt.append(chirptime.imr_time(fstart, m1[i]*Msun, m2[i]*Msun, a1[i] ,a2[i]))
    return max(tt)




def get_network_response(hdict, ifos, injection_parameters):
    h_list = []
    for det in ifos:
        resp = det.get_detector_response(waveform_polarizations=hdict, parameters=injection_parameters)
        h_list.append(resp)
    return h_list

def get_network_dtdphi(h1_list, h2_list, ifos):
    '''
    h1_list: [hEOB_H1, hEOB_L1, hEOB_V1] if it's a 3-det event
    h2_list: [hIMR_H1, hIMR_L1, hIMR_V1] if it's a 3-det event
    '''
    X_of_f_list = []
    for i in range(len(ifos)):
        det=ifos[i]
        psd = det.power_spectral_density_array
        f_array = det.frequency_array
        h1 = h1_list[i]
        h2 = h2_list[i]
        X_of_f_list.append(h1*h2.conjugate()/psd)
    X_of_f = sum(X_of_f_list)  # X_net(f) = X_H(f) + X_L(f) + X_V(f)
    
    # zero padding
    add_zero = np.zeros(int(63*len(X_of_f)))
    X_of_f = np.append(X_of_f,add_zero)
    
    X_of_t = np.fft.ifft(X_of_f)
    
    timelength = 1/(f_array[1]-f_array[0])  # assume 3 det have the same freq array, so just use the last f_array of the loop
    t = np.linspace(-timelength/2,timelength/2,len(X_of_t))
    X_shifted = np.roll(X_of_t,len(X_of_t)//2)

    jmax = np.argmax( abs(X_shifted) )
    deltat = t[jmax]
    phase1 = 2*np.pi*f_array*deltat
    #print('len(phase1)', len(phase1))
    inner_product=0
    for ii in range(len(ifos)):
        det = ifos[ii]
        freq_mask = det.strain_data.frequency_mask
        #inner_product += det.inner_product_2(h1_list[ii].conjugate(),
        #                                     h2_list[ii].conjugate()*np.exp(1j*phase1) )
        inner_product += gwutils.noise_weighted_inner_product(
                            aa=h1_list[ii].conjugate()[freq_mask],
                            bb=(h2_list[ii].conjugate() * np.exp(1j*phase1))[freq_mask],
                            power_spectral_density=det.power_spectral_density_array[freq_mask],
                            duration=det.strain_data.duration)
    
    deltaphi = -np.angle(inner_product)    
    return deltat,deltaphi
  
def get_shifted_h2list(h1_list, h2_list, ifos):
    deltat,deltaphi = get_network_dtdphi(h1_list, h2_list, ifos)
    f_array = ifos[0].frequency_array
    exp_phase = np.exp(-1j*(2*np.pi*f_array*deltat + deltaphi) )

    return (h2_list*exp_phase).tolist()


def calculate_deltasq_kernel(sample_ID, Deltasq_list, samples, waveform_generator_EOB, waveform_generator_IMR, ifos):

    inj_para = get_inj_paras(samples[sample_ID])

    h_EOB=waveform_generator_EOB.frequency_domain_strain(parameters=inj_para)
    h_IMR=waveform_generator_IMR.frequency_domain_strain(parameters=inj_para)
    resp_list_EOB = get_network_response(h_EOB, ifos, inj_para)
    resp_list_IMR = get_network_response(h_IMR, ifos, inj_para)
    
    ra = inj_para['ra']
    dec = inj_para['dec']
    geoc_time = inj_para['geocent_time']
    psi = inj_para['psi']
    
    temp_deltasq_list = []
    
    
    # Shift according to whole network
    resp_list_IMR_Netshifted = get_shifted_h2list(resp_list_EOB, resp_list_IMR, ifos)
    for i in range(len(ifos)):
        det = ifos[i]
        #Deltasq_netshifted = (det.inner_product_2(resp_list_EOB[i]-resp_list_IMR_Netshifted[i],
        #                            resp_list_EOB[i]-resp_list_IMR_Netshifted[i])).real
        dh = resp_list_EOB[i]-resp_list_IMR_Netshifted[i]
        Deltasq_netshifted = gwutils.noise_weighted_inner_product(
                            aa=dh[det.strain_data.frequency_mask],
                            bb=dh[det.strain_data.frequency_mask],
                            power_spectral_density=det.power_spectral_density_array[det.strain_data.frequency_mask],
                            duration=det.strain_data.duration)
        temp_deltasq_list.append(Deltasq_netshifted.real)

    # Now the temp_deltasq_list is like [H1net, L1net, V1net]
    for i in range(len(temp_deltasq_list)):
        Deltasq_list[sample_ID*len(ifos)+i] = temp_deltasq_list[i]


# conda activate igwn-py39

if __name__ == '__main__':
    time_start = time.time()
    parser = argparse.ArgumentParser(description='Arguments for calculating waveforms and waveform differences')
    parser.add_argument('--filename', type=str,
                    help='Path to the PEsummary file')
    parser.add_argument('--f_low', type=float,
                    help='Lower frequency for generating waveforms')
    parser.add_argument('--f_ref', type=float,
                    help='Reference frequency for generating waveforms')
    parser.add_argument('--duration', type=float,
                    help='Duration for generating waveforms')
    parser.add_argument('--ncpu', type=int,
                    help='Number of CPUs you want to use')

    args = parser.parse_args()

    filename = args.filename
    fmin = args.f_low
    fref = args.f_ref
    duration = args.duration
    sampling_frequency = 4096

    core_num = args.ncpu

    # Read file
    
    f = h5py.File(filename,'r')
    info_IMR = f['C01:IMRPhenomXPHM']
    info_EOB = f['C01:SEOBNRv4PHM']
    info_mixed = f['C01:Mixed']
    postsample_mixed = info_mixed['posterior_samples']

    # Read posterior samples. 
    # t_c is set as constant as only IMR gives its estimation. 
    # Other paras are from mixed samples. 
    geocent_time_sample = f['C01:IMRPhenomXPHM']['posterior_samples']['geocent_time']
    geocent_time_est = np.mean(geocent_time_sample)
    #geocent_time_est = float(list(info_IMR['config_file']['config']['trigger-time'])[0])

    postsample_mixed = info_mixed['posterior_samples']
    para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',  # 8 intrinsic
             'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic
    samples = np.zeros(shape=(len(postsample_mixed['chirp_mass']), len(para_names)) )
    for i in range(len(para_names)-1):
        samples[:,i] = postsample_mixed[para_names[i]] 
    samples[:,-1] = np.zeros(len(postsample_mixed['chirp_mass']) ) + geocent_time_est


    # Sampling freq
    
    print("\nSampling_frequency = {}Hz.".format(sampling_frequency))

    waveform_arguments_IMR = dict(waveform_approximant='IMRPhenomXPHM',
                            reference_frequency=fref, minimum_frequency=fmin)  

    waveform_arguments_EOB = dict(waveform_approximant='SEOBNRv4PHM',
                            reference_frequency=fref, minimum_frequency=fmin)


    waveform_generator_IMR = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments_IMR)

    waveform_generator_EOB = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments_EOB)

    # Define detectors
    detctor_name_list = list(info_IMR['psds'])
    ifos = bilby.gw.detector.InterferometerList(detctor_name_list)

    # Set detector paramaters
    for i in range(len(ifos)):
        det = ifos[i]
        det_name = detctor_name_list[i]
        
        det.duration = duration
        det.sampling_frequency=sampling_frequency
        
        real_PSD = info_IMR['psds'][det_name]
        det.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=np.linspace(0,round(real_PSD[:,0][-1]),len(real_PSD[:,0])),
        psd_array=real_PSD[:,1])
        
        #det.frequency_mask[np.where(det.frequency_array==1024)[0][0]+1 : -1] = False
    
    # Calculate Deltasq
    Nsample = samples.shape[0]
    Nsample = 5
    
    # The array to save
    # [factorsqH, .. , .. , DeltasqH, .. , .. , DeltasqH_netshifted, .. , ..]
    #Deltasq_list = np.zeros(shape=(Nsample,3*len(ifos)))
    manager = multiprocessing.Manager()
    Ncol = len(ifos)
    Deltasq_list = manager.Array('d', range(Nsample*Ncol ))

    partial_work = partial(calculate_deltasq_kernel, Deltasq_list=Deltasq_list, samples=samples, waveform_generator_EOB=waveform_generator_EOB, waveform_generator_IMR=waveform_generator_IMR, ifos=ifos)


    with Pool(core_num) as p:
        p.map(partial_work, range(Nsample) )
        #p.apply_async(partial_work, range(Nsample) ) 
    
    Deltasq_list_reshaped = np.reshape(Deltasq_list, (Nsample, len(ifos)))
    # Calculation done. Save Deltasq_list
    #file_suffix = detctor_name_list[0] + detctor_name_list[1]
    #if len(detctor_name_list)==3:
    #    file_suffix += detctor_name_list[2]
    #savefilename = "Deltasq_" + event_name + "_" + file_suffix + ".txt"
    #save_folder = "result_O3bBBH/"
    Delta_net = sum(delta_det for delta_det in Deltasq_list_reshaped.T)
    Delta_net = np.sqrt(Delta_net / len(ifos))

    savefilename = 'example_output_2.txt'
    np.savetxt(savefilename, Delta_net)

    time_end = time.time()
    timecost = time_end-time_start
    print('Timecost: {}s'.format(timecost))
    #print("\n------ "+event_name,'calculation done (cost ' +str(int(timecost))+ ' s). File saved to <'+ save_folder + savefilename + '>. ------\n')
