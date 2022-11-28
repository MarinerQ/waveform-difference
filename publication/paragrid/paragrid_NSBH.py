import h5py
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (14,8.6)
import numpy as np
import bilby
import time 
import sys
from multiprocessing import Pool
import multiprocessing
from functools import partial
from bilby.gw import conversion
from pesummary.gw.conversions import spins as pespin
#import os
#os.environ['LAL_DATA_PATH'] = "/Users/qianhu/lalsuite-extra-master/data/lalsimulation"


##################### Time and phase shift for one detector #####################
def get_dtdphi_withift(h1,h2,det):

    psd = det.power_spectral_density_array
    f_array = det.frequency_array
    
    X_of_f = h1*h2.conjugate()/psd
    
    # zero padding
    add_zero = np.zeros(int(31*len(X_of_f)))
    X_of_f = np.append(X_of_f,add_zero)
    
    X_of_t = np.fft.ifft(X_of_f)
    
    timelength = 1/(f_array[1]-f_array[0])
    t = np.linspace(-timelength/2,timelength/2,len(X_of_t))
    X_shifted = np.roll(X_of_t,len(X_of_t)//2)

    jmax = np.argmax( abs(X_shifted) )
    deltat = t[jmax]
    phase1 = 2*np.pi*f_array*deltat
    
    inner_product = det.inner_product_2(h1.conjugate(), h2.conjugate()*np.exp(1j*phase1) )
    
    deltaphi = -np.angle(inner_product)
    #phase2 = deltaphi
    
    return deltat,deltaphi


def get_shifted_h2(h1,h2,det):
    '''
    Return the h2*exp(-i*phase_shift), i.e. h2* exp -i*(2\pi f \Delta t + \Delta \phi)
    '''
    deltat,deltaphi = get_dtdphi_withift(h1,h2,det)
    f_array = det.frequency_array
    exp_phase = np.exp(-1j*(2*np.pi*f_array*deltat + deltaphi) )
    return h2*exp_phase


##################### Time and phase shift for detector network #####################


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
    add_zero = np.zeros(int(31*len(X_of_f)))
    X_of_f = np.append(X_of_f,add_zero)
    
    X_of_t = np.fft.ifft(X_of_f)
    
    timelength = 1/(f_array[1]-f_array[0])  # assume 3 det have the same freq array, so just use the last f_array of the loop
    t = np.linspace(-timelength/2,timelength/2,len(X_of_t))
    X_shifted = np.roll(X_of_t,len(X_of_t)//2)

    jmax = np.argmax( abs(X_shifted) )
    deltat = t[jmax]
    phase1 = 2*np.pi*f_array*deltat
    
    inner_product=0
    for ii in range(len(ifos)):
        inner_product += det.inner_product_2(h1_list[ii].conjugate(),
                                             h2_list[ii].conjugate()*np.exp(1j*phase1) )
    
    deltaphi = -np.angle(inner_product)    
    return deltat,deltaphi
    
    
def get_shifted_h2list(h1_list, h2_list, ifos):
    deltat,deltaphi = get_network_dtdphi(h1_list, h2_list, ifos)
    f_array = ifos[0].frequency_array
    exp_phase = np.exp(-1j*(2*np.pi*f_array*deltat + deltaphi) )

    return (h2_list*exp_phase).tolist()

def get_inj_paras(parameter_values, parameter_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','lambda_1','lambda_2','theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']):
        inj_paras = dict()
        for i in range(len(parameter_names)):
            inj_paras[parameter_names[i]] = parameter_values[i]
        return inj_paras
    
def calculate_deltasq_kernal(sample_ID, Deltasq_list, samples, waveform_generator1, waveform_generator2, ifos):

    inj_para = get_inj_paras(samples[sample_ID])
    #print(inj_para)
    h_EOB=waveform_generator1.frequency_domain_strain(parameters=inj_para)['plus']
    h_IMR=waveform_generator2.frequency_domain_strain(parameters=inj_para)['plus']
    
    temp_deltasq_list = []
    
        
    # Shift according to individual detector
    det = ifos[0]
    resp_list_IMR_shifted = get_shifted_h2(h_EOB, h_IMR, det)
    Deltasq = (det.inner_product_2(h_EOB-resp_list_IMR_shifted,
                                    h_EOB-resp_list_IMR_shifted)).real
    opt_snr_EOB = np.sqrt(det.inner_product_2(h_EOB,h_EOB).real)
    opt_snr_IMR = np.sqrt(det.inner_product_2(h_IMR,h_IMR).real)

    temp_deltasq_list.append(Deltasq)
    temp_deltasq_list.append(opt_snr_EOB)
    temp_deltasq_list.append(opt_snr_IMR)


    # Now the temp_deltasq_list is like [Deltasq, rho1, rho2]
    for i in range(len(temp_deltasq_list)):
        Deltasq_list[sample_ID*3*len(ifos)+i] = temp_deltasq_list[i]
    #Deltasq_list.append(temp_deltasq_list)
    #Deltasq_list[sample_ID] = temp_deltasq_list



if __name__ == '__main__':
    time_start = time.time()
    input_sampling_frequency_times4k = float(sys.argv[1])
    core_num = int(sys.argv[2])

    print("\n ------ Start generating parameter sets ------\n")

    Ngrid1 = 120
    Ngrid2 = 150
    Nsample = Ngrid1*Ngrid2

    q_grid = np.logspace(np.log10(0.02), np.log10(0.25), Ngrid1)
    lambda2_grid = np.linspace(0,5000,Ngrid2)

    para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','lambda_1','lambda_2',  # 8+1 intrinsic
                'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic


    mass_ratio = np.reshape(np.meshgrid(q_grid, lambda2_grid)[0], (Nsample,))
    lambda_2 = np.reshape(np.meshgrid(q_grid, lambda2_grid)[1], (Nsample,))

    mass_1 = 1.4/mass_ratio  # q<=1
    mass_2 = np.zeros(Nsample) + 1.4  
    chirp_mass = conversion.component_masses_to_chirp_mass(mass_1,mass_2)
    
    
    a_add = 0.5
    a_1 = np.zeros(Nsample) + a_add
    a_2 = np.zeros(Nsample) + a_add
    tilt_1 = np.zeros(Nsample)
    tilt_2 = np.zeros(Nsample)
    phi_12 = np.zeros(Nsample)
    phi_jl = np.zeros(Nsample)
    lambda_1 = np.zeros(Nsample)

    theta_jn = np.zeros(Nsample)
    psi = np.zeros(Nsample) 
    phase = np.zeros(Nsample) 
    ra = np.zeros(Nsample) 
    dec = np.zeros(Nsample)
    luminosity_distance = np.zeros(Nsample) + 1000
    geocent_time = np.zeros(Nsample) + 1187008882.4

    para_list = [chirp_mass,mass_ratio,a_1,a_2,tilt_1,tilt_2,phi_12,phi_jl,lambda_1,lambda_2,
                theta_jn, psi, phase, ra, dec, luminosity_distance, geocent_time]

    samples = np.zeros(shape=(Nsample,len(para_list)) )

    for i in range(len(para_list)):
        samples[:,i] = para_list[i] 


    duration = 32. 
    sampling_frequency = 4096.*input_sampling_frequency_times4k

    waveform_arguments1 = dict(waveform_approximant="IMRPhenomNSBH",
                            reference_frequency=50., minimum_frequency=30)

    waveform_arguments2 = dict(waveform_approximant="SEOBNRv4_ROM_NRTidalv2_NSBH",
                            reference_frequency=50., minimum_frequency=30)


    waveform_generator1 = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments1)

    waveform_generator2 = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments2)

    detctor_name_list = ['H1']
    ifos = bilby.gw.detector.InterferometerList(detctor_name_list)

    # set detector paramaters
    for i in range(len(ifos)):
        det = ifos[i]
        det.duration = duration
        det.sampling_frequency=sampling_frequency
        det.minimum_frequency = 30
    
    # Calculate Deltasq
    
    # The array to save
    # [factorsqH, .. , .. , DeltasqH, .. , .. , DeltasqH_netshifted, .. , ..]
    #Deltasq_list = np.zeros(shape=(Nsample,3*len(ifos)))
    manager = multiprocessing.Manager()
    Deltasq_list = manager.Array('d', range(Nsample* 3*len(ifos) ))

    partial_work = partial(calculate_deltasq_kernal, Deltasq_list=Deltasq_list, samples=samples, waveform_generator1=waveform_generator1, waveform_generator2=waveform_generator2, ifos=ifos)


    with Pool(core_num) as p:
        p.map(partial_work, range(Nsample) )
        #p.apply_async(partial_work, range(Nsample) ) 
    
    Deltasq_list_reshaped = np.reshape(Deltasq_list, (Nsample, 3*len(ifos)))
    # Calculation done. Save Deltasq_list
    
    #savefilename = "paragrid_NSBH_H1" + ".txt"
    savefilename = "paragrid_NSBHPAD_a5_H1" + ".txt"
    save_folder = "grid_output/"
    np.savetxt(save_folder + savefilename, Deltasq_list_reshaped)

    time_end = time.time()
    timecost = time_end-time_start
    print('\n------ Calculation done (cost ' +str(int(timecost))+ ' s). File saved to <'+ save_folder + savefilename + '>. ------\n')