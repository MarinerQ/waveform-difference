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
import os
os.environ['LAL_DATA_PATH'] = "/Users/qianhu/lalsuite-extra-master/data/lalsimulation"


file_folder = '/home/daniel.williams/events/O3/o3a_final/releases/release-v3-sandbox/'
#file_folder = 'pesummary_file/'

files_GWTC2p1 = [file_folder+'IGWN-GWTC2p1-v2-GW170814_103043_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190916_200658_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW170818_022509_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190512_180714_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190720_000836_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW170104_101158_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190719_215514_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190707_093326_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190521_074359_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW151012_095443_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW170823_131358_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190620_030421_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190930_133541_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190426_190642_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190403_051519_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190814_211039_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190514_065416_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190924_021846_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190803_022701_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190513_205428_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190915_235702_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190708_232457_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190805_211137_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190727_060333_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW170608_020116_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190925_232845_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190828_063405_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190413_052954_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190910_112807_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190917_114630_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190421_213856_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190519_153544_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190517_055101_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190731_140936_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190425_081805_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW170729_185629_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190706_222641_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190630_185205_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW170809_082821_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190602_175927_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190929_012149_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190413_134308_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190926_050336_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190503_185404_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190527_092055_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190408_181802_PEDataRelease_mixed_cosmo.h5', #
file_folder+'IGWN-GWTC2p1-v2-GW190828_065509_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW151226_033853_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190725_174728_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW190412_053044_PEDataRelease_mixed_cosmo.h5', # error
file_folder+'IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5',
file_folder+'IGWN-GWTC2p1-v2-GW190701_203306_PEDataRelease_mixed_cosmo.h5'] #


def get_inj_paras(parameter_values, parameter_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl', 'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']):
    inj_paras = dict()
    for i in range(len(parameter_names)):
        inj_paras[parameter_names[i]] = parameter_values[i]
    return inj_paras

def get_inj_paras2(parameter_values, parameter_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','lambda_1', 'lambda_2', 'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']):
    inj_paras = dict()
    for i in range(len(parameter_names)):
        inj_paras[parameter_names[i]] = parameter_values[i]
    return inj_paras

##################### Time and phase shift for one detector #####################
def get_dtdphi_withift(h1,h2,det):

    psd = det.power_spectral_density_array
    f_array = det.frequency_array
    
    X_of_f = h1*h2.conjugate()/psd
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

def find_pe_file(eventname, pe_file_list):
    for pefilename in pe_file_list:
        if pefilename.find(eventname) != -1:
            return pefilename
    return ''

def calculate_deltasq_kernal(sample_ID, Deltasq_list, samples, waveform_generator_EOB, waveform_generator_IMR, ifos):

    #inj_para = get_inj_paras(samples[sample_ID])
    inj_para = get_inj_paras2(samples[sample_ID])
    h_IMR=waveform_generator_IMR.frequency_domain_strain(parameters=inj_para)
    #print(inj_para)
    inj_para.pop('lambda_1')
    inj_para.pop('lambda_2')
    h_EOB=waveform_generator_EOB.frequency_domain_strain(parameters=inj_para)
    
    resp_list_EOB = get_network_response(h_EOB, ifos, inj_para)
    resp_list_IMR = get_network_response(h_IMR, ifos, inj_para)
    
    ra = inj_para['ra']
    dec = inj_para['dec']
    geoc_time = inj_para['geocent_time']
    psi = inj_para['psi']
    
    temp_deltasq_list = []
    
    for i in range(len(ifos)):
        det = ifos[i]
        fp = det.antenna_response(ra,dec,geoc_time,psi,'plus')
        fc = det.antenna_response(ra,dec,geoc_time,psi,'cross')
        antenna_factor = (abs(fp) + abs(fc))**2
        temp_deltasq_list.append(antenna_factor)
        
    for i in range(len(ifos)):
    # Shift according to individual detector
        det = ifos[i]
        resp_list_IMR_shifted = get_shifted_h2(resp_list_EOB[i], resp_list_IMR[i], det)
        Deltasq = (det.inner_product_2(resp_list_EOB[i]-resp_list_IMR_shifted,
                                    resp_list_EOB[i]-resp_list_IMR_shifted)).real
        temp_deltasq_list.append(Deltasq)

    # Shift according to whole network
    resp_list_IMR_Netshifted = get_shifted_h2list(resp_list_EOB, resp_list_IMR, ifos)
    for i in range(len(ifos)):
        det = ifos[i]
        Deltasq_netshifted = (det.inner_product_2(resp_list_EOB[i]-resp_list_IMR_Netshifted[i],
                                    resp_list_EOB[i]-resp_list_IMR_Netshifted[i])).real
        temp_deltasq_list.append(Deltasq_netshifted)
    # Now the temp_deltasq_list is like [factorH, factorL, factorV, H1, L1, V1, H1net, L1net, V1net]
    for i in range(len(temp_deltasq_list)):
        Deltasq_list[sample_ID*3*len(ifos)+i] = temp_deltasq_list[i]
    #Deltasq_list.append(temp_deltasq_list)
    #Deltasq_list[sample_ID] = temp_deltasq_list



if __name__ == '__main__':
    time_start = time.time()

    input_paras = sys.argv
    input_event_name = str(sys.argv[1])
    input_sampling_frequency_times4k = float(sys.argv[2])
    fmin_EOB = float(sys.argv[3]) # input 0 uses default
    core_num = int(sys.argv[4])
    # example: nohup python wavediff_O3bNSBH_manual_mp.py GW200115 4 0 2 PHM &

    event_name = input_event_name  
    print("\n ------ Start analyzing "+ event_name + ' ------\n')

    # Read file
    
    if event_name=="GW170817":
        filaneme170817 = "/home/qian.hu/waveform_diff/manualrun3/cosmo_reweight.h5"
        f = h5py.File(filaneme170817,'r')
        info_IMR = f['IMRPhenomPv2NRT_lowSpin']
        postsample_IMR = info_IMR['posterior_samples']
        approx_IMR = "IMRPhenomPv2_NRTidal"
        approx_EOB = "SEOBNRv4T_surrogate"
        para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','lambda_1','lambda_2',  # 8+2 intrinsic
             'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic
    elif event_name=="GW190425_081805":
        filename = find_pe_file(input_event_name, files_GWTC2p1)
        f = h5py.File(filename,'r')
        info_IMR = f['C01:IMRPhenomPv2_NRTidal:LowSpin']
        postsample_IMR = info_IMR['posterior_samples']
        approx_IMR = "IMRPhenomPv2_NRTidal"
        approx_EOB = "IMRPhenomXPHM"
        para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','lambda_1','lambda_2',  # 8+2 intrinsic
             'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic

    # Read posterior samples. 
    # t_c is set as constant as only IMR gives its estimation. 
    # Other paras are from mixed samples. 
    if event_name=="GW170817":
        geocent_time_est = 1187008882.4
    elif event_name=="GW190425_081805":
        geocent_time_sample = info_IMR['posterior_samples']['geocent_time']
        geocent_time_est = np.mean(geocent_time_sample)
        #geocent_time_est = float(list(info_IMR['config_file']['config']['trigger-time'])[0])

    #postsample_mixed = info_mixed['posterior_samples']
    
    samples = np.zeros(shape=(len(postsample_IMR['chirp_mass']), len(para_names)) )
    for i in range(len(para_names)-1):
        samples[:,i] = postsample_IMR[para_names[i]] 
    samples[:,-1] = np.zeros(len(postsample_IMR['chirp_mass']) ) + geocent_time_est

    # Define waveform generators

    duration = 64

    sampling_frequency = 4096.* input_sampling_frequency_times4k
    if event_name=="GW170817":
        fref_EOB=100
        fmin_EOB=20
    elif event_name=="GW190425_081805":
        fref_EOB = 20
        fmin_EOB = 20


    waveform_arguments_IMR = dict(waveform_approximant=approx_IMR,
                            reference_frequency=fref_EOB, minimum_frequency=fmin_EOB)  

    waveform_arguments_EOB = dict(waveform_approximant=approx_EOB,
                            reference_frequency=fref_EOB, minimum_frequency=fmin_EOB)


    waveform_generator_IMR = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments_IMR)

    waveform_generator_EOB = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
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
    #Nsample = 4
    
    # The array to save
    # [factorsqH, .. , .. , DeltasqH, .. , .. , DeltasqH_netshifted, .. , ..]
    #Deltasq_list = np.zeros(shape=(Nsample,3*len(ifos)))
    manager = multiprocessing.Manager()
    Deltasq_list = manager.Array('d', range(Nsample* 3*len(ifos) ))

    partial_work = partial(calculate_deltasq_kernal, Deltasq_list=Deltasq_list, samples=samples, waveform_generator_EOB=waveform_generator_EOB, waveform_generator_IMR=waveform_generator_IMR, ifos=ifos)


    with Pool(core_num) as p:
        p.map(partial_work, range(Nsample) )
        #p.apply_async(partial_work, range(Nsample) ) 
    
    Deltasq_list_reshaped = np.reshape(Deltasq_list, (Nsample, 3*len(ifos)))
    # Calculation done. Save Deltasq_list
    file_suffix = detctor_name_list[0] + detctor_name_list[1]
    if len(detctor_name_list)==3:
        file_suffix += detctor_name_list[2]
    savefilename = "Deltasq_" + event_name + "_" + file_suffix + ".txt"
    save_folder = "result_GWTC2p1BNS/"
    np.savetxt(save_folder + savefilename, Deltasq_list_reshaped)

    time_end = time.time()
    timecost = time_end-time_start
    print("\n------ "+event_name,'calculation done (cost ' +str(int(timecost))+ ' s). File saved to <'+ save_folder + savefilename + '>. ------\n')