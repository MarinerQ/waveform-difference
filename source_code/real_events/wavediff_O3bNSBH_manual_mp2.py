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
#os.environ['LAL_DATA_PATH'] = "/Users/qianhu/lalsuite-extra-master/data/lalsimulation"


file_folder = '/home/qian.hu/gwosc_PEresult/O3b/'
#file_folder = '/Users/qianhu/Documents/Glasgow/research/waveform_diff/pesummary_file/'
'''
files_O3b = [file_folder+"IGWN-GWTC3p0-v1-GW191103_012549_PEDataRelease_mixed_cosmo.h5",
file_folder+"IGWN-GWTC3p0-v1-GW200129_065458_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191105_143521_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200202_154313_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191109_010717_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200208_130117_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191113_071753_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200208_222617_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191126_115259_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200209_085452_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191127_050227_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200210_092255_PEDataRelease_mixed_cosmo.h5", # NSBH or low mass BBH
file_folder+"IGWN-GWTC3p0-v1-GW191129_134029_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200216_220804_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191204_110529_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200219_094415_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191204_171526_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200220_061928_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191215_223052_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200220_124850_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191216_213338_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200224_222234_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191219_163120_PEDataRelease_mixed_cosmo.h5", # NSBH
file_folder+"IGWN-GWTC3p0-v1-GW200225_060421_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191222_033537_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200302_015811_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW191230_180458_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200306_093714_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200105_162426_PEDataRelease_mixed_cosmo.h5", # NSBH
file_folder+"IGWN-GWTC3p0-v1-GW200308_173609_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200112_155838_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200311_115853_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200115_042309_PEDataRelease_mixed_cosmo.h5", # NSBH
file_folder+"IGWN-GWTC3p0-v1-GW200316_215756_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200128_022011_PEDataRelease_mixed_cosmo.h5", 
file_folder+"IGWN-GWTC3p0-v1-GW200322_091133_PEDataRelease_mixed_cosmo.h5"] # 35+1 in total. Run 32
'''
files_O3b = [file_folder+"IGWN-GWTC3p0-v1-GW200210_092255_PEDataRelease_mixed_cosmo.h5",
file_folder+"IGWN-GWTC3p0-v1-GW191219_163120_PEDataRelease_mixed_cosmo.h5",
file_folder+"IGWN-GWTC3p0-v1-GW200105_162426_PEDataRelease_mixed_cosmo.h5",
file_folder+"IGWN-GWTC3p0-v1-GW200115_042309_PEDataRelease_mixed_cosmo.h5"]


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
    #print(inj_para)
    h_EOB=waveform_generator_EOB.frequency_domain_strain(parameters=inj_para)
    h_IMR=waveform_generator_IMR.frequency_domain_strain(parameters=inj_para)
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
    flag = str(sys.argv[5])  # PHM, NSBHLOW, NSBHHIGH
    # example: nohup python wavediff_O3bNSBH_manual_mp.py GW200115 4 0 2 PHM &

    event_name = input_event_name  
    print("\n ------ Start analyzing "+ event_name + ' ------\n')

    # Read file
    filename = find_pe_file(input_event_name, files_O3b)
    f = h5py.File(filename,'r')
    if flag=="PHM":
        info_IMR = f['C01:IMRPhenomXPHM:LowSpin']
        info_EOB = f['C01:SEOBNRv4PHM']
        info_mixed = f['C01:Mixed']
        postsample_mixed = info_mixed['posterior_samples']
        approx_IMR = "IMRPhenomXPHM"
        approx_EOB = "SEOBNRv4PHM"
        para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',  # 8 intrinsic
             'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic
    if flag=="NSBHLOW":
        info_IMR = f['C01:IMRPhenomNSBH:LowSpin']
        info_EOB = f['C01:SEOBNRv4_ROM_NRTidalv2_NSBH:LowSpin']
        #info_mixed = f['C01:Mixed:NSBH:LowSpin']
        #postsample_mixed = info_mixed['posterior_samples']
        postsample_IMR = info_IMR['posterior_samples']
        postsample_EOB = info_EOB['posterior_samples']
        approx_IMR = "IMRPhenomNSBH"
        approx_EOB = "SEOBNRv4_ROM_NRTidalv2_NSBH"
        para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','lambda_1','lambda_2',  # 8+2 intrinsic
             'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic
    elif flag=="NSBHHIGH":
        info_IMR = f['C01:IMRPhenomNSBH:HighSpin']
        info_EOB = f['C01:SEOBNRv4_ROM_NRTidalv2_NSBH:HighSpin']
        postsample_IMR = info_IMR['posterior_samples']
        postsample_EOB = info_EOB['posterior_samples']
        approx_IMR = "IMRPhenomNSBH"
        approx_EOB = "SEOBNRv4_ROM_NRTidalv2_NSBH"
        para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','lambda_1','lambda_2',  # 8+2 intrinsic
             'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic



    # Read posterior samples. 
    # t_c is set as constant as only IMR gives its estimation. 
    # Other paras are from mixed samples. 
    #geocent_time_sample = f['C01:IMRPhenomXPHM']['posterior_samples']['geocent_time']
    #geocent_time_est = np.mean(geocent_time_sample)
    geocent_time_est = float(list(info_IMR['config_file']['config']['trigger-time'])[0])

    #postsample_mixed = info_mixed['posterior_samples']
    
    Ntake = 5000
    samples = np.zeros(shape=(2*Ntake, len(para_names)) )
    for i in range(len(para_names)-1):
        sp1 = postsample_IMR[para_names[i]][0:Ntake]
        sp2 = postsample_EOB[para_names[i]][0:Ntake]
        samples[:,i] = np.append(sp1,sp2)
    samples[:,-1] = np.zeros(2*Ntake) + geocent_time_est

    # Define waveform generators
    try:
        duration = list(info_IMR['meta_data']['meta_data']['duration'])[0]
    except:
        print("\nWarning: Duration is not stored in PE summary file. Using default duration = 32s. \n")
        duration = 32

    sampling_frequency = 4096.* input_sampling_frequency_times4k
    #fref_EOB = float(list(info_EOB['config_file']['engine']['fref'])[0])
    fref_EOB = 20
    if fmin_EOB==0.:
        fmin_EOB = float(list(info_EOB['config_file']['engine']['fmin-template'])[0])

    waveform_arguments_IMR = dict(waveform_approximant=approx_IMR,
                            reference_frequency=20., minimum_frequency=fmin_EOB)  # f_min for IMR doesn't matter

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
    savefilename = "Deltasq_" + event_name + "_" + file_suffix +"_" + flag + ".txt"
    save_folder = "result_O3bNSBH/"
    np.savetxt(save_folder + savefilename, Deltasq_list_reshaped)

    time_end = time.time()
    timecost = time_end-time_start
    print("\n------ "+event_name,'calculation done (cost ' +str(int(timecost))+ ' s). File saved to <'+ save_folder + savefilename + '>. ------\n')