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

#file_folder = '/home/daniel.williams/events/O3/o3a_final/releases/release-v3-sandbox/'
#file_folder = '/home/daniel.williams/events/O3/o3a_final/releases/release-v5d1-sandbox/'
file_folder = '/home/qian.hu/gwosc_PEresult/gwtc2p1/'


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

    inj_para = get_inj_paras(samples[sample_ID])
    #if sample_ID%50 == 0:
    #    print("Current sample ID: {}\n".format(sample_ID))
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
    fmin_EOB = float(sys.argv[3])
    core_num = int(sys.argv[4])

    event_name = input_event_name  
    print("\n ------ Start analyzing "+ event_name + ' ------\n')
    print("\ncore_num = {}".format(core_num))

    # Read file
    filename = find_pe_file(input_event_name, files_GWTC2p1)
    f = h5py.File(filename,'r')
    info_IMR = f['C01:IMRPhenomXPHM']
    info_EOB = f['C01:SEOBNRv4PHM']
    info_mixed = f['C01:Mixed']
    postsample_mixed = info_mixed['posterior_samples']

    # Read posterior samples. 
    # t_c is set as constant as only IMR gives its estimation. 
    # Other paras are from mixed samples. 
    if event_name in ["GW190707_093326","GW190728_064510","GW190924_021846","GW190725_174728",'GW190720_000836','GW170608_020116']:
        geocent_time_sample = f['C01:IMRPhenomXPHM']['posterior_samples']['geocent_time']
        geocent_time_est = np.mean(geocent_time_sample)
    else:
        geocent_time_est = float(list(info_IMR['config_file']['config']['trigger-time'])[0])

    postsample_mixed = info_mixed['posterior_samples']
    para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',  # 8 intrinsic
             'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic
    samples = np.zeros(shape=(len(postsample_mixed['chirp_mass']), len(para_names)) )
    for i in range(len(para_names)-1):
        samples[:,i] = postsample_mixed[para_names[i]] 
    samples[:,-1] = np.zeros(len(postsample_mixed['chirp_mass']) ) + geocent_time_est


    # Sampling freq
    sampling_frequency = 4096.* input_sampling_frequency_times4k
    print("\nSampling_frequency = {}Hz.".format(sampling_frequency))

    # f_Ref
    try:
        fref_EOB = float(list(info_EOB['config_file']['engine']['fref'])[0])
    except:
        print("\ninfo_EOB['config_file']['engine']['fref'] not found. Using fref_EOB=20Hz.")
        fref_EOB = 20
    try:
        fref_IMR = float(list(info_IMR['config_file']['config']['reference-frequency'])[0])
    except:
        print("\ninfo_IMR['config_file']['config']['reference-frequency'] not found. Using fref_IMR=20Hz.")
        fref_IMR = 20
    
    # f_min
    if fmin_EOB==0.:
        mtot_max = max(info_mixed['posterior_samples']['total_mass'])
        #fmin_EOB = int(get_fmin(mtot_max))
        fmin_EOB = 0.99*get_fmin(mtot_max)
        if fmin_EOB==0:
            fmin_EOB=get_fmin(mtot_max)/2.0
        else:
            fmin_EOB = float(fmin_EOB)
        if fmin_EOB>20:
            fmin_EOB=20.0
        print("\nUsing fmin_EOB = " + str(fmin_EOB) + "Hz.\n")

    # Define waveform generators 
    duration_lowerlimit = get_chirptime(postsample_mixed,fmin_EOB)
    print("\nMax chirp time = {}s".format(duration_lowerlimit))
    '''
    try:
        duration = list(info_IMR['meta_data']['meta_data']['duration'])[0]
        print("Using duration stored in PE file {}s".format(duration))
        if duration<duration_lowerlimit:
            print("Warning: this is less than chirp time.")

    except:
        if duration_lowerlimit<4:
            duration = 4
            print("\nWarning: Duration is not stored in PE summary file. Using 4s. \n")
        elif duration_lowerlimit<8:
            duration = 8
            print("\nWarning: Duration is not stored in PE summary file. Using 8s. \n")
        elif duration_lowerlimit<16:
            duration = 16
            print("\nWarning: Duration is not stored in PE summary file. Using 16s. \n")
        elif duration_lowerlimit<32:
            duration = 32
            print("\nWarning: Duration is not stored in PE summary file. Using 32s. \n")
        elif duration_lowerlimit<64:
            duration = 64
            print("\nWarning: Duration is not stored in PE summary file. Using 64s. \n")
        elif duration_lowerlimit>64:
            duration = 64
            print("\nWarning: Duration is not stored in PE summary file, and chirp time>64s. Using duration=64s anyway. \n")
'''
    try:
        duration = list(info_IMR['meta_data']['meta_data']['duration'])[0]
        print("\nDuration in PE file = {}s.".format(duration))
    except:
        print("\nDuration not in PE file.")
        
    if duration_lowerlimit<4:
        duration = 4
        print("\nWarning: Not using duration stored in PE summary file. Using 4s. \n")
    elif duration_lowerlimit<8:
        duration = 8
        print("\nWarning: Not using duration stored in PE summary file. Using 8s. \n")
    elif duration_lowerlimit<16:
        duration = 16
        print("\nWarning: Not using duration stored in PE summary file. Using 16s. \n")
    elif duration_lowerlimit<32:
        duration = 32
        print("\nWarning: Not using duration stored in PE summary file. Using 32s. \n")
    elif duration_lowerlimit<64:
        duration = 64
        print("\nWarning: Not using duration stored in PE summary file. Using 64s. \n")
    elif duration_lowerlimit>64:
        duration = 64
        print("\nWarning: Not using duration stored in PE summary file, and chirp time>64s. Using duration=64s anyway. \n")

    waveform_arguments_IMR = dict(waveform_approximant='IMRPhenomXPHM',
                            reference_frequency=fref_IMR, minimum_frequency=fmin_EOB)  # f_min for IMR doesn't matter

    waveform_arguments_EOB = dict(waveform_approximant='SEOBNRv4PHM',
                            reference_frequency=fref_EOB, minimum_frequency=fmin_EOB)


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
    #Nsample = 200
    
    # The array to save
    # [factorsqH, .. , .. , DeltasqH, .. , .. , DeltasqH_netshifted, .. , ..]
    #Deltasq_list = np.zeros(shape=(Nsample,3*len(ifos)))
    manager = multiprocessing.Manager()
    Deltasq_list = manager.Array('d', range(Nsample* 3*len(ifos) ))

    partial_work = partial(calculate_deltasq_kernal, Deltasq_list=Deltasq_list, samples=samples, waveform_generator_EOB=waveform_generator_EOB, waveform_generator_IMR=waveform_generator_IMR, ifos=ifos)
    
    f.close()

    with Pool(core_num) as p:
        p.map(partial_work, range(Nsample) )
        #p.apply_async(partial_work, range(Nsample) ) 
    
    Deltasq_list_reshaped = np.reshape(Deltasq_list, (Nsample, 3*len(ifos)))
    # Calculation done. Save Deltasq_list
    file_suffix = detctor_name_list[0] + detctor_name_list[1]
    if len(detctor_name_list)==3:
        file_suffix += detctor_name_list[2]
    savefilename = "Deltasq_" + event_name + "_" + file_suffix + ".txt"
    save_folder = "result_GWTC2p1BBH/"
    np.savetxt(save_folder + savefilename, Deltasq_list_reshaped)

    time_end = time.time()
    timecost = time_end-time_start
    print("\n------ "+event_name,'calculation done (cost ' +str(int(timecost))+ ' s). File saved to <'+ save_folder + savefilename + '>. ------\n')
