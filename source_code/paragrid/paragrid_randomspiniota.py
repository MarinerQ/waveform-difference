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


def chipchieff_to_spin12xyz_aligned(chip,chieff):
    ''' S1 = S2'''
    spin_1x = chip/np.sqrt(2)
    spin_1y = chip/np.sqrt(2)
    spin_2x = chip/np.sqrt(2)
    spin_2y = chip/np.sqrt(2)
    
    spin_1z = chieff
    spin_2z = chieff
    
    return spin_1x,spin_1y,spin_1z,spin_2x,spin_2y,spin_2z


def generate_random_spin(Nsample):
    ''' 
    a random point in unit sphere
    (r,theta,phi) is the sphere coordinate
    '''
    r = np.random.random(Nsample)
    phi = 2*np.pi*np.random.random(Nsample)
    cos_theta = 2*np.random.random(Nsample)-1.0
    
    sin_theta = np.sqrt(1-cos_theta**2)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    spin_x = r*sin_theta*cos_phi
    spin_y = r*sin_theta*sin_phi
    spin_z = r*cos_theta
    
    return spin_x, spin_y, spin_z
##################### Time and phase shift for one detector #####################
def get_dtdphi_withift(h1,h2,det):

    psd = det.power_spectral_density_array
    f_array = det.frequency_array

    
    X_of_f = h1*h2.conjugate()/psd
    
    # zero padding
    add_zero = np.zeros(int(63*len(X_of_f)))
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
    add_zero = np.zeros(int(63*len(X_of_f)))
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

def get_inj_paras(parameter_values, parameter_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',
                'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']):
        inj_paras = dict()
        for i in range(len(parameter_names)):
            inj_paras[parameter_names[i]] = parameter_values[i]
        return inj_paras
    
def calculate_deltasq_kernal(sample_ID, Deltasq_list, samples, waveform_generator1, waveform_generator2, ifos, Ncol):

    inj_para = get_inj_paras(samples[sample_ID])
    #print(inj_para)
    all_para = bilby.gw.conversion.generate_all_bbh_parameters(inj_para)
    chi_p = all_para['chi_p']
    chi_eff = all_para['chi_eff']
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

    temp_deltasq_list.append(chi_p)
    temp_deltasq_list.append(chi_eff)

    temp_deltasq_list.append(inj_para['a_1'])
    temp_deltasq_list.append(inj_para['a_2'])
    temp_deltasq_list.append(inj_para['tilt_1'])
    temp_deltasq_list.append(inj_para['tilt_2'])
    temp_deltasq_list.append(inj_para['phi_12'])
    temp_deltasq_list.append(inj_para['phi_jl'])
    temp_deltasq_list.append(inj_para['theta_jn'])


    # Now the temp_deltasq_list is like [Deltasq, rho1, rho2]
    for i in range(len(temp_deltasq_list)):
        Deltasq_list[sample_ID*Ncol*len(ifos)+i] = temp_deltasq_list[i]
    #Deltasq_list.append(temp_deltasq_list)
    #Deltasq_list[sample_ID] = temp_deltasq_list



if __name__ == '__main__':
    time_start = time.time()
    input_sampling_frequency_times4k = float(sys.argv[1])
    input_q = float(sys.argv[2])
    core_num = int(sys.argv[3])
    

    print("\n ------ Start generating parameter sets ------\n")

    Nsample = 6000
    fref_list = np.zeros(Nsample)+20.0
    phiref_list = np.zeros(Nsample)

    para_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',  # 8 intrinsic
                'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']  # 6 extrinsic

    # mass, # = 2
    #q = 1  
    q = input_q
    mass_1 = np.zeros(Nsample) + 30
    mass_2 = mass_1 * q  # q<=1

    mass_ratio = np.zeros(Nsample) + q
    chirp_mass = conversion.component_masses_to_chirp_mass(mass_1,mass_2)

    # spins + iota, #  = 7
    spin_1x, spin_1y, spin_1z = generate_random_spin(Nsample)
    spin_2x, spin_2y, spin_2z = generate_random_spin(Nsample)


    #chieff_sample = np.reshape(np.meshgrid(chieff_grid, chip_grid)[0], (Nsample,))
    #chip_sample = np.reshape(np.meshgrid(chieff_grid, chip_grid)[1], (Nsample,))

    #iota = np.zeros(Nsample) 
    cosiota = 2*np.random.random(Nsample) - 1
    iota = np.arccos(cosiota)

    converted_spin = pespin.spin_angles(mass_1,mass_2,iota , spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,spin_2z, fref_list,phiref_list)

    theta_jn = converted_spin[:,0]
    phi_jl = converted_spin[:,1]
    tilt_1 = converted_spin[:,2]
    tilt_2 = converted_spin[:,3]
    phi_12 = converted_spin[:,4]
    a_1 = converted_spin[:,5]
    a_2 = converted_spin[:,6]


    # other extrinsic, # = 6
    psi = np.zeros(Nsample)
    phase = np.zeros(Nsample)
    ra = np.zeros(Nsample) 
    dec = np.zeros(Nsample) 
    luminosity_distance = np.zeros(Nsample) + 400
    geocent_time = np.zeros(Nsample) + 1187008882.4

    para_list = [chirp_mass,mass_ratio,a_1,a_2,tilt_1,tilt_2,phi_12,phi_jl,
                theta_jn, psi, phase, ra, dec, luminosity_distance, geocent_time]



    samples = np.zeros(shape=(Nsample,len(para_list)) )

    for i in range(len(para_list)):
        samples[:,i] = para_list[i] 


    duration = 8. 
    sampling_frequency = 4096.*input_sampling_frequency_times4k

    #waveform_arguments1 = dict(waveform_approximant='IMRPhenomPv2',
    #                        reference_frequency=20., minimum_frequency=20.)
    '''
    waveform_arguments1 = dict(waveform_approximant='IMRPhenomC',
                            reference_frequency=20., minimum_frequency=20.)

    waveform_arguments2 = dict(waveform_approximant='IMRPhenomD',
                            reference_frequency=20., minimum_frequency=20.)
    
    waveform_arguments1 = dict(waveform_approximant='TaylorF2',
                            reference_frequency=20., minimum_frequency=20.)

    waveform_arguments2 = dict(waveform_approximant='TaylorF2Ecc',
                            reference_frequency=20., minimum_frequency=20.)
    '''
    waveform_arguments1 = dict(waveform_approximant='IMRPhenomXPHM',
                            reference_frequency=20., minimum_frequency=20)

    waveform_arguments2 = dict(waveform_approximant='SEOBNRv4PHM',
                            reference_frequency=20., minimum_frequency=20)


    waveform_generator1 = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments1)

    waveform_generator2 = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments2)

    detctor_name_list = ['H1']
    ifos = bilby.gw.detector.InterferometerList(detctor_name_list)

    # set detector paramaters
    for i in range(len(ifos)):
        det = ifos[i]
        det.duration = duration
        det.sampling_frequency=sampling_frequency
    
    # Calculate Deltasq
    
    # The array to save
    # [factorsqH, .. , .. , DeltasqH, .. , .. , DeltasqH_netshifted, .. , ..]
    #Deltasq_list = np.zeros(shape=(Nsample,3*len(ifos)))
    manager = multiprocessing.Manager()
    Ncol = 3 + 2 + 6 + 1  # (Deltasq, rho1, rho2,   chip, chieff,    6spins+theta_jn)
    Deltasq_list = manager.Array('d', range(Nsample* Ncol *len(ifos) ))

    partial_work = partial(calculate_deltasq_kernal, Deltasq_list=Deltasq_list, samples=samples, waveform_generator1=waveform_generator1, waveform_generator2=waveform_generator2, ifos=ifos, Ncol=Ncol)


    with Pool(core_num) as p:
        p.map(partial_work, range(Nsample) )
        #p.apply_async(partial_work, range(Nsample) ) 
    
    Deltasq_list_reshaped = np.reshape(Deltasq_list, (Nsample, Ncol*len(ifos)))
    # Calculation done. Save Deltasq_list
    
    #savefilename = "paragrid_RandomSpinQp264k_H1" + ".txt"
    qlabel = str(int(10*input_q))
    savefilename = "paragrid_RandomSpinIotaQp{}PAD_H1.txt".format(qlabel)
    save_folder = "grid_output_v2/"
    np.savetxt(save_folder + savefilename, Deltasq_list_reshaped)

    time_end = time.time()
    timecost = time_end-time_start
    print('\n------ Calculation done (cost ' +str(int(timecost))+ ' s). File saved to <'+ save_folder + savefilename + '>. ------\n')
    
# nohup python paragrid_randomspiniota.py 4 1 5 >nh_outdir/bbhrerun21.out &
# nohup python paragrid_randomspiniota.py 4 0.8 5 >nh_outdir/bbhrerun22.out &
# nohup python paragrid_randomspiniota.py 4 0.5 5 >nh_outdir/bbhrerun23.out &
# nohup python paragrid_randomspiniota.py 4 0.2 5 >nh_outdir/bbhrerun24.out &