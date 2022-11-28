# Waveform difference

Calculating differences in waveform models. Two models can not be both accurate enough if the difference is large (>2).

## Repository structure

### publication

Codes for "Assessing the model waveform accuracy of gravitational waves", Phys. Rev. D 106, 044042 (2022).

real_events/*: Take pesummary file as input and calculate waveform difference. Used to generate Table I,II and Fig. 2,3,4.

paragrid/paragrid_randomspiniota.py: Fig.5

paragrid/paragrid_BNS.py and paragrid/paragrid_NSBH.py: Fig.6

### O4

Implementation of O4 parameter estimation post-processing. Under development.


