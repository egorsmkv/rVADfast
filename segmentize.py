import numpy 
import sys
import speechproc
from scipy.signal import lfilter
import scipy.io.wavfile as wav
from copy import deepcopy

if len(sys.argv) != 2:
    print('Incorrect usage')
    exit(1)

finwav = sys.argv[1]

###

winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
ftThres=0.5; vadThres=0.4
opts=1

fs, data = speechproc.speech_wave(finwav)   
ft, flen, fsh10, nfr10, x_frames =speechproc.sflux(data, fs, winlen, ovrlen, nftt)


# --spectral flatness --
pv01=numpy.zeros(nfr10)
pv01[numpy.less_equal(ft, ftThres)]=1 
pitch=deepcopy(ft)

pvblk=speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)


# --filtering--
ENERGYFLOOR = numpy.exp(-50)
b=numpy.array([0.9770,   -0.9770])
a=numpy.array([1.0000,   -0.9540])
fdata=lfilter(b, a, data, axis=0)


#--pass 1--
noise_samp, noise_seg, n_noise_samp=speechproc.snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)

#sets noisy segments to zero
for j in range(n_noise_samp):
    fdata[range(int(noise_samp[j,0]),  int(noise_samp[j,1]) +1)] = 0 


vad_seg=speechproc.snre_vad(fdata,  nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)

nb = 16
max_nb = float(2 ** (nb - 1))

## Save into wav files
## Code is from https://github.com/eziolotta/rVADfast/blob/master/rVAD_fast.py#L78

speech_seg_data = []
curr_seg = None
for i in range(len(vad_seg)):
    is_speech = vad_seg[i]
    if is_speech == 0:
        ## is noise-silence segment
        if(curr_seg!=None):
            ## get flat array
            curr_seg = numpy.concatenate(curr_seg)
            ## calculate original signal
            curr_seg = curr_seg * (max_nb + 1.0)
            curr_seg = numpy.int16(curr_seg)
            speech_seg_data.append(curr_seg)
        curr_seg = None
    else:
        ## is speech segment
        curr_seg = [] if curr_seg==None else curr_seg
        c_data = numpy.asarray(x_frames[i]).reshape(-1)
        ## append previous
        curr_seg.append(c_data)

sampling_rate, _ = wav.read(finwav) 

## save wav
for i in range(len(speech_seg_data)):
    d= speech_seg_data[i]
    print(d)
    print(type(d))
    output_file = f'segment_{i}.wav'

    wav.write(output_file,sampling_rate,d)

print('Done')
