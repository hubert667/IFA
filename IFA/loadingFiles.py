### Disclaimer: This loading and saving code is reutilized from the one given in the ML2 ICA lab of 2013/2014, 1st semester.
import numpy as np
import scipy.io.wavfile


wav_data = []



def save_data(X,name,sample_rate):
    for i in range(X.shape[0]):
        save_wav(X[i], name + str(i) + '.wav', sample_rate)
        
def load_data():
    #source_files = ['beet.wav', 'beet9.wav', 'beet92.wav', 'mike.wav', 'street.wav']
    source_files = ['mike.wav', 'street.wav']
    sample_rate = None
    for f in source_files:
        sr, data = scipy.io.wavfile.read(f)
        if sample_rate is None:
            sample_rate = sr
        else:
            assert(sample_rate == sr)
        wav_data.append(data[5000:150000])  # cut off the last part so that all signals have same length
    return sample_rate

def GetData(source,size,period):
    """"""
    data = wav_data[source]
    data=data[period*size:period*size+size]
    data=data.astype(float)
    data/=np.max(wav_data)
    return data

def GetAllData(source):
    data = wav_data[source]
    data=data.astype(float)
    data/=np.max(wav_data)
    return data


def save_wav(data, out_file, rate):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    scipy.io.wavfile.write(out_file, rate, scaled)




# Load audio sources while guaranteeing that all the sample rates from the different sound files are the same



# Save reconstructed signals to disk (assuming each row is a source):


    
    
    