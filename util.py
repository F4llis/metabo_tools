import pymzml
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import scipy


precision = 0.005  # m/z precision for raw data
data_per_sec = 2  # spectrum per second
half_time_window = 30  # time range before and after RT in sec
number_ticks = half_time_window * 2 * data_per_sec  # number of spectrum per data
intensity_treshold = 8000  # everything under this will be annotated false
number_sample = 120


def filter_threshold(values):
    if max(values) >= intensity_treshold:
        return True
    return False

def massage_data(X_i,y_i, mz_i, rt_i):

    dfi = pd.DataFrame(X_i)

    dfi = pd.DataFrame.rename(dfi, columns={0: 'x'})

    dfi['y'] = y_i
    dfi['mz'] = mz_i
    dfi['rt'] = rt_i

    dfi['treshold_satisfied'] = dfi['x'].apply(lambda x: 1 if np.amax(x) > intensity_treshold else 0)
    dfi['y'] = dfi['y'].apply(lambda x: 1 if x == True else 0)

    dfi['norm'] = dfi['x'].apply(lambda x: NormalizeData(x))

    dfi['smooth'] = dfi['norm'].apply(lambda x: smooth(x))
    dfi['smooth'] = dfi['smooth'].apply(lambda x: x[5:-5])

    dfi['grad1'] = dfi['smooth'].apply(lambda x: np.gradient(x))
    dfi['grad2'] = dfi['grad1'].apply(lambda x: np.gradient(x))

    # Maxima & Minima
    dfi['maxima'] = dfi['smooth'].apply(
        lambda x: scipy.signal.find_peaks(x, width=None, wlen=None, rel_height=0.5, plateau_size=None)[0])
    dfi['minima'] = dfi['smooth'].apply(lambda x: scipy.signal.find_peaks(-x, height=None, threshold=None, distance=None,
                                                                        prominence=None, width=None, wlen=None,
                                                                        rel_height=0.5, plateau_size=None)[0])

    dfi['maxval'] = dfi[['smooth', 'maxima']].apply(lambda x: [x.smooth[i] for i in x.maxima], axis=1)
    dfi['minval'] = dfi[['smooth', 'minima']].apply(lambda x: [x.smooth[i] for i in x.minima], axis=1)

    dfi['maxvalnorm'] = dfi['maxval'].apply(lambda x: NormalizeData(x) if len(x) > 0 else [])
    dfi['minvalnorm'] = dfi['minval'].apply(lambda x: NormalizeData(x) if len(x) > 0 else [])

    # Stats
    dfi['maxvalstats'] = dfi['maxvalnorm'].apply(lambda x: getStat(x))
    dfi['minvalstats'] = dfi['minvalnorm'].apply(lambda x: getStat(x))

    dfi['maximastats'] = dfi['maxima'].apply(lambda x: getStat(x))
    dfi['minimastats'] = dfi['minima'].apply(lambda x: getStat(x))

    dfi['stats'] = dfi['smooth'].apply(lambda x: getStat(x))

    dfi['fft'] = dfi['x'].apply(lambda x: np.fft.fft(x))
    dfi['fftr'] = dfi['fft'].apply(lambda x: np.real(x))
    dfi['ffti'] = dfi['fft'].apply(lambda x: np.real(np.imag(x)))

    return dfi


def getStat(data):
    s = [0, 0, 0, 0, 0]  # nbrObs, mean, var,skew, kurtosis

    if len(data) > 0:
        de = scipy.stats.describe(data)
        s = [de[0]/number_sample, de[2], de[3], de[4], de[5]]

    return np.nan_to_num(np.array(s))
def get_data_mz_batch(data_mz ,features_list):

    run = pymzml.run.Reader(data_mz,MS1_Precision=5e-3, MSn_Precision=5e-3,  MS_precisions = {
        1 : 5e-3,
        2 : 5e-3
    })


    ticks_processed_list = {}
    for feee in features_list:
        ticks_processed_list[feee[0]] = 0

    data_processed = {}
    for fee in features_list:
        data_processed[fee[0]] = []

    for i, scan in enumerate(run):

        if scan.ms_level == 1:

            t, measure = scan.scan_time  # get scan time
            t = t*60

            peaks = defaultdict(list)
            for pk in scan.peaks('raw'):
                peaks[int(pk[0])].append(pk)

            for fe in features_list:

                mz = fe[0]
                ticks_processed = ticks_processed_list[fe[0]]
                data = data_processed[fe[0]]

                if t >= (fe[1]-half_time_window) and ticks_processed < number_ticks :

                    ticks_processed_list[fe[0]] += 1


                    scans = []
                    if int(mz)-1 in peaks:
                        scans += peaks[int(mz)-1]
                    if int(mz) in peaks:
                        scans += peaks[int(mz)]
                    if int(mz)+1 in peaks:
                        scans += peaks[int(mz)+1]

                    pipeak = 0
                    for d in scans:
                        mz_scan = d[0]
                        i_scan = d[1]

                        if mz_scan >= mz - precision and mz_scan <= mz + precision:
                            pipeak = i_scan
                            break

                    data.append(pipeak)


            if i % 50 == 0:
                print(i, run.get_spectrum_count())

    X_ = []
    y_ = []
    mz_ = []
    rt_ = []

    for fi in features_list:
        X_.append(data_processed[fi[0]])
        y_.append(None)
        mz_.append(fi[0])
        rt_.append(fi[1])

    return  [X_, y_, mz_, rt_ ]

def get_data_mz(data_mz ,mz , retention_time , mz_rt_sec=True):

    run = pymzml.run.Reader(data_mz)

    data = []

    ticks_processed = 0

    for i, scan in enumerate(run):

        if scan.ms_level == 1:

            t, measure = scan.scan_time  # get scan time

            if not mz_rt_sec:
                t = t*60

            if t >= (retention_time-half_time_window):

                ticks_processed +=1

                mz_in_range = []
                for d in scan.peaks('raw'):
                    mz_scan = d[0]
                    i_scan = d[1]

                    if mz_scan >= mz - precision  and mz_scan <= mz + precision:
                        mz_in_range.append(i_scan)

                if not mz_in_range:
                    data.append(0)
                else:
                    data.append(mz_in_range[0])

                if ticks_processed == number_ticks:
                    break

    return data

def build_data_ml(path, data_mz_path, annotation=True, mz_rt_sec=True):

    X_ = []
    y_ = []
    mz_ = []
    rt_ = []


    # get all Raw data
    with open(path, newline='') as f:

        it = 0
        reader = csv.reader(f,delimiter=';')
        for line in list(reader):

            spectrum = get_data_mz(data_mz_path, float(line[1]), float(line[2])*60, mz_rt_sec)

            X_.append(spectrum)
            if annotation:
                y_.append(1 if line[6] == 'YES' else 0)
            else:
                y_.append(True)
            mz_.append(float(line[1]))
            rt_.append(float(line[2])*60)

            it +=1
            if it %50 == 0:
                print(it)

    return [X_, y_, mz_, rt_]

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    if window_len<3:
        return x


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def NormalizeData(data):
    if np.max(data) - np.min(data) == 0.0:
        return data
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ret_mats(df):
    xfft = np.hstack( [np.array(df['treshold_satisfied']).reshape((len(df), -1)), np.vstack(df['fftr']), np.vstack(df['ffti'])] )
    x = np.stack([ np.vstack(df['norm']), np.vstack(df['smooth']) ,
              np.vstack(df['grad1']) , np.vstack(df['grad2'])  ] , axis = 2)
    y = df.y.map(lambda x : float(x))

    return x,xfft,y

def ret_mats2(df):
    xfft = np.hstack([np.vstack(df['fftr']), np.vstack(df['ffti']),
                      np.vstack(df['maxvalstats']),
                  np.vstack(df['minvalstats']),
                  np.vstack(df['maximastats']),
                  np.vstack(df['minimastats']),
                  np.vstack(df['stats'])
                  ])

    xstat = np.hstack([np.vstack(df['maxvalstats']),
                       np.vstack(df['minvalstats']),
                  np.vstack(df['maximastats']),
                  np.vstack(df['minimastats']),
                  np.vstack(df['stats'])
                  ])

    x = np.stack([ np.vstack(df['norm']), np.vstack(df['smooth']) ,
              np.vstack(df['grad1']) , np.vstack(df['grad2'])  ] , axis = 2)
    y = df.y.map(lambda x : float(x))

    return x,xfft,xstat, y