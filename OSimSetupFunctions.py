# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:59:01 2023

@author: emily
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd
from scipy.signal import butter, filtfilt
import sys
import opensim as osim
import json
import c3d
import pyomeca
import matplotlib.pyplot as plt
from pyomeca import Markers, Analogs

def convertFPdata_OpenSim(input_file, directory, Settings,Trials,trial_num):
    #commented out FPcal_data bc we don't have one from the site
    c3d_path = os.path.join(directory,input_file)
    raw = Analogs.from_c3d(c3d_path, prefix_delimiter=":")
    analog_names = raw.coords['channel'].values
    time = raw.coords['time'].values
    FC_threshold = 20
    
    T = time[1]-time[0]
    cutoff = 12
    order = 3
    b, a = butter(order, cutoff * 2 * T, btype='low', analog=False)
    analog_filt = filtfilt(b,a,raw)
    
    ## Rotate forces for OpenSim
    #assume two force plates are being used
    FPused = [1,2]
    AnalogDataNew = []

    # Split the array into two sets of 6 elements each
    set1 = analog_names[:6]
    set2 = analog_names[6:]
    
    # Add 'FP1' to the names in set1 and 'FP2' to the names in set2
    set1 = [f'FP1 {name}' for name in set1]
    set2 = [f'FP2 {name}' for name in set2]
    
    # Concatenate the modified sets
    ChannelsNew = np.array(set1 + set2)

    for i in range(len(FPused)):
        FPnumber = FPused[i]
        FPname = f'FP{FPnumber}'
        FP_location = np.where(np.char.find(ChannelsNew, FPname) != -1)[0]
        AnalogDataNew.append(analog_filt[FP_location,:])
        
    # OpenSim XYZ orientation: X = AP, Y = vertical, Z = ML
    # NEEDS TO BE CHANGED FOR EACH SITE LOCATION
    for fp in range(len(AnalogDataNew)):
        AnalogDataNew[fp][[0, 1, 2, 3, 4, 5]] = AnalogDataNew[fp][[1, 2, 0, 4, 5, 3]]
        AnalogDataNew[fp][[1, 4]] *= -1
    # Convert the lists to arrays
    AnalogDataNew = np.array(AnalogDataNew)
    
    # write to MOT file
    output_file = input_file.replace('.c3d', '_OpenSimGRF.mot')
    
    # Save output
    Analog_Output = AnalogDataNew.reshape((AnalogDataNew.shape[0] * AnalogDataNew.shape[1], AnalogDataNew.shape[2]))
    Analog_Output = np.transpose(Analog_Output)
    Analog_Output[Analog_Output[:, 1] < 20, 1] = 0
    Analog_Output[Analog_Output[:, 7] < 20, 7] = 0

    # Length of output
    npts, _ = Analog_Output.shape
    
    if not directory:
        file_path = os.path.join('OpenSim', output_file)
    else:
        file_path = os.path.join(directory, 'OpenSim', output_file)

    # Open the file for writing
    with open(file_path, 'w') as fid:
        # Write the header
        fid.write(output_file + '\n')
        fid.write('version=1\n')
        fid.write(f'nRows={npts}\n')
        fid.write('nColumns=13\n')
        fid.write('inDegrees=yes\n')
        fid.write('endheader\n')

        # Define column headers
        ChanOutds = [
            'time', 'FP1_vx', 'FP1_vy', 'FP1_vz',
            'FP1_x', 'FP1_y', 'FP1_z',
            'FP2_vx', 'FP2_vy', 'FP2_vz',
            'FP2_x', 'FP2_y', 'FP2_z'
        ]

        # Write the data
        for i in range(npts):
            if i == 0:
                fid.write('\t'.join(ChanOutds) + '\n')
            else:
                fid.write(f'{time[i-1]}\t')
                fid.write('\t'.join(map(lambda x: f'{x:.4f}', Analog_Output[i-1, :])))
                fid.write('\n')

    # Calculate mass
    if 'Static' in input_file  or 'static' in input_file or 'STATIC'  in input_file:
        mass = sum(Analog_Output[:, 1] + Analog_Output[:, 7]) / 9.81 / npts
    else:
        mass = None
    print(f'Wrote {npts} frames of force data to {output_file}')
    
    if 'Static' in input_file  or 'static' in input_file or 'STATIC' not in input_file:
        NewThresh = 10  # threshold in N for steps
        BackupThresh = 10  # in unable to apply new threshold, use backup
        window = 30  # search window for fine-tuning step timing
        
        # First force plate
        vGRF = Analog_Output[:, 1]  # pull vertical ground reaction force
        Threshed = Analog_Output[:, 1]
        Changes = np.array([np.nonzero(Threshed)[0][0],np.nonzero(Threshed)[0][-1]])
        #Changes = np.argmax(Threshed > 0)
        Change_vals = np.diff((Threshed > 0).astype(int))
        
        NewChanges = np.zeros(len(Changes))
        # Number of heel strikes and toe offs
        HS = np.count_nonzero(Change_vals == 1)
        TO = np.count_nonzero(Change_vals == -1)
        # Frames and times of heel strikes and toe offs
        Strikes = np.array([time[np.nonzero(Threshed)[0][0]], np.nonzero(Threshed)[0][0]])
        Offs = np.array([time[np.nonzero(Threshed)[0][-1]], np.nonzero(Threshed)[0][-1]])
        
        for ch in range(len(Changes)):
            if Threshed[Changes[ch]] > 0 and Threshed[Changes[ch] - 1] == 0:
                Srch = 1 if Changes[ch] - window <= 0 else Changes[ch] - window
                Count = np.where(vGRF[Changes[ch]:Srch:-1] < NewThresh)[0][0] - 1 if len(np.where(vGRF[Changes[ch]:Srch:-1] < NewThresh)[0]) > 0 else -1
        
                if Count == -1:  # backup
                    Count = np.where(vGRF[Changes[ch]:Srch:-1] < BackupThresh)[0][0] - 1 if len(np.where(vGRF[Changes[ch]:Srch:-1] < BackupThresh)[0]) > 0 else -1
                    if Count == -1:  # worst case, use original threshold
                        Count = np.where(vGRF[Changes[ch]:Srch:-1] < FC_threshold)[0][0] - 1 if len(np.where(vGRF[Changes[ch]:Srch:-1] < FC_threshold)[0]) > 0 else -1
        
                NewChanges[ch] = Changes[ch] - Count
        
                if Change_vals[Changes[0]-1] == 1:
                    StartEvent = "Strike"  # label first event
            elif Threshed[Changes[ch]+1] == 0 and Threshed[Changes[ch] - 1] > 0:
                Srch = (len(vGRF) if Changes[ch] + window > len(vGRF) else Changes[ch] + window)
                Count = np.where(vGRF[Changes[ch]:Srch] < NewThresh)[0][0] - 1 if len(np.where(vGRF[Changes[ch]:Srch] < NewThresh)[0]) > 0 else -1
        
                if Count == -1:
                    Count = np.where(vGRF[Changes[ch]:Srch] < BackupThresh)[0][0] - 1 if len(np.where(vGRF[Changes[ch]:Srch] < BackupThresh)[0]) > 0 else -1
                    if Count == -1:  # worst case, use original threshold
                        Count = np.where(vGRF[Changes[ch]:Srch] < FC_threshold)[0][0] - 1 if len(np.where(vGRF[Changes[ch]:Srch] < FC_threshold)[0]) > 0 else -1
        
                NewChanges[ch] = Count + Changes[ch]
        
                if Change_vals[Changes[0]-1] == -1:
                    StartEvent = "Off"  # label first event
        
        New_fc_temp1 = np.zeros(len(Analog_Output), dtype=bool)  # initialize
        
        # apply new times to new logical
        if StartEvent == "Strike":
            for ch in range(len(NewChanges)):
                if ch % 2 == 0:  # odd event is strike
                    END = len(Analog_Output) if i == len(NewChanges) - 1 else NewChanges[ch + 1]
                    New_fc_temp1[int(NewChanges[ch]):int(END)] = True
        elif StartEvent == "Off":
            for ch in range(len(NewChanges)):
                if ch == 0:  # signal to start off pulling data
                    New_fc_temp1[: NewChanges[ch] + 1] = True
                if ch % 2 == 1:  # even event is strike
                    END = len(Analog_Output) if i == len(NewChanges) - 1 else NewChanges[ch + 1]
                    New_fc_temp1[int(NewChanges[ch] + 1): int(END)] = True
        
        New_fc_temp1 = np.logical_and(New_fc_temp1, True)
        Trials[trial_num]["TSData"] = {}
        Trials[trial_num]["TSData"]["FP1_NumStrikes"] = HS
        Trials[trial_num]["TSData"]["FP1_NumOffs"] = TO
        Trials[trial_num]["TSData"]["FP1_Strikes"] = Strikes
        Trials[trial_num]["TSData"]["FP1_Offs"] = Offs
        
        
        # Clear variables
        del vGRF, Threshed, Changes, NewChanges, Count, StartEvent, HS, TO, Strikes, Offs
        
        # Second force plate
        vGRF = Analog_Output[:, 7]  # pull vertical ground reaction force
        Threshed = Analog_Output[:, 7]
        Changes = np.array([np.nonzero(Threshed)[0][0],np.nonzero(Threshed)[0][-1]])
        #Changes = np.argmax(Threshed > 0)
        Change_vals = np.diff((Threshed >= FC_threshold).astype(int))
        NewChanges = np.zeros(len(Changes), dtype=int)
        HS = np.count_nonzero(Change_vals == 1)
        TO = np.count_nonzero(Change_vals == -1)
        Strikes = np.array([time[np.nonzero(Threshed)[0][0]], np.nonzero(Threshed)[0][0]])
        Offs = np.array([time[np.nonzero(Threshed)[0][-1]], np.nonzero(Threshed)[0][-1]])
        
        for ch in range(len(Changes)):
            if Threshed[Changes[ch]] >= FC_threshold and Threshed[Changes[ch] - 1] == 0:
                Srch = 1 if Changes[ch] - window <= 0 else Changes[ch] - window
                Count = np.where(vGRF[Changes[ch]:Srch:-1] < NewThresh)[0][0] - 1 if len(np.where(vGRF[Changes[ch]:Srch:-1] < NewThresh)[0]) > 0 else -1
        
                if Count == -1:  # backup
                    Count = np.where(vGRF[Changes[ch]:Srch:-1] < BackupThresh)[0][
                        0
                    ] - 1 if len(np.where(vGRF[Changes[ch]:Srch:-1] < BackupThresh)[0]) > 0 else -1
                    if Count == -1:  # worst case, use original threshold
                        Count = np.where(vGRF[Changes[ch]:Srch:-1] < FC_threshold)[0][
                            0
                        ] - 1 if len(np.where(vGRF[Changes[ch]:Srch:-1] < FC_threshold)[0]) > 0 else -1
        
                NewChanges[ch] = Changes[ch] - Count
        
                if ch == 0:
                    StartEvent = "Strike"  # label first event
            elif Threshed[Changes[ch]] < FC_threshold and Threshed[Changes[ch] - 1] > FC_threshold:
                Srch = (len(vGRF) if Changes[ch] + window > len(vGRF) else Changes[ch] + window)
                Count = np.where(vGRF[Changes[ch]:Srch] < NewThresh)[0][
                    0
                ] - 1 if len(np.where(vGRF[Changes[ch]:Srch] < NewThresh)[0]) > 0 else -1
        
                if Count == -1:
                    Count = np.where(vGRF[Changes[ch]:Srch] < BackupThresh)[0][
                        0
                    ] - 1 if len(np.where(vGRF[Changes[ch]:Srch] < BackupThresh)[0]) > 0 else -1
                    if Count == -1:  # worst case, use original threshold
                        Count = np.where(vGRF[Changes[ch]:Srch] < FC_threshold)[0][
                            0
                        ] - 1 if len(np.where(vGRF[Changes[ch]:Srch] < FC_threshold)[0]) > 0 else -1
        
                NewChanges[ch] = Count + Changes[ch]
        
                if ch == 0:
                    StartEvent = "Off"  # label first event
        
        New_fc_temp2 = np.zeros(len(Analog_Output), dtype=bool)  # initialize new timing logical
        
        # apply new times to new logical
        if StartEvent == "Strike":
            for ch in range(len(NewChanges)):
                if ch % 2 == 0:  # odd event is strike
                    END = len(Analog_Output) if ch == len(NewChanges) - 1 else NewChanges[ch + 1] - 1
                    New_fc_temp2[NewChanges[ch] + 1 : END] = True
        elif StartEvent == "Off":
            for ch in range(len(NewChanges)):
                if ch == 0:  # signal to start off pulling data
                    New_fc_temp2[: NewChanges[ch] + 1] = True
                if ch % 2 == 1:  # even event is strike
                    END = len(Analog_Output) if ch == len(NewChanges) - 1 else NewChanges[ch + 1] - 1
                    New_fc_temp2[NewChanges[ch] + 1 : END] = True
        
        New_fc_temp2 = np.logical_and(New_fc_temp2, True)
        Trials[trial_num]["TSData"]["FP2_NumStrikes"] = HS
        Trials[trial_num]["TSData"]["FP2_NumOffs"] = TO
        Trials[trial_num]["TSData"]["FP2_Strikes"] = Strikes
        Trials[trial_num]["TSData"]["FP2_Offs"] = Offs

        return mass,Trials

def writeTRC(trc_table, file_path, filter_freq):
    if file_path is None:
        file_path = os.path.join(tempfile.gettempdir(), 'temp.trc')
    elif not file_path.endswith('.trc'):
        file_path += '.trc'

    if not trc_table.columns[0] == 'Header':
        raise ValueError('Unrecognized table format')

    if not (len(trc_table.columns) - 1) % 3 == 0:
        raise ValueError('Number of channels in the table does not match with 3D coordinate system')

    frames = pd.DataFrame({'Frame': np.arange(len(trc_table.Header))})
    full_table = pd.concat([frames, trc_table], axis=1)

    fs = 1 / np.mean(np.diff(trc_table.Header))
    fc = 6 #cutoff frequency
    
    data = full_table.values
    # if filter_freq > 0:
    #     data[:, 2:] = butterworth_filter(data[:, 2:],fc, fs, order=4)
    
    marker_names = [label.replace('_x', '') for label in trc_table.columns[1::3]]
    n_markers = len(marker_names)

    with open(file_path, 'w') as file:
        file.write(f'PathFileType\t3\t(X/Y/Z)\t{file_path}\n')
        file.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        file.write(f'{fs}\t{fs}\t{len(data)}\t{n_markers}\tmm\t{fs}\t{data[0, 0]}\t{len(data)}\n')
        file.write('Frame\tTime\t' + '\t\t\t'.join(marker_names) + '\n')
            
        coord_headers = '\t'.join([f'X{i}\tY{i}\tZ{i}' for i in range(1, n_markers + 1)])
        file.write(f'\t\t{coord_headers}\n')
        
        np.savetxt(file, data, delimiter='\t', fmt='%1.8f', comments='')

        
    return file_path

def butterworth_filter(data, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq/(fs/2)    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def LoadGRF(Files, ToPlot,Trials,StaticTrial):
    # Load OpenSim ground reaction force data

    # Input Parameters
    if ToPlot is None:
        ToPlot = 0  # default plot setting = off

    # Select Files for input
    if Files is None:
        Files = []
        path = ''
        while not Files:
            Files = input('Enter GRF .mot files separated by space: ').split()
        path = os.path.dirname(Files[0])
    else:
        if isinstance(Files, list):
            NumTrials = len(Files)
            for i in range(NumTrials):
                path = os.path.dirname(Files[i])
        else:
            path = os.path.dirname(Files)

    # Load data
    # if isinstance(Files, list):
    #     NumTrials = len(Files)
    #     GRFdata = [{'FileName': 'placeholder', 'AllData': 'placeholder', 'HeaderInfo': 'placeholder'} for _ in range(NumTrials)]
    #     for i in range(NumTrials):
    #         data = np.loadtxt(Files[i], skiprows=7)
    #         GRFdata[i]['FileName'] = Files[i]
    #         GRFdata[i]['AllData'] = data
    #         GRFdata[i]['HeaderInfo'] = np.genfromtxt(Files[i], max_rows=7, dtype=str)
    # else:
    #     NumTrials = 1
    #     data = np.loadtxt(Files, skiprows=7)
    #     time_column = data[:, 0]
    #     force_columns = data[:, 1:4]
    
    GRFdata = []
    with open(Files[i], 'r') as f:
        # Read the first 7 lines as header info
        header_info = [next(f).strip() for _ in range(7)]

        # Use np.genfromtxt to load the rest of the data
        data = np.genfromtxt(f)
    
        # Append data to GRFdata list
    GRFdata.append({'FileName': Trials[StaticTrial]["name"]+'.c3d', 'AllData': data, 'HeaderInfo': header_info})

    # Set up variables prior to plotting
    Headers = ['time', 'FP1_vx', 'FP1_vy', 'FP1_vz',
               'FP1_x', 'FP1_y', 'FP1_z',
               'FP2_vx', 'FP2_vy', 'FP2_vz',               
               'FP2_x', 'FP2_y', 'FP2_z']

    for i in range(NumTrials):
        GRFdata[i]['ColHeaders'] = Headers

    LW = 2  # set line width

    # Display trials loaded
    for i in range(NumTrials):
        print(f"{GRFdata[i]['FileName']} loaded and saved in structure")
    
    return GRFdata