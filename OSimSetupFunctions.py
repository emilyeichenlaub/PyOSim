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
#import c3d
import matplotlib.pyplot as plt
from pyomeca import Markers, Analogs, DataArrayAccessor

# =============================================================================
# C3d2OpenSim.py code written by Emily Eichenlaub and assisted
# by Mohammadreza Rezaie 
# =============================================================================
def C3D2OpenSim(input_file, directory, Settings,Trials,trial_num):
    
    c3d_path = os.path.join(directory,input_file)
    raw = Analogs.from_c3d(c3d_path, prefix_delimiter=":")
    analog_names = raw.coords['channel'].values
    time = raw.coords['time'].values

    NewMkrFile = os.path.join(Trials[trial_num]["folder"],'OpenSim',input_file.replace('.c3d', '_OpenSim.trc'))
    NewFrcFile = os.path.join(Trials[trial_num]["folder"],'OpenSim',input_file.replace('.c3d', '_OpenSimGRF.mot'))
    rot=[1,2,0]; mag1=-np.pi/2; mag2=-np.pi/2 # MGH lab: 'YZX'
    
    # transform (rotate) TimeSeriesTableVec3
    def rotation(vectorVec3, axis, magnitude):
        	if   axis==0: vec=osim.Vec3(1,0,0) # X
        	elif axis==1: vec=osim.Vec3(0,1,0) # Y
        	elif axis==2: vec=osim.Vec3(0,0,1) # Z
        	rot = osim.Rotation(magnitude, vec)
        	for i in range(vectorVec3.getNumRows()):
        		new = rot.multiply(vectorVec3.getRowAtIndex(i))
        		vectorVec3.setRowAtIndex(i, new)
    
    # read markers
    c3dAdapter = osim.C3DFileAdapter()
    c3dAdapter.setLocationForForceExpression(c3dAdapter.ForceLocation_CenterOfPressure)
    tables = c3dAdapter.read(c3d_path)
    # read markers and forces from C3D file
    markers = c3dAdapter.getMarkersTable(tables)
    forces  = c3dAdapter.getForcesTable(tables)
                
    # rotate dynamic markers and forces data
    rotation(markers, rot.index(1), mag1)
    rotation(markers, rot.index(2), mag2)
    rotation(forces, rot.index(1), mag1)
    rotation(forces, rot.index(2), mag2)

    # write static markers to TRC file
    osim.TRCFileAdapter().write(markers, NewMkrFile)
    print(f'Wrote {markers.getNumRows()} frames of marker data to {input_file.replace(".c3d", "_OpenSim.trc")}')

    # force data labels
    labels = ['ground_force_1_v', 'ground_force_1_p', 'ground_moment_1_m',
    		 'ground_force_2_v', 'ground_force_2_p', 'ground_moment_2_m']
    forces.setColumnLabels(labels)
    
    # flat the TimeSeriesTableVec3
    forces = forces.flatten(['x', 'y', 'z'])
    
    # convert mm and Nmm to m and Nm, respectively (if applicable)
    labels = forces.getColumnLabels()
    units = forces.getDependentsMetaDataString('units')
    forces.removeDependentsMetaDataForKey('units')
    
    for label,unit in zip(labels, units):
       	# print(label, unit)
       	column = forces.getDependentColumn(label).to_numpy()
       	forces.removeColumn(label)
           
        if label == 'ground_force_1_vy':
            column = np.where(column < 20, 0, column)
        if label == 'ground_force_2_vy':
            column = np.where(column < 20, 0, column)
            
       	# convert nans to zero
       	np.nan_to_num(column, copy=False, nan=0)
       	if unit.endswith('mm'):
       		column /= 1000.0
    
       	forces.appendColumn(label, osim.Vector(column))
    
    # add metadata
    forces.addTableMetaDataString('nColumns', str(forces.getNumColumns()))
    forces.addTableMetaDataString('nRows',    str(forces.getNumRows()))	
    forces.addTableMetaDataString('inDegrees', 'no')
    
    # write forces data to MOT file
    osim.STOFileAdapter().write(forces, NewFrcFile)

    print(f'Wrote {forces.getNumRows()} frames of force data to {input_file.replace(".c3d", "_OpenSimGRF.mot")}')
    
    if Settings["GaitEvents"] == 'Yes':
        '''get timing of gait events (Heel Strike and Toe Off)
        '''
        if 'Static' in input_file  or 'static' in input_file or 'STATIC' not in input_file:
            FC_threshold = 20
            NewThresh = 10  # threshold in N for steps
            BackupThresh = 10  # in unable to apply new threshold, use backup
            window = 30  # search window for fine-tuning step timing
            force_data = pd.read_csv(NewFrcFile, skiprows=8, delimiter='\t')
            force_array = force_data.values
            time = force_array[:,0]
    
            # First force plate
            vGRF = force_array[:, 2]  # pull vertical ground reaction force
            Threshed = force_array[:, 2]
            Changes = np.array([np.nonzero(Threshed)[0][0],np.nonzero(Threshed)[0][-1]])
            #Changes = np.argmax(Threshed > 0)
            Change_vals = np.diff((Threshed > FC_threshold).astype(int))
            
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
            
            New_fc_temp1 = np.zeros(len(force_array), dtype=bool)  # initialize
            
            # apply new times to new logical
            if StartEvent == "Strike":
                for ch in range(len(NewChanges)):
                    if ch % 2 == 0:  # odd event is strike
                        END = len(force_array) if ch == len(NewChanges) - 1 else NewChanges[ch + 1]
                        New_fc_temp1[int(NewChanges[ch]):int(END)] = True
            elif StartEvent == "Off":
                for ch in range(len(NewChanges)):
                    if ch == 0:  # signal to start off pulling data
                        New_fc_temp1[: NewChanges[ch] + 1] = True
                    if ch % 2 == 1:  # even event is strike
                        END = len(force_array) if ch == len(NewChanges) - 1 else NewChanges[ch + 1]
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
            vGRF = force_array[:, 11]  # pull vertical ground reaction force
            Threshed = force_array[:, 11]
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
            
            New_fc_temp2 = np.zeros(len(force_array), dtype=bool)  # initialize new timing logical
            
            # apply new times to new logical
            if StartEvent == "Strike":
                for ch in range(len(NewChanges)):
                    if ch % 2 == 0:  # odd event is strike
                        END = len(force_array) if ch == len(NewChanges) - 1 else NewChanges[ch + 1] - 1
                        New_fc_temp2[NewChanges[ch] + 1 : END] = True
            elif StartEvent == "Off":
                for ch in range(len(NewChanges)):
                    if ch == 0:  # signal to start off pulling data
                        New_fc_temp2[: NewChanges[ch] + 1] = True
                    if ch % 2 == 1:  # even event is strike
                        END = len(force_array) if ch == len(NewChanges) - 1 else NewChanges[ch + 1] - 1
                        New_fc_temp2[NewChanges[ch] + 1 : END] = True
            
            New_fc_temp2 = np.logical_and(New_fc_temp2, True)
            Trials[trial_num]["TSData"]["FP2_NumStrikes"] = HS
            Trials[trial_num]["TSData"]["FP2_NumOffs"] = TO
            Trials[trial_num]["TSData"]["FP2_Strikes"] = Strikes
            Trials[trial_num]["TSData"]["FP2_Offs"] = Offs
    
            return Trials

def writeTRC(trc_table, file_path, filter_freq):
    '''write data to TRC file compatible with OpenSim
    Inputs:
        trc_table       table of marker values
        file_path       path to Subject's OpenSim folder
        filter_freq     frequency used to filter marker data (Hz)
    Outputs:
        file_path      saves TRC file to Subject's OpenSim folder
    Not used in this current pipeline because of read/write compatibility between Python and OpenSim
    No current open & resave option to make this feasible.
    '''
    if file_path is None:
        file_path = os.path.join(tempfile.gettempdir(), 'temp.trc')
    elif not file_path.endswith('.trc'):
        file_path += '.trc'

    if not trc_table.columns[0] == 'Header':
        raise ValueError('Unrecognized table format')

    if not (len(trc_table.columns) - 1) % 3 == 0:
        raise ValueError('Number of channels in the table does not match with 3D coordinate system')

    frames = pd.DataFrame({'Frame#': np.arange(len(trc_table.Header))})
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
        file.write('Frame#\tTime\t' + '\t\t\t'.join(marker_names) + '\n')
            
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

def LoadGRF(Files,Trials,StaticTrial):
    '''LoadGRF
    Inputs: 
        Files          path to Subject folder with updated _OpenSimGRF.mot file for static trial
        Trials         list of trials for subjects
        StaticTrial    index of the static trial in Trials
    Outputs:
        adds GRF data to static trial in Subjects.json,
        used to calculate subject mass.
    '''

    # Load OpenSim ground reaction force data

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
    
    GRFdata = []
    with open(Files[i], 'r') as f:
        # Read the first 7 lines as header info
        header_info = [next(f).strip() for _ in range(9)]

        # Use np.genfromtxt to load the rest of the data
        data = np.genfromtxt(f)
    
        # Append data to GRFdata list
    GRFdata.append({'FileName': Trials[StaticTrial]["name"]+'.c3d', 'AllData': data, 'HeaderInfo': header_info})

    # Set up variables prior to plotting
    Headers = header_info[8].split('\t')

    for i in range(NumTrials):
        GRFdata[i]['ColHeaders'] = Headers

    LW = 2  # set line width

    # Display trials loaded
    for i in range(NumTrials):
        print(f"{GRFdata[i]['FileName']} loaded and saved in structure")
    
    return GRFdata
