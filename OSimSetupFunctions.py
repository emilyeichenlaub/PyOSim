# -*- coding: utf-8 -*-
"""
functions called within MainOSimSetup:
    C3D2OpenSim
    writeTRC
    butterworth_filter
    extract_strength_data
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.linalg import svd
from scipy.signal import butter, filtfilt, find_peaks
import sys
import opensim as osim
import json
import math
import matplotlib.pyplot as plt

# =============================================================================
# C3d2OpenSim.py code written by Emily Eichenlaub and assisted
# by Mohammadreza Rezaie 
# =============================================================================
def C3D2OpenSim(input_file, directory, Settings,Trials,trial_num):
    site = Settings['Site'] # name of lab site
    coordination = Settings[site] # coordinates of lab site

    # define which axes (X,Y,Z in order (first or second)) and how much must be rotated 
    # for the data coordinations, rot=[X,Y,Z]
    if   'XYZ' in coordination: rot=[0,0,0]; Mag1=0       ; Mag2=0 # OpenSim
    elif 'XZY' in coordination: rot=[1,0,0]; Mag1=-np.pi/2; Mag2=0
    elif 'ZYX' in coordination: rot=[0,1,0]; Mag1=+np.pi/2; Mag2=0
    elif 'ZXY' in coordination: rot=[1,0,2]; Mag1=+np.pi/2; Mag2=+np.pi/2
    elif 'YXZ' in coordination: rot=[0,2,1]; Mag1=+np.pi/2; Mag2=-np.pi
    elif 'YZX' in coordination: rot=[1,2,0]; Mag1=-np.pi/2; Mag2=-np.pi/2
    else: raise RuntimeError(f'The {coordination} is not defined properly.')
    if coordination.startswith('-'): Mag3=np.pi # for the direction of movement
    else: Mag3=0 
    
    c3d_path = os.path.join(directory,input_file)
    raw = Analogs.from_c3d(c3d_path, prefix_delimiter=":")
    analog_names = raw.coords['channel'].values
    time = raw.coords['time'].values

    NewMkrFile = os.path.join(Trials[trial_num]["folder"],'OpenSim',input_file.replace('.c3d', '_OpenSim.trc'))
    NewFrcFile = os.path.join(Trials[trial_num]["folder"],'OpenSim',input_file.replace('.c3d', '_OpenSimGRF.mot'))
    
    # transform (rotate) TimeSeriesTableVec3
    def rotation(vectorVec3, magnitude,axis):
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
    
    # Get number of frames (rows) and markers (columns)
    num_rows = markers.getNumRows()
    num_markers = markers.getNumColumns()
    time_values = markers.getIndependentColumn()  # Time column
    
    for marker_idx in range(num_markers):
        # initiate xyz data for each marker
        x_data = []
        y_data = []
        z_data = []
        time_data = []
        missing_data = []
        
        # iterate through each row in c3d file
        for row_idx in range(num_rows):
            # get xyz marker data for each row
            mrk_val = markers.getRowAtIndex(row_idx)[marker_idx] # Vec3 data type now
            # if it is nan or 0, that row is empty
            if (mrk_val.get(0) == 0 and mrk_val.get(1) == 0 and mrk_val.get(2) == 0) or (math.isnan(mrk_val.get(0)) and math.isnan(mrk_val.get(1)) and math.isnan(mrk_val.get(2))):
                # add row index to missing_data
                missing_data.append(row_idx)
            else:
                # add non-empty indices to time and xyz
                time_data.append(time_values[row_idx])
                x_data.append(mrk_val.get(0))
                y_data.append(mrk_val.get(1))
                z_data.append(mrk_val.get(2))  
                
        if len(missing_data)>0: # if there are gaps in the data
            # Cubic Spline xyz data
            spline_x = CubicSpline(time_data, x_data, bc_type='not-a-knot')
            spline_y = CubicSpline(time_data, y_data, bc_type='not-a-knot')
            spline_z = CubicSpline(time_data, z_data, bc_type='not-a-knot')
            
            # for each row in missing data, get the xyz value for the missing index
            for row_idx in missing_data:
                interpolated_x = spline_x(time_values[row_idx])
                interpolated_y = spline_y(time_values[row_idx])
                interpolated_z = spline_z(time_values[row_idx])
    
                # Vec3 can only take in floats - convert array to float
                new_vec3 = osim.Vec3(float(interpolated_x), float(interpolated_y), float(interpolated_z))
                row_vec3 = markers.getRowAtIndex(row_idx)
                row_vec3[marker_idx] = new_vec3  # Update only this marker in the row
    
                markers.setRowAtIndex(row_idx, row_vec3)  # Save back into OpenSim table
                    
    # rotate dynamic markers and forces data
    if Mag1!=0: rotation(markers, Mag1, rot.index(1)) # Mag1
    if Mag2!=0: rotation(markers, Mag2, rot.index(2)) # Mag2
    if Mag3!=0: rotation(markers, Mag3, 1) # Y (vertical axis)    
    
    if Mag1!=0: rotation(forces, Mag1, rot.index(1)) # Mag1
    if Mag2!=0: rotation(forces, Mag2, rot.index(2)) # Mag2
    if Mag3!=0: rotation(forces, Mag3, 1) # Y (vertical axis)     

    # write static markers to TRC file
    osim.TRCFileAdapter().write(markers, NewMkrFile)
    print(f'Wrote {markers.getNumRows()} frames of marker data to {input_file.replace(".c3d", "_OpenSim.trc")}')
    
    og_labels = forces.getColumnLabels()
    # force data labels
    fpnum = int(len(og_labels)/3)
    labels = list()
    for fp in range(1,fpnum+1):
        labels += [f"ground_force_{fp}_v", f"ground_force_{fp}_p", f"ground_moment_{fp}_m"]
        
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
        '''get timing of gait events (heel strike and toe off)
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

def extract_strength_data(position_data, torque_data,collection_type):
    # 8 Hz lowpass 4th order butterworth filter
    torque_data = butterworth_filter(torque_data,8, 100, order=4)

    if collection_type == 'humac':
        f2e = np.diff(position_data) < 0 # flexion to extension - negative slope
        e2f = np.diff(position_data) > 0 # extension to flexion - positive slope
        
        peaks, _ = find_peaks(torque_data[0:len(torque_data)],distance=100,height=1)    
        
        plt.scatter(peaks,np.zeros(len(peaks)),label='peaks')
        plt.plot(position_data)
        plt.plot(torque_data, label='torque')
        
        tor_vals = torque_data[peaks]
        # convert to series with indices of peak torques
        tor_vals = pd.Series(tor_vals, index=peaks)
    
        extension = []
        flexion = []
        for idx in peaks:  # Use 'peaks' to get the original index
            peak_val = tor_vals[idx]  # Get the value from 'tor_vals' at the current peak index
            position_val = position_data[idx]
            if f2e[idx] == True:  # Check whether the corresponding f2e value is True
                extension.append((position_val, peak_val))  # Append the index and the value to extension
            else:
                flexion.append((position_val, peak_val))
                
    if collection_type == 'biodex':
        
        position_data = butterworth_filter(position_data,8, 100, order=4)
        
        # biodex = extension is more positive, flexion is less positive
        f2e = np.diff(position_data) < 0 # flexion to extension - negative slope (should be pos torque vals)
        e2f = np.diff(position_data) > 0 # extension to flexion - positive slope (should be neg torque vals)
        
        # extension torques
        ext_peaks, _ = find_peaks(torque_data[0:len(torque_data)],distance=200,prominence=1)  
        
        plt.scatter(ext_peaks,np.zeros(len(ext_peaks)),label='extension peaks')
        plt.plot(position_data)
        plt.plot(torque_data, label='extension torque')
        
        # flexion torques (originally negative values)
        inverted_torque_data = -torque_data
        flex_peaks, _ = find_peaks(inverted_torque_data, distance=200,prominence=1)
                
        plt.scatter(flex_peaks,np.zeros(len(flex_peaks)),label='flexion peaks')
        plt.plot(position_data, label='position(ext+/flex-')
        plt.plot(inverted_torque_data, label='flexion torque')
        plt.legend()
        plt.show()
        
        peaks = np.concatenate((flex_peaks, ext_peaks),axis = 0) # concatenate flexion and extension indices
        tor = torque_data[peaks] # concatenate flexion and extension torque values
        tor_vals = pd.Series(tor, index=peaks) # get series that combines peak indices and torque values

        extension = []
        flexion = []
        for idx in peaks:  # Use 'peaks' to get the original index
            peak_val = tor_vals[idx]  # Get the value from 'tor_vals' at the current peak index
            position_val = position_data[idx]
            if f2e[idx] == True:  # Check whether the corresponding f2e value is True
                extension.append((position_val, peak_val))  # Append the index and the value to extension
            else:
                flexion.append((position_val, peak_val))
            
    flex = pd.DataFrame(flexion, columns=['Position(Degrees)', 'Torque(Foot-Pounds)'])
    ext = pd.DataFrame(extension, columns=['Position(Degrees)', 'Torque(Foot-Pounds)'])

    return ext, flex
   
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
