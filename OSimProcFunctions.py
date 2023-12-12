# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:50:43 2023

@author: emily
"""
import os
import numpy as np
import pandas as pd
# from scipy.spatial.transform import Rotation as R
# from scipy.linalg import svd
# from scipy.signal import butter, filtfilt
import sys
import opensim as osim
import json
import pyomeca
import shutil
import xml.etree.ElementTree as ET

sys.path.append('C:\\Users\\emily\\OneDrive\\Desktop\\Misc\\UNC Research\\OpenSim_Python\\Functions')

def Scale(Subjects, Settings, static_folder, osim_folder, subj):
    '''Batch process OpenSim Scaling
    Inputs: 
        Settings            a dictionary of parameter settings that controls processing operations
        directory           a path to the subject's primary directory
        static_file_name    the file to use for model scaling
    Outputs:
        scaled_model        an ".osim" model file that is anthropometrically scaled to that participant 
    '''
    # create opensim and scale folders
    if not os.path.exists(osim_folder):
        os.mkdir(osim_folder)
    scale_folder = os.path.join(osim_folder, 'ScaleFiles')
    if not os.path.exists(scale_folder):
        os.mkdir(scale_folder)
        
    # get virtual markers to use for scaling
    os.chdir(osim_folder)
    StaticTrials = [trial["type"] == "static" for trial in Subjects[subj]["Trials"]]
    StaticTrial = [index for index, value in enumerate(StaticTrials) if value]
    # Make virtual markers and copy into scale folder
    StaticMarkerFile = os.path.join(Subjects[subj]["Folders.OpenSimFolder"], Subjects[subj]["Trials"][0]["files"]["OpenSimTRC"])
    StaticVirtualFile = os.path.join(scale_folder, f"{Subjects[subj]['Trials'][0]['files']['OpenSimTRC'][:-4]}_Virtual.trc")

    MakeVirtualMkr(StaticMarkerFile, StaticVirtualFile, Subjects[subj]["Trials"][0]["files"]["OpenSimTRC"], Subjects[subj]["Folders"]["folder"])
    vFile = [x for x in os.listdir(scale_folder) if 'Virtual.trc' in x]
    static_file = [x for x in os.listdir() if 'STATIC' in x][0]
    # if len(vFile) == 0:
    #     virtual_file = getVirtualMarkers(static_file)
    #     t = 'Open & ReSave'
    #     m = 'Virtual Static Marker file created at: ' + virtual_file + '  Open & re-save file before continuing'
    #     ans = messagebox.showinfo(title=t, message=m)
    # else:
    #     virtual_file = vFile[0]

    # state generic setup files
    setup_scale_file = [x for x in os.listdir(Settings['GenericPath']) if 'Setup_Scale_Torso.xml' in x][0]
    markerset_file = [x for x in os.listdir(Settings['GenericPath']) if 'gait2392_Scale_MarkerSet_ABL_Virtual' in x][0]
    orig_model_file = [x for x in os.listdir(Settings['GenericPath']) if 'frontHingesKnee' in x][0]
    orig_model_path = os.path.join(Settings['GenericPath'],orig_model_file)
    subj_model_path = os.path.join(scale_folder, orig_model_file)
    shutil.copy(orig_model_path,subj_model_path)
    
    # get subject mass from average force during static trial
    static_force_file = static_file[:-4] + 'GRF.mot'
    static_forces = pd.read_csv(static_force_file, skiprows=8, sep='\t')
    subject_mass = round((static_forces['ground_force_1_vy'].mean() + static_forces['ground_force_2_vy'].mean()) / 9.81, 2)

    orig_mkrset = os.path.join(Settings["GenericPath"], [name for name in Settings["GenericDir"] if 'gait2392_Scale_MarkerSet_ABL_Virtual.xml' in name][0])
    subj_mkrset = os.path.join(scale_folder, f"{Subjects[subj]['name']}_MkrSet.xml")
    shutil.copy(orig_mkrset, subj_mkrset)
    
    # copy scale setup to run
    tree = ET.parse(os.path.join(Settings['GenericPath'], setup_scale_file))
    root = tree.getroot()
    for elem1 in root:
        #print('1 ' + elem1.tag + ':   ' + str(elem1.text))
        if elem1.tag == 'ScaleTool':
            elem1.attrib['name'] = Subjects[subj]["name"]
        for elem2 in elem1:
            if elem2.tag == 'mass':
                elem2.text = str(subject_mass)
            #print('2 ' + elem2.tag + ':   ' + str(elem2.text))
            for elem3 in elem2:
                if elem3.tag == 'marker_file':
                    #elem3.text = os.path.join(static_folder, vFile[0])
                    elem3.text = vFile[0]
                if elem3.tag == 'marker_set_file':
                    elem3.text = f"{Subjects[subj]['name']}_MkrSet.xml"
                # elem3.text = os.path.join(subj_mkrset)
                if elem3.tag == 'model_file':
                    elem3.text = orig_model_file
                if elem3.tag == 'output_model_file':
                    elem3.text = os.path.join(Subjects[subj]["name"] + '_Scaled.osim')
                if elem3.tag == 'output_scale_file':
                    elem3.text = os.path.join(scale_folder, Subjects[subj]["name"] + '_ScaleSet.xml')
                if elem3.tag == 'output_marker_file':
                    # elem3.text = os.path.join(scale_folder, Subjects[subj]["name"] + '_Scale_Markers.xml')
                    elem3.text = f"{Subjects[subj]['name']}_Scale_Markers.xml"

                #print('3 ' + elem3.tag + ':   ' + str(elem3.text))
                #for elem4 in elem3:
                    #print('4 ' + elem4.tag + ':   ' + str(elem4.text))
    # save scale setup as new file in scale folder
    runScale = os.path.join(scale_folder, Subjects[subj]["name"] + '_Scale_Setup.xml')
    tree.write(runScale)

    # Run Scaling
    # os.chdir(scale_folder)
    # os.system('opensim-cmd run-tool ' + runScale)
    scale = osim.ScaleTool(runScale)  # Open scaling tool with new attributes

    # Run Scaling
    scale.run()


def readTRC(file):
    assert file.lower().endswith('.trc'), 'This is not a TRC file (*.trc).'
    # os.path.join(os.getcwd(),dir)
    # Import data from file
    with open(file, 'r') as fh:
        for _ in range(3):
            fh.readline()
        l = fh.readline().strip()

    markers = l.split('\t')
    data = pd.read_csv(file, skiprows=4, delimiter='\t')
    data = data.values

    if np.sum(data[:, -1]) == 0:
        data = np.delete(data, -1, axis=1)

    if np.sum(data[0, :]) == 0:
        data = np.delete(data, 0, axis=0)

    if markers[-1] == '':
        markers.pop()

    # get the subject ID
    subject_id = markers[2].split(':')[0]
    # remove the subject ID
    markers = [item for item in markers if item != ""]
    labels = np.array(markers[2:])
    col_names = [f'{label}_{axis}' for label in labels.flatten() for axis in ['x', 'y', 'z']]
    col_names.insert(0,'Frame#')
    col_names.insert(1,'Time')

    col_names = np.array(col_names)
    out = pd.DataFrame(data[:, :-1], columns=col_names.flatten())
    
    var_names =  col_names

    out.columns = var_names

    return out

def writeTRC(trc_table, file_path, filter_freq):
    if file_path is None:
        file_path = os.path.join(tempfile.gettempdir(), 'temp.trc')
    elif not file_path.endswith('.trc'):
        file_path += '.trc'
    if not trc_table.columns[0] == 'Frame#':
        raise ValueError('Unrecognized table format')

    if not (len(trc_table.columns) - 2) % 3 == 0:
        raise ValueError('Number of channels in the table does not match with 3D coordinate system')

    # frames = pd.DataFrame({'Frame#': np.arange(len(trc_table.Header))})
    # full_table = pd.concat([frames, trc_table], axis=1)

    fs = 1 / np.mean(np.diff(trc_table.Time))
    fc = 6 #cutoff frequency
    
    data = trc_table.values
    # if filter_freq > 0:
    #     data[:, 2:] = butterworth_filter(data[:, 2:],fc, fs, order=4)
    
    marker_names = [label.replace('_z', '') for label in trc_table.columns[1::3]]
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

def MakeVirtualMkr(StaticMarkerFile, file2write, file, dir):
    if file2write is None:
        file2write = f"{os.path.splitext(StaticMarkerFile)[0]}_Virtual.trc"

    # # Load Static TRC file
    # trcAdapter = osim.TRCFileAdapter()

    # trcAdapter.extendRead(StaticMarkerFile)
    
    trc = readTRC(StaticMarkerFile)
    MkrNames = trc.columns
    trcData = trc.to_numpy()
    trcAdapter = osim.TRCFileAdapter()
    trc = osim.TimeSeriesTableVec3(StaticMarkerFile)
    #trc = trc.flatten(['x','y','z'])
    mrks = trc.getColumnLabels()
    mkr_num = trc.getNumColumns()
    
    VirtualHeaders = [
        'Mid.ASIS', 'Mid.PSIS', 'Mid.Pelvis', 'R.KJC', 'L.KJC', 'R.AJC', 'L.AJC',
        'R.AJC_Floor', 'L.AJC_Floor', 'R.Heel_Floor', 'L.Heel_Floor', 'R.MT1_Floor', 'L.MT1_Floor', 'R.MT5_Floor',
        'L.MT5_Floor', 'R.MidMT_Floor', 'L.MidMT_Floor'
    ]

    VH = [f'{header}_{axis}' for header in VirtualHeaders for axis in ['x', 'y', 'z']]
    markers = list(mrks) + VH

    # Make Virtual Markers
    Mid = {}
    L = {}
    R = {}

    # Mid ASIS
    S = ['LASIS', 'LASI']
    Ind = MkrNames.str.contains('|'.join(S))
    L['ASIS'] = trcData[:, Ind]
    S = ['RASIS', 'RASI']
    Ind = MkrNames.str.contains('|'.join(S))
    R['ASIS'] = trcData[:, Ind]
    A = np.stack([L['ASIS'], R['ASIS']], axis=0)
    Mid['ASIS'] = np.mean(A, axis=0)

    # Mid PSIS
    S = ['LPSIS', 'LPSI']
    Ind = MkrNames.str.contains('|'.join(S))
    L['PSIS'] = trcData[:, Ind]
    S = ['RPSIS', 'RPSI']
    Ind = MkrNames.str.contains('|'.join(S))
    R['PSIS'] = trcData[:, Ind]
    A = np.stack([L['PSIS'], R['PSIS']], axis=0)
    Mid['PSIS'] = np.mean(A, axis=0)

    # Mid pelvis
    A = np.stack([Mid['ASIS'], Mid['PSIS']], axis=0)
    Mid['Pelvis'] = np.mean(A, axis=0)

    # Knee joint center
    S = ['LLEPI', 'LKNEL']
    Ind = MkrNames.str.contains('|'.join(S))
    L['Knee'] = trcData[:, Ind]
    S = ['LMEPI', 'LKNEM']
    Ind = MkrNames.str.contains('|'.join(S))
    L['MKnee'] = trcData[:, Ind]
    A = np.stack([L['Knee'], L['MKnee']], axis=0)
    L['KJC'] = np.mean(A, axis=0)

    S = ['RLEPI', 'RKNEL']
    Ind = MkrNames.str.contains('|'.join(S))
    R['Knee'] = trcData[:, Ind]
    S = ['RMEPI', 'RKNEM']
    Ind = MkrNames.str.contains('|'.join(S))
    R['MKnee'] = trcData[:, Ind]
    A = np.stack([R['Knee'], R['MKnee']], axis=0)
    R['KJC'] = np.mean(A, axis=0)

    # Ankle joint center
    S = ['LLMAL', 'LANKL']
    Ind = MkrNames.str.contains('|'.join(S))
    L['Ankle'] = trcData[:, Ind]
    S = ['LMMAL', 'LANKM']
    Ind = MkrNames.str.contains('|'.join(S))
    L['MAnkle'] = trcData[:, Ind]
    A = np.stack([L['Ankle'], L['MAnkle']], axis=0)
    L['AJC'] = np.mean(A, axis=0)

    S = ['RLMAL', 'RANKL']
    Ind = MkrNames.str.contains('|'.join(S))
    R['Ankle'] = trcData[:, Ind]
    S = ['RMMAL', 'RANKM']
    Ind = MkrNames.str.contains('|'.join(S))
    R['MAnkle'] = trcData[:, Ind]
    A = np.stack([R['Ankle'], R['MAnkle']], axis=0)
    R['AJC'] = np.mean(A, axis=0)

    # For floor markers, set Y coords to 0 (putting them on the floor)
    # AJC floor
    L['AJC_Floor'] = L['AJC'].copy()
    L['AJC_Floor'][1] = 0
    R['AJC_Floor'] = R['AJC'].copy()
    R['AJC_Floor'][1] = 0

    # Heel floor
    S = ['LCALC', 'LHEE']
    Ind = MkrNames.str.contains('|'.join(S))
    L['Heel'] = trcData[:, Ind]
    L['Heel_Floor'] = L['Heel'].copy()
    L['Heel_Floor'][1] = 0
    S = ['RCALC', 'RHEE']
    Ind = MkrNames.str.contains('|'.join(S))
    R['Heel'] = trcData[:, Ind]
    R['Heel_Floor'] = R['Heel'].copy()
    R['Heel_Floor'][1] = 0

    # MT1 floor
    S = ['L1ST', 'LMT1']
    Ind = MkrNames.str.contains('|'.join(S))
    L['MT1'] = trcData[:, Ind]
    L['MT1_Floor'] = L['MT1'].copy()
    L['MT1_Floor'][1] = 0
    S = ['R1ST', 'RMT1']
    Ind = MkrNames.str.contains('|'.join(S))
    R['MT1'] = trcData[:, Ind]
    R['MT1_Floor'] = R['MT1'].copy()
    R['MT1_Floor'][1] = 0

    # MT5 floor
    S = ['L5TH', 'LMT5']
    Ind = MkrNames.str.contains('|'.join(S))
    L['MT5'] = trcData[:, Ind]
    L['MT5_Floor'] = L['MT5'].copy()
    L['MT5_Floor'][1] = 0
    S = ['R5TH', 'RMT5']
    Ind = MkrNames.str.contains('|'.join(S))
    R['MT5'] = trcData[:, Ind]
    R['MT5_Floor'] = R['MT5'].copy()
    R['MT5_Floor'][1] = 0

    # MidMT floor
    A = np.stack([L['MT1_Floor'], L['MT5_Floor']], axis=0)
    L['MidMT_Floor'] = np.mean(A, axis=0)
    A = np.stack([R['MT1_Floor'], R['MT5_Floor']], axis=0)
    R['MidMT_Floor'] = np.mean(A, axis=0)


    # Export static trial with virtual makers to new TRC file
    VirtualData = pd.DataFrame({
        'Mid.ASIS_x': Mid['ASIS'][:, 0],
        'Mid.ASIS_y': Mid['ASIS'][:, 1],
        'Mid.ASIS_z': Mid['ASIS'][:, 2],
        'Mid.PSIS_x': Mid['PSIS'][:, 0],
        'Mid.PSIS_y': Mid['PSIS'][:, 1],
        'Mid.PSIS_z': Mid['PSIS'][:, 2],
        'Mid.Pelvis_x': Mid['Pelvis'][:, 0],
        'Mid.Pelvis_y': Mid['Pelvis'][:, 1],
        'Mid.Pelvis_z': Mid['Pelvis'][:, 2],
        'R.KJC_x': R['KJC'][:, 0],
        'R.KJC_y': R['KJC'][:, 1],
        'R.KJC_z': R['KJC'][:, 2],
        'L.KJC_x': L['KJC'][:, 0],
        'L.KJC_y': L['KJC'][:, 1],
        'L.KJC_z': L['KJC'][:, 2],
        'R.AJC_x': R['AJC'][:, 0],
        'R.AJC_y': R['AJC'][:, 1],
        'R.AJC_z': R['AJC'][:, 2],
        'L.AJC_x': L['AJC'][:, 0],
        'L.AJC_y': L['AJC'][:, 1],
        'L.AJC_z': L['AJC'][:, 2],
        'R.AJC_Floor_x': R['AJC_Floor'][:, 0],
        'R.AJC_Floor_y': R['AJC_Floor'][:, 1],
        'R.AJC_Floor_z': R['AJC_Floor'][:, 2],
        'L.AJC_Floor_x': L['AJC_Floor'][:, 0],
        'L.AJC_Floor_y': L['AJC_Floor'][:, 1],
        'L.AJC_Floor_z': L['AJC_Floor'][:, 2],
        'R.Heel_Floor_x': R['Heel_Floor'][:, 0],
        'R.Heel_Floor_y': R['Heel_Floor'][:, 1],
        'R.Heel_Floor_z': R['Heel_Floor'][:, 2],
        'L.Heel_Floor_x': L['Heel_Floor'][:, 0],
        'L.Heel_Floor_y': L['Heel_Floor'][:, 1],
        'L.Heel_Floor_z': L['Heel_Floor'][:, 2],
        'R.MT1_Floor_x': R['MT1_Floor'][:, 0],
        'R.MT1_Floor_y': R['MT1_Floor'][:, 1],
        'R.MT1_Floor_z': R['MT1_Floor'][:, 2],
        'L.MT1_Floor_x': L['MT1_Floor'][:, 0],
        'L.MT1_Floor_y': L['MT1_Floor'][:, 1],
        'L.MT1_Floor_z': L['MT1_Floor'][:, 2],
        'R.MT5_Floor_x': R['MT5_Floor'][:, 0],
        'R.MT5_Floor_y': R['MT5_Floor'][:, 1],
        'R.MT5_Floor_z': R['MT5_Floor'][:, 2],
        'L.MT5_Floor_x': L['MT5_Floor'][:, 0],
        'L.MT5_Floor_y': L['MT5_Floor'][:, 1],
        'L.MT5_Floor_z': L['MT5_Floor'][:, 2],
        'R.MidMT_Floor_x': R['MidMT_Floor'][:, 0],
        'R.MidMT_Floor_y': R['MidMT_Floor'][:, 1],
        'R.MidMT_Floor_z': R['MidMT_Floor'][:, 2],
        'L.MidMT_Floor_x': L['MidMT_Floor'][:, 0],
        'L.MidMT_Floor_y': L['MidMT_Floor'][:, 1],
        'L.MidMT_Floor_z': L['MidMT_Floor'][:, 2],
    })
    
    VirtualData.index = trcData[:,0]
    # add left virtual markers to static TRC
    for key,mrkdata in L.items():
        marker = mrkdata
        markername = f"L.{key}"
        value = [osim.Vec3(marker[i,:]) for i in range(marker.shape[0])]
        trc.appendColumn(markername, osim.VectorVec3(value))
    # add right virtual markers to static TRC    
    for key,mrkdata in R.items():
        marker = mrkdata
        markername = f"R.{key}"
        value = [osim.Vec3(marker[i,:]) for i in range(marker.shape[0])]
        trc.appendColumn(markername, osim.VectorVec3(value))
    for key,mrkdata in Mid.items():
        marker = mrkdata
        markername = f"Mid.{key}"
        value = [osim.Vec3(marker[i,:]) for i in range(marker.shape[0])]
        trc.appendColumn(markername, osim.VectorVec3(value))
    # write to file
    osim.TRCFileAdapter().write(trc, file2write)


def IK(Settings, data_folder, osim_folder, subj_name, trial_name):
    '''Batch process OpenSim Inverse Kinematics (IK)
    Inputs: 
        Settings            a dictionary of parameter settings that controls processing operations
        data_folder         the file to run IK on
        subj_name           name of the subject
        trial_name          name of the trial
    Outputs:
        
    '''

    # ensure scale folder exists
    scale_folder = os.path.join(osim_folder, 'ScaleFiles')
    if not os.path.exists(scale_folder):
        raise FileNotFoundError('must scale model prior to running IK')
    model_file =  [x for x in os.listdir(scale_folder) if '_Scaled.osim' in x][0]

    # create IK folder
    ik_folder = os.path.join(osim_folder, 'IK')
    if not os.path.exists(ik_folder):
        os.mkdir(ik_folder)

    # state generic setup files
    setup_ik_file = [x for x in os.listdir(Settings['GenericPath']) if 'Setup_IK.xml' in x][0]
    mkr_file = [x for x in os.listdir(data_folder) if trial_name + '_OpenSim.trc' in x][0]

    # get start and stop times
    #ik_data = pd.read_csv(os.path.join(data_folder, mkr_file), sep='\t', skiprows=4)
    ik_data = pd.read_csv(os.path.join(data_folder, mkr_file), skiprows=4, sep='\t',encoding='utf-8')
    # ik_data = pd.read_csv(os.path.join(data_folder, mkr_file), header=5, sep = '\t', skiprows=lambda x: x == 5 and pd.isna(pd.read_csv(os.path.join(data_folder, mkr_file), header=None, nrows=1).iloc[0, 0]))
    start_time = ik_data.iloc[0,1]
    stop_time = ik_data.iloc[len(ik_data)-1,1]

    # copy ik setup to run
    tree = ET.parse(os.path.join(Settings['GenericPath'], setup_ik_file))
    root = tree.getroot()
    for elem1 in root:
        print('1 ' + elem1.tag + ':   ' + str(elem1.text))
        if elem1.tag == 'ScaleTool':
            elem1.attrib['name'] = subj_name
        for elem2 in elem1:
            if elem2.tag == 'marker_file':
                elem2.text = os.path.join(data_folder, mkr_file)
            if elem2.tag == 'model_file':
                elem2.text = os.path.join(scale_folder, model_file)
            if elem2.tag == 'output_motion_file':
                elem2.text = os.path.join(ik_folder, trial_name + '_IK.mot')
            if elem2.tag == 'results_directory':
                elem2.text = ik_folder
            if elem2.tag == 'input_directory':
                elem2.text = data_folder
            if elem2.tag == 'time_range':
                elem2.text = str(start_time) + ' ' + str(stop_time)   
            print('2 ' + elem2.tag + ':   ' + str(elem2.text))
            # for elem3 in elem2:
                # print('3 ' + elem3.tag + ':   ' + str(elem3.text))
                # for elem4 in elem3:
                #     print('4 ' + elem4.tag + ':   ' + str(elem4.text))
    
    # save scale setup as new file in scale folder
    runIK = os.path.join(ik_folder, trial_name + '_IK_Setup.xml')
    tree.write(runIK)
    ik = osim.InverseKinematicsTool(runIK)  # Open scaling tool with new attributes
    ik.run()

#%%
def ID(Settings, data_folder, osim_folder, subj_name, trial_name, trial_num,Subjects):
    '''Batch process OpenSim Inverse Dynamics (ID)
    Inputs: 
        Settings            a dictionary of parameter settings that controls processing operations
        data_folder         the file to run IK on
        subj_name           name of the subject
        trial_name          name of the trial
    Outputs:
        creates inverse dynamics files in 
    '''

    subj_trial = trial_name

    # ensure scale folder exists
    scale_folder = os.path.join(osim_folder, 'ScaleFiles')
    if not os.path.exists(scale_folder):
        raise FileNotFoundError('must scale model prior to running ID')
    model_file =  [x for x in os.listdir(scale_folder) if '_Scaled.osim' in x][0]

    # ensure IK folder exists
    ik_folder = os.path.join(osim_folder, 'IK')
    if not os.path.exists(ik_folder):
        raise FileNotFoundError('must scale model prior to running IK')
    ik_file = [x for x in os.listdir(ik_folder) if x == subj_trial + '_IK.mot'][0]

    # create ID folder
    id_folder = os.path.join(osim_folder, 'ID')
    if not os.path.exists(id_folder):
        os.mkdir(id_folder)

    # state generic setup files
    setup_id_file = [x for x in os.listdir(Settings['GenericPath']) if 'Setup_ID.xml' in x][0]
    mkr_file = [x for x in os.listdir(data_folder) if trial_name + '_OpenSim.trc' in x][0]
    force_file = [x for x in os.listdir(data_folder) if trial_name + '_OpenSimGRF.mot' in x][0]
    ext_lds_fileG = [x for x in os.listdir(Settings['GenericPath']) if 'ExternalLoads.xml' in x][0]

    # determine which FP goes to which foot using getGaitCycles
    # GC = getGaitCycles(os.path.join(data_folder, mkr_file), os.path.join(data_folder, force_file))

    # update external loads file
    tree = ET.parse(os.path.join(Settings['GenericPath'], ext_lds_fileG))
    root = tree.getroot()
    for elem1 in root:
        # print('1 ' + elem1.tag + ':   ' + str(elem1.text))
        for elem2 in elem1:
            if elem2.tag == 'datafile':
                elem2.text = os.path.join(data_folder, force_file)
            # if elem2.tag == 'external_loads_model_kinematics_file':
            #     elem2.text = os.path.join(ik_folder,ik_file)
            # print('2 ' + elem2.tag + ':   ' + str(elem2.text))
            for elem3 in elem2:
                # print('3 ' + elem3.tag + ':   ' + str(elem3.text))
                for elem4 in elem3:
                    if elem3.attrib['name'] == 'FP2_v': # set left forces
                        if elem4.tag == 'applied_to_body':
                            elem4.text = 'calcn_r'
                        if elem4.tag == 'force_identifier':
                            elem4.text = 'ground_force_1_v'
                        if elem4.tag == 'point_identifier':
                            elem4.text = 'ground_force_1_p'
                        if elem4.tag == 'torque_identifier':
                            elem4.text = 'ground_force_1_m'
                        if elem4.tag == 'data_source_name':
                            elem4.text = force_file
                    if elem3.attrib['name'] == 'FP1_v': # set right forces
                        if elem4.tag == 'applied_to_body':
                            elem4.text = 'calcn_l'
                        if elem4.tag == 'force_identifier':
                            elem4.text = 'ground_force_2_v'
                        if elem4.tag == 'point_identifier':
                            elem4.text = 'ground_force_2_p'
                        if elem4.tag == 'torque_identifier':
                            elem4.text = 'ground_force_2_m'
                        if elem4.tag == 'data_source_name':
                            elem4.text = force_file
                    # print('4 ' + elem4.tag + ':   ' + str(elem4.text))
    ext_lds_file = os.path.join(id_folder, subj_trial + '_ExtLds.xml')
    tree.write(ext_lds_file)

    # get start and stop times
    mkr_data = pd.read_csv(os.path.join(data_folder, mkr_file), sep='\t', skiprows=4)
    start_time = mkr_data.iloc[0,1]
    stop_time = mkr_data.iloc[len(mkr_data)-1,1]

    # copy id setup to run
    tree = ET.parse(os.path.join(Settings['GenericPath'], setup_id_file))
    root = tree.getroot()
    for elem1 in root:
        # print('1 ' + elem1.tag + ':   ' + str(elem1.text))
        for elem2 in elem1:
            if elem2.tag == 'coordinates_file':
                elem2.text = os.path.join(ik_folder, ik_file)
            if elem2.tag == 'model_file':
                elem2.text = os.path.join(scale_folder, model_file)
            if elem2.tag == 'external_loads_file':
                elem2.text = ext_lds_file
            if elem2.tag == 'time_range':
                elem2.text = str(start_time) + ' ' + str(stop_time)   
            if elem2.tag == 'results_directory':
                elem2.text = id_folder
            if elem2.tag == 'output_gen_force_file':
                elem2.text = subj_trial + '_ID.sto'
            if elem2.tag == 'joints_to_report_body_forces':
                elem2.text = 'All'
            if elem2.tag == 'output_body_forces_file':
                elem2.text = subj_trial + '_body_forces.sto'
            # print('2 ' + elem2.tag + ':   ' + str(elem2.text))
    # save id setup as new file in id folder
    runID = os.path.join(id_folder, subj_trial + '_ID_Setup.xml')
    tree.write(runID)
    # Run ID
    id = osim.InverseDynamicsTool(runID)  # Open scaling tool with new attributes
    id.run()
