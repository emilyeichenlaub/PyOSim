# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:50:43 2023

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
import pyomeca
import matplotlib.pyplot as plt
import shutil
import xml.etree.ElementTree as ET
import re
import numpy as np
import datetime
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

    MakeVirtualMkr(static_folder, StaticVirtualFile, Subjects[subj]["Trials"][0]["files"]["OpenSimTRC"], Subjects[subj]["Folders.OpenSimFolder"])
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

    # get subject mass from average force during static trial
    static_force_file = static_file[:-4] + 'GRF.mot'
    static_forces = pd.read_csv(static_force_file, skiprows=6, sep='\t')
    subject_mass = round((static_forces['FP1_vy'].mean() + static_forces['FP2_vy'].mean()) / 9.81, 2)

    # copy scale setup to run
    tree = ET.parse(os.path.join(Settings['GenericPath'], setup_scale_file))
    root = tree.getroot()
    for elem1 in root:
        print('1 ' + elem1.tag + ':   ' + str(elem1.text))
        if elem1.tag == 'ScaleTool':
            elem1.attrib['name'] = subj_name
        for elem2 in elem1:
            if elem2.tag == 'mass':
                elem2.text = str(subject_mass)
            print('2 ' + elem2.tag + ':   ' + str(elem2.text))
            for elem3 in elem2:
                if elem3.tag == 'marker_file':
                    elem3.text = os.path.join(static_folder, virtual_file)
                if elem3.tag == 'marker_set_file':
                    elem3.text = os.path.join(Settings['GenericPath'], markerset_file)
                if elem3.tag == 'model_file':
                    elem3.text = os.path.join(Settings['GenericPath'], orig_model_file)
                if elem3.tag == 'output_model_file':
                    elem3.text = os.path.join(scale_folder, subj_name + '_scaled_model.osim')
                if elem3.tag == 'output_scale_file':
                    elem3.text = os.path.join(scale_folder, 'scale_set_applied.xml')
                if elem3.tag == 'output_marker_file':
                    elem3.text = os.path.join(scale_folder, 'marker_set_applied.xml')
                print('3 ' + elem3.tag + ':   ' + str(elem3.text))
                for elem4 in elem3:
                    print('4 ' + elem4.tag + ':   ' + str(elem4.text))
    
    # save scale setup as new file in scale folder
    runScale = os.path.join(scale_folder, subj_name + '_Scale_Setup.xml')
    tree.write(runScale)

    # Run Scaling
    os.chdir(scale_folder)
    os.system('opensim-cmd run-tool ' + runScale)



def readTRC(file):
    assert file.lower().endswith('.trc'), 'This is not a TRC file (*.trc).'
    # os.path.join(os.getcwd(),dir)
    # Import data from file
    with open(file, 'r') as fh:
        for _ in range(3):
            fh.readline()
        l = fh.readline().strip()

    markers = l.split('\t')
    data = np.loadtxt(file, delimiter='\t', skiprows=5)

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
    col_names.insert(0, 'Time')
    col_names = np.array(col_names)
    out = pd.DataFrame(data[:, 1:], columns=col_names.flatten())
    
    var_names =  col_names

    out.columns = var_names

    return out

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

def MakeVirtualMkr(StaticMarkerFile, file2write, file, dir):
    if file2write is None:
        file2write = f"{os.path.splitext(StaticMarkerFile)[0]}_Virtual.trc"

    # Load Static TRC file
    trc = readTRC(StaticMarkerFile)
    MkrNames = trc.columns
    trcData = trc.to_numpy()

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

    VirtualHeaders = [
        'Mid.ASIS', 'Mid.PSIS', 'Mid.Pelvis', 'R.KJC', 'L.KJC', 'R.AJC', 'L.AJC',
        'R.AJC_Floor', 'L.AJC_Floor', 'R.Heel_Floor', 'L.Heel_Floor', 'R.MT1_Floor', 'L.MT1_Floor', 'R.MT5_Floor',
        'L.MT5_Floor', 'R.MidMT_Floor', 'L.MidMT_Floor'
    ]

    VH = [f'{header}_{axis}' for header in VirtualHeaders for axis in ['x', 'y', 'z']]
    MkrNames = list(MkrNames) + VH
    V = pd.concat([pd.DataFrame(trcData).reset_index(drop=True), VirtualData.reset_index(drop=True)], axis=1, ignore_index=True)
    V.columns = MkrNames
    V.columns.values[0] = 'Header'
    writeTRC(V, file2write,6);
    #writeTRC(V, 'FilePath', file2write)


def ABL_Scale(Settings, Subjects):
    GenericFilePath = Settings["GenericPath"]
    GenericDir = Settings["GenericDir"]

    for subj in range(0, len(Subjects)):
        # Create scale folder and copy over TRC and GRF files
        ScaleFolder = os.path.join(Subjects[subj]["Folders.OpenSimFolder"], "ScaleFiles")
        os.makedirs(ScaleFolder, exist_ok=True)
        StaticTrials = [trial["type"] == "static" for trial in Subjects[subj]["Trials"]]
        StaticTrial = [index for index, value in enumerate(StaticTrials) if value]
        # Make virtual markers and copy into scale folder
        StaticMarkerFile = os.path.join(Subjects[subj]["Folders.OpenSimFolder"], Subjects[subj]["Trials"][0]["files"]["OpenSimTRC"])
        StaticVirtualFile = os.path.join(ScaleFolder, f"{Subjects[subj]['Trials'][0]['files']['OpenSimTRC'][:-4]}_Virtual.trc")
        MakeVirtualMkr(StaticMarkerFile, StaticVirtualFile, Subjects[subj]["Trials"][0]["files"]["OpenSimTRC"], Subjects[subj]["Folders.OpenSimFolder"])

        print(f"Scaling {Subjects[subj]['name']}")

        # Prep To Scale - Define inputs & outputs
        orig_model = os.path.join(GenericFilePath, [name for name in GenericDir if 'gait2392_frontHingesKnee_BFUNC.osim' in name][0])
        subj_model = os.path.join(ScaleFolder, 'gait2392_frontHingesKnee_BFUNC.osim')
        shutil.copy(orig_model, subj_model)

        # Load the model and initialize
        model = osim.Model(subj_model)
        model.initSystem()

        # Copy over original markerset file
        orig_markerset_file = os.path.join(GenericFilePath, [name for name in GenericDir if 'gait2392_Scale_MarkerSet_ABL_Virtual.xml' in name][0])
        mkr_set_file = os.path.join(ScaleFolder, f"{Subjects[subj]['name']}_MkrSet.xml")
        shutil.copy(orig_markerset_file, mkr_set_file)

        # Identify setup XML file
        orig_scale_setup_file = os.path.join(GenericFilePath, [name for name in GenericDir if 'Setup_Scale' in name][0])
        scale_tool, _, _ = xmlread(orig_scale_setup_file)

        # Change attributes in structure
        scale_tool["ScaleTool"][0]["CONTENT"]["mass"] = Subjects[subj]["Mass"]
        scale_tool["ScaleTool"][0]["ATTRIBUTE"]["name"] = Subjects[subj]["name"]

        # GenericModelMaker
        scale_tool["ScaleTool"][0]["CONTENT"]["GenericModelMaker"][0]["model_file"] = 'gait2392_frontHingesKnee_BFUNC.osim'
        scale_tool["ScaleTool"][0]["CONTENT"]["GenericModelMaker"][0]["marker_set_file"] = f"{Subjects[subj]['name']}_MkrSet.xml"

        # ModelScaler
        scale_tool["ScaleTool"][0]["CONTENT"]["ModelScaler"][0]["marker_file"] = f"{Subjects[subj]['Trials'][StaticTrials][0]['files']['OpenSimTRC'][:-4]}_Virtual.trc"
        output_model_file = f"{Subjects[subj]['name']}_Scaled.osim"
        scale_tool["ScaleTool"][0]["CONTENT"]["ModelScaler"][0]["output_model_file"] = output_model_file

        # MarkerPlacer
        scale_tool["ScaleTool"][0]["CONTENT"]["MarkerPlacer"][0]["marker_file"] = f"{Subjects[subj]['Trials'][StaticTrials][0]['files']['OpenSimTRC'][:-4]}_Virtual.trc"
        scale_tool["ScaleTool"][0]["CONTENT"]["MarkerPlacer"][0]["output_model_file"] = output_model_file
        scale_tool["ScaleTool"][0]["CONTENT"]["MarkerPlacer"][0]["output_motion_file"] = f"{Subjects[subj]['name']}_Scale_Motion.sto"
        scale_tool["ScaleTool"][0]["CONTENT"]["MarkerPlacer"][0]["output_marker_file"] = f"{Subjects[subj]['name']}_Scale_Markers.xml"
        scale_tool["ScaleTool"][0]["CONTENT"]["MarkerPlacer"][0]["output_scale_file"] = f"{Subjects[subj]['name']}_ScaleSet.xml"

        # Export XML to SetupDir (specific to each subject)
        setup_scale = os.path.join(ScaleFolder, f"{Subjects[subj]['name']}_Setup_Scale.xml")
        xml_write(setup_scale, scale_tool, "ScaleTool")

        scale = osim.ScaleTool(setup_scale)  # Open scaling tool with new attributes

        # Run Scaling
        scale.run()

        # Save printed results in OpenSim log file
        # FID = open('opensim.log')
        # TXT = textscan(FID, '%s')
        # Look for each subject's model scaling
        # Extract marker error locations
        # Save scale sets
        # Save marker errors
        # Subjects[subj]["Trials"][StaticTrials]["ScaleErr"]["TotalSqErr"] = float(TXT[1][Ind + 4])
        # Subjects[subj]["Trials"][StaticTrials]["ScaleErr"]["RMSErr"] = float(TXT[1][Ind + 9])
        # Subjects[subj]["Trials"][StaticTrials]["ScaleErr"]["MaxErr"] = float(TXT[1][Ind + 12])
        # Subjects[subj]["Trials"][StaticTrials]["ScaleErr"]["MaxMkr"] = TXT[1][Ind + 13]

        # print(f"Total Square Error = {TXT[1][Ind + 4]}")
        print(' ')
