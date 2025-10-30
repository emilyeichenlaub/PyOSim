"""
Inputs:
    .c3d files

Outputs:
    Subjects JSON file - compiles all subjects/trial data
        converts .c3d files to .trc and .mot files for OpenSim compatibility
        subject mass from static trial
        5 max flexion/extension torque values from biodex
        
        
Updated 12/2024 to include strength outcomes from biodex/humac
Updated 03/2025 to remove opensource method of reading in c3d files
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
#from pyomeca import Markers, Analogs, DataArrayAccessor
import OSimSetupFunctions


if __name__ == "__main__":

    # Setup Directories
    CurrPath = os.getcwd()
    path = 'C:\\Users\\emily\\OneDrive\\Desktop\\Misc\\UNC Research\\BFUNC\\SMI' #mocap data (.c3ds)
    strength_path = 'C:\\Users\\emily\\OneDrive\\Desktop\\Misc\\UNC Research\\BFUNC\\Strength Outcomes\\UGA' # biodex data

    # Modular Settings
    settings = {
        'Site': 'SMI',
        'ConvertC3D': 'Yes',
        'GaitEvents': 'No',
        'GetMass': 'Yes',
        'Strength': 'No',
        
        'MGH':'YZX',
        'UNC': 'XZY',
        'EMORY':'-XZY',
        'NEBRASKA':'YZX',
        'HSS':'XZY',
        'IOWA': 'YZX',
        'SMI': 'YZX',
        'OSU':'XZY'
    }
    
    # Set location of directories
    if os.path.exists(path):
        StudyFolder = path
    else:
        StudyFolder = input('Select Folder Containing Subject Data: ')
    
    SubjPath = os.listdir(StudyFolder)
    sys.path.append(StudyFolder)
    # Loop Through Subjects
    NumSubjFolders = sum(os.path.isdir(os.path.join(StudyFolder, item)) for item in SubjPath)
    SubjFolders = [os.path.isdir(os.path.join(StudyFolder, item)) for item in SubjPath]

    # Initialize subject JSON file
    json_file_path = os.path.join(path,'Subjects.json')
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            Subjects = json.load(json_file)
    else:
        Subjects = []
        
    SC = 0
        
    # Loop through each subject
    # for SubjLoop in range(0, len(SubjFolders)):  # first two items are garbage
    #     if not SubjFolders[SubjLoop]:
    #         continue  # skip iteration because it isn't a subject folder
    SubjLoop = 0
    for subj in SubjPath:  # Iterate over folder/file names in SubjPath
        
        subj_folder_path = os.path.join(StudyFolder, subj)
        if not os.path.isdir(subj_folder_path):
            continue  # Skip if not a folder
        
        # Define subject filename, folder, path, and directory
        subject_exists = any(subject['name'] == SubjPath[SubjLoop] for subject in Subjects)
        
        if not subject_exists:
            Subjects.append({'name': SubjPath[SubjLoop],
                             'Folders': {'folder': os.path.join(StudyFolder, SubjPath[SubjLoop]),
                                         'path': os.path.join(StudyFolder, SubjPath[SubjLoop]),
                                         'dir': os.listdir(os.path.join(StudyFolder, SubjPath[SubjLoop])),
                                         'strength folder':  os.path.join(strength_path, SubjPath[SubjLoop])},
                             'Folders.OpenSimFolder': None,
                             'Trials': []})
        SC += 1
        
        # Display which subject is being processed
        print('\nPROCESSING ', Subjects[SC - 1]['name'], '\n')
        
        # If non-existent, create OpenSim Folder to store data
        Subjects[SC - 1]['Folders.OpenSimFolder'] = os.path.join(Subjects[SC - 1]['Folders']['path'], 'OpenSim')
        if not os.path.exists(Subjects[SC - 1]['Folders.OpenSimFolder']):
            os.makedirs(Subjects[SC - 1]['Folders.OpenSimFolder'])
        sys.path.append(StudyFolder + SubjPath[SubjLoop])
        
        # Set up Loop through all Trials of Subject
        # Identify number of trials
        NumTrials = sum('.c3d' in item for item in Subjects[SC - 1]['Folders']['dir'])
        Trials = []
        TC = 0  # trial loop counter
        Trials2Skip = ['LHJC', 'L_HJC', 'RHJC', 'R_HJC', 'lhjc', 'rhjc']
        for j in range(len(Subjects[SC - 1]['Folders']['dir'])):
            if '.c3d' not in Subjects[SC - 1]['Folders']['dir'][j] or any(skip in Subjects[SC - 1]['Folders']['dir'][j] for skip in Trials2Skip):
                continue
            
            TrialName = Subjects[SC - 1]['Folders']['dir'][j][:-4] #take off file type
            Trials.append({'name': TrialName,
                           'subject': Subjects[SC - 1]['name'],
                           'folder': Subjects[SC - 1]['Folders']['path'],
                           'type': 'static' if 'Static' in TrialName or 'static' in TrialName or 'STATIC' in TrialName else 'walking',
                           'files': {'OpenSimTRC': TrialName + '_OpenSim.trc',
                                     'OpenSimGRF': TrialName + '_OpenSimGRF.mot',
                                     'c3d': TrialName + '.c3d'}})
            TC += 1  # on to the next trial (next TRC file)
            if 'Static' in TrialName  or 'static' in TrialName or 'STATIC' in TrialName:
                StaticTrial = TC-1
            Subjects[SC-1]["Trials"] = Trials

        if settings["ConvertC3D"] == 'Yes':
            for j in range(len(Trials)):                   
                os.chdir(Subjects[SC - 1]['Folders']['folder'])
                #convertFPdata_OpenSim(input_file, FPcal_data, directory, zero_file, Settings):
                OSimSetupFunctions.C3D2OpenSim(Trials[j]["files"]["c3d"], Trials[j]["folder"],settings,Trials,j);
            
        if settings["GetMass"] == 'Yes':
            i = StaticTrial
            print('\nGetting subject mass from static trial')
            # Load GRF
            Trials[i]["GRF"] = OSimSetupFunctions.LoadGRF([os.path.join(Trials[i]["folder"], 'OpenSim', Trials[i]["files"]["OpenSimGRF"])],Trials,i)
            # Define vGRF columns
            Cols = ['ground_force_1_vy', 'ground_force_2_vy']
            # Identify columns from the full dataset
            vGRFs = Trials[i]["GRF"][0]["AllData"][:, [Trials[i]["GRF"][0]["ColHeaders"].index(col) for col in Cols]]
            # Sum and average
            AvgvGRF = np.mean(np.sum(vGRFs, axis=1))
            # Calculate mass from Newtons
            Subjects[SC-1]["Mass"] = AvgvGRF / 9.81
            # Clear variables
            print('\n')
            Subjects[SC-1]["Trials"] = Trials
            
            
            
        # strength path is different than gait trials path
        if settings['Strength'] == 'Yes':
             # Set location of directories
             if os.path.exists(strength_path):
                 StudyFolder = strength_path
             else:
                 StudyFolder = input('Select Folder Containing Subject Data: ')
             
             SubjPath = os.listdir(StudyFolder)
             sys.path.append(StudyFolder)
             # Loop Through Subjects
             NumSubjFolders = sum(os.path.isdir(os.path.join(StudyFolder, item)) for item in SubjPath)
             SubjFolders = [os.path.isdir(os.path.join(StudyFolder, item)) for item in SubjPath]

             # Initialize subject JSON file if mo cap data hasn't been processed yet
             # json_file_path = os.path.join(path,'Subjects.json')
             # if os.path.exists(json_file_path):
             #     with open(json_file_path, 'r') as json_file:
             #         Subjects = json.load(json_file)
             # else:
             #     Subjects = []
                 
             # subject loop
                 
             if 'Strength Outcomes' not in Subjects[SubjLoop]:
                Subjects[SubjLoop]['Strength Outcomes'] = []
                Subjects[SubjLoop]['Strength Outcomes'] = {
                    'folder': os.path.join(StudyFolder, SubjPath[SubjLoop]),
                    'dir': os.listdir(os.path.join(StudyFolder, SubjPath[SubjLoop])),
                    '12 MO': {'involved':{'flex':[],'ext':[]},
                              'uninvolved': {'flex':[],'ext':[]}},
                    '24 MO': {'involved':{'flex':[],'ext':[]},
                              'uninvolved': {'flex':[],'ext':[]}}}
                
            
            # Display which subject is being processed
             print('\nPROCESSING ', Subjects[SC - 1]['name'], ' STRENGTH OUTCOMES\n')
             print('\n')
             sys.path.append(StudyFolder + SubjPath[SubjLoop])
            
             # Set up Loop through all Trials of Subject
             # Identify number of trials
             NumTrials = sum(item.endswith(('.csv', '.txt')) for item in os.listdir(Subjects[SC - 1]['Strength Outcomes']['folder']))
             strength_trials = []
             # label whether it was collected on humac or biodex for different analysis methods
             # csv = humac; txt = biodex
             if any(item.endswith('.CSV') for item in Subjects[SC - 1]['Strength Outcomes']['dir']):
                Subjects[SC - 1]['Strength Outcomes']['collection'] = 'humac'
                
                TC = 0  # trial loop counter
                for j in range(len(Subjects[SC - 1]['Strength Outcomes']['dir'])):
                    
                    TrialName = Subjects[SC - 1]['Strength Outcomes']['dir'][j][:-4] #take off file type
                    print('\n', TrialName, '\n')
                    if '12MO' in TrialName:
                        timepoint = '12 MO'
                    if '24MO' in TrialName:
                        timepoint = '24 MO'
                        
                    if 'LU' in TrialName or 'RU' in TrialName:
                        involve = 'uninvolved'
                    if 'LI' in TrialName or 'RI' in TrialName:
                        involve = 'involved'
                        
                        # open csv
                    df = pd.read_csv(os.path.join(StudyFolder, SubjPath[SubjLoop],Subjects[SC - 1]['Strength Outcomes']['dir'][j]))
                    pos = df['Position(Degrees)']
                    tor = df['Torque(Foot-Pounds)']
                    ext,flex = OSimSetupFunctions.extract_strength_data(pos, tor,'humac')
                  
                    Subjects[SubjLoop]['Strength Outcomes'][timepoint][involve]['ext'] = ext
                    Subjects[SubjLoop]['Strength Outcomes'][timepoint][involve]['flex'] = flex   
             else:
                Subjects[SC - 1]['Strength Outcomes']['collection'] = 'biodex'
                
                TC = 0  # trial loop counter
                for j in range(len(Subjects[SC - 1]['Strength Outcomes']['dir'])):
                    
                    TrialName = Subjects[SC - 1]['Strength Outcomes']['dir'][j][:-4] #take off file type
                    if '12MO' in TrialName:
                        timepoint = '12 MO'
                    if '24MO' in TrialName:
                        timepoint = '24 MO'
                        
                    if 'LU' in TrialName or 'RU' in TrialName:
                        involve = 'uninvolved'
                    if 'LI' in TrialName or 'RI' in TrialName:
                        involve = 'involved'
                        
                        # open csv
                    df = pd.read_csv(os.path.join(StudyFolder, SubjPath[SubjLoop],Subjects[SC - 1]['Strength Outcomes']['dir'][j]), delimiter=r'\s+',skiprows=4)
                    
                    pos = pd.to_numeric(df['POS'][1:])
                    tor = pd.to_numeric(df['TORQUE'][1:])
                    ext,flex = OSimSetupFunctions.extract_strength_data(pos, tor,'biodex')
                  
                    Subjects[SubjLoop]['Strength Outcomes'][timepoint][involve]['ext'] = ext
                    Subjects[SubjLoop]['Strength Outcomes'][timepoint][involve]['flex'] = flex
                                      
                SubjLoop += 1
             
               
        print('All Subjects Processed')


    class NumpyEncoder(json.JSONEncoder):
        # def default(self, obj):
        #     if isinstance(obj, np.ndarray):
        #         return obj.tolist()
        #     return super(NumpyEncoder, self).default(obj)
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
        
    
    # Save the list as a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(Subjects, json_file,cls=NumpyEncoder)
        
    
    
    print(f'Subjects saved to {json_file_path}')

