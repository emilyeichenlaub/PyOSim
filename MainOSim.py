# -*- coding: utf-8 -*-
"""
Inputs:
    Subject folders with converted .trc and .mot files
    Subjects.json file
Outputs:
    Subject-specific scaled model
    Inverse kinematics
    Inverse dynamics
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
import OSimProcFunctions

sys.path.append('C:\\Users\\emily\\OneDrive\\Desktop\\Misc\\UNC Research\\OpenSim_Python')

if __name__ == "__main__":

    # Setup Directories
    CurrPath = os.getcwd()
    path = 'C:\\Users\\emily\\OneDrive\\Desktop\\Misc\\UNC Research\\BFUNC\\SMI'
    # Modular Settings
    settings = {
        'Site': 'SMI',
        'Scale': 'No',
        'IK': 'Yes',
        'ID': 'Yes',
    }
        
    # Set up OpenSim tools
    geopath = 'C:/OpenSim 4.4/Geometry'

    # Set up tool directories
    GenericFilePath = 'C:\\Users\\emily\\OneDrive\\Desktop\\Misc\\UNC Research\\OpenSim_Python\\OpenSimProcessingFiles'
    model = osim.Model(os.path.join(GenericFilePath,'gait2392_frontHingesKnee_BFUNC.osim'))
    osim.ModelVisualizer.addDirToGeometrySearchPaths(geopath)
    sys.path.append(GenericFilePath)
    GenericDir = os.listdir(GenericFilePath)
    settings["GenericPath"] = GenericFilePath
    settings["GenericDir"] = GenericDir
    
    # Set location of directories
    if os.path.exists(path):
        subjectPath = path
    else:
        subjectPath = input('Select Folder Containing Subject Data: ')
    json_file_path = os.path.join(subjectPath,'Subjects.json')
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            Subjects = json.load(json_file)
    
    # Set up paths
    os.chdir('C:\\Users\\emily\\OneDrive\\Desktop\\Misc\\UNC Research\\OpenSim_Python')

    if settings["Scale"] == 'Yes':
        for S in range(0, len(Subjects)):
            for t in range(0,len(Subjects[S]['Trials'])):
                if 'Static' in Subjects[S]['Trials'][t]['name'] or 'static' in Subjects[S]['Trials'][t]['name'] or 'STATIC' in Subjects[S]['Trials'][t]['name']:
                    trial_name = Subjects[S]['Trials'][t]['name']
                    subj_trial_folder = os.path.join(Subjects[S]['Trials'][0]['folder'], trial_name+ '.c3d')
                    osim_folder = Subjects[S]['Folders.OpenSimFolder']
                    print('Scaling Subject: ', Subjects[S],'   Trial:', trial_name)
                    OSimProcFunctions.Scale(Subjects,settings, subj_trial_folder, osim_folder, S)
            
    if settings["IK"] == 'Yes':
        for S in range(0, len(Subjects)):
            for T in range(0,len(Subjects[S]['Trials'])):
                subj_trial_folder = Subjects[S]['Folders.OpenSimFolder']
                osim_folder = Subjects[S]['Folders.OpenSimFolder']
                print('Running IK on Subject: ', S,'   Trial:', T)
                OSimProcFunctions.IK(settings, subj_trial_folder, osim_folder, Subjects[S]['name'], Subjects[S]['Trials'][T]['name'])
                
    if settings['ID'] == 'Yes':
        for S in range(0, len(Subjects)):
            for T in range(0,len(Subjects[S]['Trials'])):
                if 'Static' in Subjects[S]['Trials'][T]['name'] or 'STATIC' in Subjects[S]['Trials'][T]['name'] or 'static' in Subjects[S]['Trials'][T]['name']:
                    continue
                else:
                    subj_trial_folder = Subjects[S]['Folders.OpenSimFolder']
                    osim_folder = Subjects[S]['Folders.OpenSimFolder']
                    print('Running ID on Subject: ', Subjects[S]['name'],'   Trial:', Subjects[S]['Trials'][T]['name'])
                    OSimProcFunctions.ID(settings, subj_trial_folder, osim_folder, Subjects[S]['name'], Subjects[S]['Trials'][T]['name'],T,Subjects)
                    