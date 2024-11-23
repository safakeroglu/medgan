"""
MIMIC-III dataset processor for medGAN
Builds a binary or count matrix suitable for training medGAN.

Usage: 
    python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output_file> <"binary"|"count">

Outputs:
    <output_file>.pids: Pickled list of unique Patient IDs
    <output_file>.matrix: Numpy float32 matrix (patients × diagnosis codes)
    <output_file>.types: Pickled dictionary mapping diagnosis codes to integers
"""

import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

def convert_to_icd9(dx_str: str) -> str:
    """Convert ICD9 code to standard format with decimal point."""
    if dx_str.startswith('E'):
        return f"{dx_str[:4]}.{dx_str[4:]}" if len(dx_str) > 4 else dx_str
    return f"{dx_str[:3]}.{dx_str[3:]}" if len(dx_str) > 3 else dx_str

def convert_to_3digit_icd9(dx_str: str) -> str:
    """Convert ICD9 code to 3-digit format."""
    if dx_str.startswith('E'):
        return dx_str[:4] if len(dx_str) > 4 else dx_str
    return dx_str[:3] if len(dx_str) > 3 else dx_str

def process_admissions(admission_file: Path) -> Tuple[Dict[int, List[int]], Dict[int, datetime]]:
    """Process admission data to create patient-admission and admission-date mappings."""
    print('Building pid-admission mapping, admission-date mapping')
    
    admissions = pd.read_csv(admission_file)
    pid_adm_map = {}
    adm_date_map = {}
    
    for _, row in admissions.iterrows():
        pid = int(row['SUBJECT_ID'])
        adm_id = int(row['HADM_ID'])
        adm_time = datetime.strptime(row['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        
        adm_date_map[adm_id] = adm_time
        pid_adm_map.setdefault(pid, []).append(adm_id)
    
    return pid_adm_map, adm_date_map

def process_diagnoses(diagnosis_file: Path, use_full_code: bool = False) -> Dict[int, List[str]]:
    """Process diagnosis data to create admission-diagnosis mapping."""
    print('Building admission-diagnosis mapping')
    
    diagnoses = pd.read_csv(diagnosis_file)
    adm_dx_map = {}
    
    for _, row in diagnoses.iterrows():
        adm_id = int(row['HADM_ID'])
        dx_str = row['ICD9_CODE'].strip("'\"")
        
        if use_full_code:
            dx_str = 'D_' + convert_to_icd9(dx_str)
        else:
            dx_str = 'D_' + convert_to_3digit_icd9(dx_str)
        
        adm_dx_map.setdefault(adm_id, []).append(dx_str)
    
    return adm_dx_map

def create_patient_sequences(
    pid_adm_map: Dict[int, List[int]],
    adm_date_map: Dict[int, datetime],
    adm_dx_map: Dict[int, List[str]]
) -> Tuple[List[int], List[List[datetime]], List[List[List[str]]]]:
    """Create ordered sequences of patient visits."""
    print('Building patient visit sequences')
    
    pid_seq_map = {
        pid: sorted([(adm_date_map[adm_id], adm_dx_map[adm_id]) 
                    for adm_id in adm_ids])
        for pid, adm_ids in pid_adm_map.items()
    }
    
    pids = []
    dates = []
    seqs = []
    
    for pid, visits in pid_seq_map.items():
        pids.append(pid)
        dates.append([visit[0] for visit in visits])
        seqs.append([visit[1] for visit in visits])
    
    return pids, dates, seqs

def create_diagnosis_matrix(
    seqs: List[List[List[str]]], 
    binary: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Create the final diagnosis matrix and code mapping."""
    print('Converting sequences to integer codes and creating matrix')
    
    # Create diagnosis code mapping
    types = {}
    new_seqs = []
    
    for patient in seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in visit:
                if code not in types:
                    types[code] = len(types)
                new_visit.append(types[code])
            new_patient.append(new_visit)
        new_seqs.append(new_patient)
    
    # Create the matrix
    matrix = np.zeros((len(new_seqs), len(types)), dtype=np.float32)
    
    for i, patient in enumerate(new_seqs):
        for visit in patient:
            for code in visit:
                if binary:
                    matrix[i, code] = 1.0
                else:
                    matrix[i, code] += 1.0
    
    return matrix, types

def main():
    if len(sys.argv) != 5:
        print("Usage: python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output_file> <binary|count>")
        sys.exit(1)
    
    admission_file = Path(sys.argv[1])
    diagnosis_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])
    binary_count = sys.argv[4]
    
    if binary_count not in ['binary', 'count']:
        print('Error: Last argument must be either "binary" or "count"')
        sys.exit(1)
    
    # Process the data
    pid_adm_map, adm_date_map = process_admissions(admission_file)
    adm_dx_map = process_diagnoses(diagnosis_file)
    pids, dates, seqs = create_patient_sequences(pid_adm_map, adm_date_map, adm_dx_map)
    matrix, types = create_diagnosis_matrix(seqs, binary=(binary_count == 'binary'))
    
    # Save outputs
    print(f'Saving outputs to {output_file}.*')
    with open(f'{output_file}.pids', 'wb') as f:
        pickle.dump(pids, f)
    with open(f'{output_file}.matrix', 'wb') as f:
        pickle.dump(matrix, f)
    with open(f'{output_file}.types', 'wb') as f:
        pickle.dump(types, f)
    
    print(f'Created matrix with shape: {matrix.shape}')
    print(f'Number of unique diagnosis codes: {len(types)}')

if __name__ == '__main__':
    main()
