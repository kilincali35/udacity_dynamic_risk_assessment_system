"""
This module will go to /sourcedata folder, to get files there, and then merge them as a final dataset.

Also it will push the details of this process into a text file and save it as a log.

author: ali.kilinc
date: 07/02/23
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
ingested_final_data = config['ingested_final_data']
record_name = config['record_name']
detailed_record_name = config['record_name_detailed']

dateTimeObj = datetime.now()
thetimenow = str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)

#############Function for data ingestion
def merge_multiple_dataframe():
    
    file_records = []
    filenames = os.listdir(os.getcwd()+input_folder_path)
    final_dataframe = pd.DataFrame()
    for name in filenames:
        currentdf = pd.read_csv(os.getcwd()+input_folder_path+name)
        final_dataframe = final_dataframe.append(currentdf).reset_index(drop=True)
        final_dataframe = final_dataframe.drop_duplicates()
        file_records.append(name)
        file_records.append(len(currentdf.index))
        
    final_dataframe.to_csv(os.getcwd()+output_folder_path+ingested_final_data)
    
    return final_dataframe, file_records

final_df, file_records = merge_multiple_dataframe()

############ Record the ingestion
def record_ingestion(file_records):
    
    #Using a for open module means append, not overwrite
    MyFile = open(os.getcwd()+output_folder_path+detailed_record_name,'a')
    MyFile.write('.........' + thetimenow + '...............')
    MyFile.write('\n')
    for item in file_records:
        MyFile.write(str(item))
        MyFile.write('\n')
    MyFile.write(os.getcwd()+output_folder_path+ingested_final_data)
    MyFile.write('\n')
    MyFile.write('\n')
    MyFile.write('........................')
    MyFile.close()
    
    #It saves only latest ingested files, not the historical record
    MyFile2 = open(os.getcwd()+output_folder_path+record_name,'w')
    for item in file_records:
        MyFile2.write(str(item))
        MyFile2.write('\n')
    MyFile2.close()

def main():
    final_df, file_records = merge_multiple_dataframe()
    record_ingestion(file_records)
    
if __name__ == '__main__':
    main()