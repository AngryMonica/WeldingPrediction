#!/user/bin/
# -* - coding:UTF-8 -*-

from pydoc import pathdirs
from runpy import run_path
from odbAccess import *
from abaqus import *
import sys
import os
import math
import numpy as np
from abaqusConstants import *

csv_stress = './output_ALL_stress/' + 't1_stress.csv'
csv_temperature = './output_ALL_stress/' + 't1_temperature.csv'

with open(csv_temperature, 'w') as fp1:
    for i in range(1,2):
        pathDir = os.getcwd()
        jobname = 'test'+str(i)
        print(jobname)
        odbName = jobname + '.odb'
        path = pathDir + '\\' + odbName
        odb = openOdb(path=path, readOnly=True)
        step_name_list = ['Weld1', 'Weld2', 'Weld3', 'Weld4']
        Step_list = [odb.steps['Weld1'], odb.steps['Weld2'], odb.steps['Weld3'], odb.steps['Weld4']]
        nodes_set = odb.rootAssembly.instances['PART-3-1']  # 部件模块创建的节点集合
        for index in range(len(Step_list)):
            for increment in range(len(Step_list[index].frames)):
                # 坐标针对节点
                frame = Step_list[index].frames[increment]
                coord = frame.fieldOutputs['NT11'].getSubset(region=nodes_set)
                step_time = frame.frameValue
                temp_str = ''
                temp_str += jobname + ',' + step_name_list[index] + ',' + str(increment)+',' + str(step_time)
                # coord = frame.fieldOutputs['NT11'].getSubset(region=nodes_set)
                # all node----instance['PART-1-1']
                # part node----node set['SET']
                fieldValues = coord.values
                for v in fieldValues:
                    temp_str += ',' + str(v.data)
                temp_str += '\n'
                fp1.write(temp_str)
                print(jobname + ',' + step_name_list[index] + ',' + str(increment)+',nodeslength:'+str(len(fieldValues)))

with open(csv_stress, 'w') as fp2:
    for i in range(1,17):
        pathDir = os.getcwd()
        jobname = 'test'+str(i)+'_stress'
        print(jobname)
        odbName = jobname + '.odb'
        path = pathDir + '\\' + odbName
        odb = openOdb(path=path, readOnly=True)
        step_name_list = ['Step-weld1', 'Step-weld2', 'Step-weld3', 'Step-weld4']
        Step_list = [odb.steps['Step-weld1'], odb.steps['Step-weld2'], odb.steps['Step-weld3'], odb.steps['Step-weld4']]
        nodes_set = odb.rootAssembly.instances['PART-3-1']  # 部件模块创建的节点集合
        all_node_labels = np.array([node.label for node in nodes_set.nodes])
        total_nodes = len(all_node_labels)
        label_to_index = {label: idx for idx, label in enumerate(all_node_labels)}
        stress_data_array = np.zeros(total_nodes)
        for index in range(len(Step_list)):
            for increment in range(1,len(Step_list[index].frames)):
                # 坐标针对节点
                frame = Step_list[index].frames[increment]
                step_time = frame.frameValue
                temp_str = ''
                temp_str += jobname + ',' + step_name_list[index] + ',' + str(increment) + ',' + str(step_time)
                # coord = frame.fieldOutputs['NT11'].getSubset(region=nodes_set)
                # coord = frame.fieldOutputs['S'].getSubset(region=nodes_set)
                # fieldValues = coord.values
                # for v in fieldValues:
                #     temp_str += ',' + str(v.mises)
                subset = frame.fieldOutputs['S'].getSubset(region=nodes_set, position=NODAL)
                if subset.values:
                    for val in subset.values:
                        idx = label_to_index.get(val.nodeLabel)
                        if idx is not None:
                            stress_data_array[idx] = val.mises  # 或者 val.data
                for v in stress_data_array:
                    temp_str += ',' + str(v)
                temp_str += '\n'
                fp2.write(temp_str)
                print(jobname + ',' + step_name_list[index] + ',' + str(increment)+',nodeslength:'+str(len(stress_data_array)))