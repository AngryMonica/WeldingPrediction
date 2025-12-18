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

csv_temperature = './output_ALL_stress/' + 'all_temperature_nodes.csv'

with open(csv_temperature, 'w') as fp1:
    for i in range(1,17):
        pathDir = os.getcwd()
        jobname = 'test'+str(i)
        print(jobname)
        odbName = jobname + '.odb'
        path = pathDir + '\\' + odbName
        odb = openOdb(path=path, readOnly=True)
        step_name_list = ['Weld1', 'Weld2', 'Weld3', 'Weld4']
        Step_list = [odb.steps['Weld1'], odb.steps['Weld2'], odb.steps['Weld3'], odb.steps['Weld4']]
        nodes_set = odb.rootAssembly.instances['PART-3-1']  # 部件模块创建的节点集合
        col_name='test,step,increment,step_time,'
        # 循环追加列名
        for node in nodes_set.nodes:
            col_name += 'N'+str(node.label)+','
        col_name+='\n'
        fp1.write(col_name)
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