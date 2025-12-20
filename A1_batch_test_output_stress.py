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

for i in range(2, 17):
    csv_stress = r'D:\Users\MXY\PycharmProjects\data\all_stress_elements'+r'\test'+str(i)+'_stress.csv'
    with open(csv_stress, 'w') as fp2:
        pathDir = os.getcwd()
        jobname = 'test' + str(i) + '_stress'
        print(jobname)
        odbName = jobname + '.odb'
        path = pathDir + '\\' + odbName
        odb = openOdb(path=path, readOnly=True)
        step_name_list = ['Step-weld1', 'Step-weld2', 'Step-weld3', 'Step-weld4']
        Step_list = [odb.steps['Step-weld1'], odb.steps['Step-weld2'], odb.steps['Step-weld3'], odb.steps['Step-weld4']]
        nodes_set = odb.rootAssembly.instances['PART-3-1']  # 部件模块创建的节点集合
        col_name='test,step,increment,step_time'
        # 循环追加列名
        for element in nodes_set.elements:
            col_name += ',E'+str(element.label)
        col_name+='\n'
        fp2.write(col_name)
        total_elements = len(nodes_set.elements)
        stress_data_array = np.zeros(total_elements)
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
                subset = frame.fieldOutputs['S'].getSubset(region=nodes_set)
                print(len(subset.values))
                for val in subset.values:
                    stress_data_array[val.elementLabel-1] = val.mises  # 或者 val.data
                for v in stress_data_array:
                    temp_str += ',' + str(v)
                temp_str += '\n'
                fp2.write(temp_str)
                print(jobname + ',' + step_name_list[index] + ',' + str(increment)+',nodeslength:'+str(len(stress_data_array)))

