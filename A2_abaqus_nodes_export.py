#!/user/bin/
# -* - coding:UTF-8 -*-

from pydoc import pathdirs
from runpy import run_path
from odbAccess import *
from abaqus import *
import sys
import os

pathDir = os.getcwd()
jobname = 'test1'
odbName = jobname + '.odb'
path = pathDir + '\\' + odbName
odb = openOdb(path=path, readOnly=True)

assembly = odb.rootAssembly
numNodes = numElements = 0
for name, instance in assembly.instances.items():
    print(len(instance.nodes))
    print(len(instance.elements))
    with open(pathDir + '\\' + '11111nodes.txt', 'w') as info:
        for node in instance.nodes:
            info.write(str(node.label) + ',' + str(node.coordinates[0]) + ',' + str(node.coordinates[1]) + ',' + str(node.coordinates[2]) + '\n')
            with open(pathDir + '\\' + '1111111elements.txt', 'w') as info1:
                for element in instance.elements:
                    for nodeNum in element.connectivity:
                        info1.write(str(element.label) + ',' +str(element.type)+','+ str(nodeNum) + '\n')
