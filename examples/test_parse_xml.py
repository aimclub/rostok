import xml.etree.ElementTree as ET

mytree = ET.parse('examples/material.xml')
myroot = mytree.getroot()
mat_NSC = myroot.find("ChMaterialSurfaceNSC")
for x in myroot[0][1].findall('SetFriction'):
    print(x.tag,x)