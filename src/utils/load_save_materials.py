import xml.etree.ElementTree as ET
import pychrono as chrono

def string_xml2ChMaterial(xml_string, class_material: str = "ChMaterialSurfaceNSC"):
    
    methods_class_material = ET.fromstring(xml_string)
    
    chrono_material = getattr(chrono, class_material)()
    if methods_class_material is None:
        raise Exception("Didn't set parameters material for your class material")
    for method in methods_class_material:
        try:
            getattr(chrono_material,method.tag)(float(method.text))
        except AttributeError:
            raise Exception("Your class material don't have method {0}".format(method.tag))
    return chrono_material

def parse_dataset_material(str_type_material: str, file: str):
    """Parse file with materials and find the material in dataset

    Args:
        str_type_material (str): Name of the material
        file (_type_): Path to the xml-file

    Returns:
        xml.etree.ElementTree.Element: Element with info and configs material
    """
    xml_material = ET.parse(file)
    dataset_material = xml_material.getroot()
    info_material = dataset_material.find(str_type_material)
    return info_material
    
def create_chrono_material(str_type_material: str, file = None, class_material: str = "ChMaterialSurfaceNSC"):
    """Create material object from xml-file

    Args:
        str_type_material (str): Name of the material
        file (str): Path to the file with the material. Defaults to "ChMaterialSurfaceNSC".

        class_material (str, optional): Type of creating object. Defaults to "ChMaterialSurfaceNSC".

    Raises:
        Exception: If your material don't have parameters for the type of object
        Exception: If in the materials method config don't exist the method for the type of object material

    Returns:
        class_material: Object of material
    """
    info_material = parse_dataset_material(str_type_material,file)
    try:
        methods_class_material =  info_material.find(class_material)
    except AttributeError:
        raise AttributeError("File don't have the material {0}".format(str_type_material))
    
    chrono_material = getattr(chrono, class_material)()
    if methods_class_material is None:
        raise Exception("Didn't set parameters material for your class material")
    for method in methods_class_material:
        try:
            getattr(chrono_material,method.tag)(float(method.text))
        except AttributeError:
            raise Exception("Your class material don't have method {0}".format(method.tag))
    return chrono_material

def save_chrono_material(object_material, name_material: str, file: str):
    """Save object material with parameters in file

    Args:
        object_material (ChMaterialSurface): Chrono object of material (Not checking with other type object)
        name_material (str): Name of material 
        file (str): Path to the xml file for saving material information
    """
    getter_material = (method for method in set(dir(object_material)) - set(dir(object)) if method[0:3]=="Get")
    
    tree = ET.ElementTree(file=file)
    root = tree.getroot()
    child_root = root.find(name_material)
    if  child_root:
        children = root.find(name_material)
        
        str_class_material = str(object_material.__class__).split(".")[-1][0:-2]
        class_material_exsist = children.find(str_class_material)
        if class_material_exsist:
            element_class_material = class_material_exsist
            is_new_class_material = False
        else:
            is_new_class_material = True
    else:
        children = ET.Element(name_material)
        root.append(children)
        str_class_material = str(object_material.__class__).split(".")[-1][0:-2]
        is_new_class_material = True
    
    if is_new_class_material:
        element_class_material = ET.Element(str_class_material)
        children.append(element_class_material)
    
    for getter in getter_material:
        value = getattr(object_material, getter)()
        setter = "Set"+getter[3:len(getter)]

        if hasattr(object_material, setter):
            exist_params = element_class_material.find(setter)
            if exist_params is not None:
                exist_params.text = str(value)
            else:
                params_material = ET.SubElement(element_class_material, setter)
                params_material.text = str(value)
            
    tree = ET.ElementTree(root)
    
    with open(file, "wb") as fh:
        tree.write(fh)