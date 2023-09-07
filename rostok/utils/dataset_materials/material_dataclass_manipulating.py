import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import pychrono as chrono


@dataclass
class Material:
    """Dataclass for materials
    """
    name: str
    type_class: str


@dataclass
class DefaultChronoMaterialNSC(Material):
    """Dataclass of default materials for chrono bodies
    """
    name: str = "default_NSC"
    type_class: str = "ChMaterialSurfaceNSC"
    Friction:float = 0.5
    Restitution:float = 0.15
    Compliance: float = 1e-6
    ComplianceT: float = 1e-6
    DampingF:float = 1e6
    
    def __hash__(self) -> int:
        return hash(("DefaultChronoMaterialNSC", self.Friction, self.Restitution, self.Compliance, self.ComplianceT, self.DampingF))

class DefaultChronoMaterialSMC():
    
    name: str = "default_SMC"
    type_class: str = "ChMaterialSurfaceSMC"
    Friction:float = 0.5
    Kn:float = 10000
    Kt:float = 10000
    Gn:float = 10000
    Gt:float = 10000
    Restitution:float = 0
    YoungModulus:float = 0
    
    def __hash__(self) -> int:
        return hash(("DefaultChronoMaterialSMC", self.Friction, self.Kn, self.Kt, self.Gn, self.Gt, self.Restitution, self.YoungModulus))


def struct_material2object_material(struct_material: Material, prefix_setter: str = "Set"):
    """Convert dataclass Materal from struct_material to some object material

    Args:
        struct_mateial (Material): Description some object material how structure.
        prefix_setter (str): Define prefix for setter of object material. Defaults to "Set".

    Raises:
        Exception: Raise when object don't have a setter method in structure

    Returns:
        struct_material.type_class: Object of material is defined dataclass Material
    """
    chrono_material = getattr(chrono, struct_material.type_class)()

    struct_material_attributes = set(dir(struct_material)) - set(["name", "type_class"]) - set(
        dir(dataclass))

    for method in struct_material_attributes:
        if method[0:2] != "__":
            value = getattr(struct_material, method)
            try:
                getattr(chrono_material, prefix_setter + method)(value)
            except AttributeError:
                raise Exception("Your class material don't have method {0}".format(prefix_setter +
                                                                                   method))
    return chrono_material


def string_xml2struct_material(xml_string):
    """Convert string xml to dataclass Material

    Args:
        xml_string (str): String of methods and class material

    Returns:
        Material: Dataclass of Material
    """

    xml_material = ET.fromstring(xml_string)

    class_material = xml_material[0]

    struct_material: Material = Material(xml_material.tag, class_material.tag)
    for method in class_material:
        setattr(struct_material, method.tag, float(method.text))

    return struct_material


def parse_dataset_material(str_type_material: str, file: str):
    """Parse file with materials and find the material in dataset

    Args:
        str_type_material (str): Name of the material
        file (str): Path to the xml-file

    Returns:
        xml.etree.ElementTree.Element: Element with info and configs material
    """
    xml_material = ET.parse(file)
    dataset_material = xml_material.getroot()
    info_material = dataset_material.find(str_type_material)
    return info_material


def create_struct_material_from_file(str_type_material: str,
                                     file=None,
                                     class_material: str = "ChMaterialSurfaceNSC"):
    """Create dataclass Material from xml-file

    Args:
        str_type_material (str): Name of the material
        file (str): Path to the file with the material. Defaults to "None".

        class_material (str, optional): Class of material. Defaults to "ChMaterialSurfaceNSC".

    Raises:
        Exception: If your material don't have parameters for the type of object

    Returns:
        struct_material(Material): Dataclass of material
    """
    struct_material = Material(str_type_material, class_material)
    info_material = parse_dataset_material(str_type_material, file)
    try:
        methods_class_material = info_material.find(class_material)
    except AttributeError:
        raise AttributeError("File don't have the material {0}".format(str_type_material))

    if methods_class_material is None:
        raise Exception("Didn't set parameters material for your class material")
    for method in methods_class_material:
        setattr(struct_material, method.tag, float(method.text))
    return struct_material


def save_object_material(object_material, name_material: str, file: str, prefix_getter="Get"):
    """Save object material with parameters in file

    Args:
        object_material (ChMaterialSurface): Chrono object of material
            (Not checking with other type object)
        name_material (str): Name of material
        file (str): Path to the xml file for saving material information
    """
    getter_material = (method for method in set(dir(object_material)) - set(dir(object))
                       if method[0:3] == prefix_getter)

    tree = ET.ElementTree(file=file)
    root = tree.getroot()
    child_root = root.find(name_material)
    if child_root:
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
        setter = getter[3:len(getter)]

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


if __name__ == "__main__":
    str_xml_material = """<test_mat>
                            <ChMaterialSurfaceNSC>}
                            <Friction>0.5</Friction>
                            <DampingF>0.1</DampingF>
                            </ChMaterialSurfaceNSC>
                        </test_mat>"""

    data_material1 = string_xml2struct_material(str_xml_material)
    chr_object = struct_material2object_material(data_material1)
    file = "./src/utils/dataset_materials/material.xml"
    data_material2 = create_struct_material_from_file("rubber", file)
    print("Done!")
