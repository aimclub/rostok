from rostok.block_builder_api.block_blueprints import PrimitiveBodyBlueprint, RevolveJointBlueprint  
from rostok.block_builder_api.block_parameters import JointInputType,  FrameTransform, DefaultFrame
from rostok.utils.dataset_materials.material_dataclass_manipulating import (
    DefaultChronoMaterialNSC, struct_material2object_material)
from rostok.block_builder_chrono.block_classes import JointInputTypeChrono


# def convert_material_to_chrono(block_material: Material) -> DefaultChronoMaterialNSC:
#     ret = DefaultChronoMaterialNSC()
#     ret.DampingF = block_material.DampingF
#     ret.Friction = block_material.Friction
#     ret.Compliance = block_material.Compliance
#     return ret

def convert_joint_input_type_to_chrono(joint_input_type: JointInputType) -> JointInputTypeChrono:
    convert_dict = {JointInputType.TORQUE : JointInputTypeChrono.TORQUE,
                    JointInputType.VELOCITY : JointInputTypeChrono.VELOCITY,
                    JointInputType.POSITION : JointInputTypeChrono.POSITION,
                    JointInputType.UNCONTROL : JointInputTypeChrono.UNCONTROL}
    return convert_dict[joint_input_type]
