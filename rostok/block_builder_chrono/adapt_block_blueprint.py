from rostok.block_builder_api.block_parameters import JointInputType
from rostok.block_builder_chrono.block_classes import JointInputTypeChrono


def convert_joint_input_type_to_chrono(joint_input_type: JointInputType) -> JointInputTypeChrono:
    convert_dict = {
        JointInputType.TORQUE: JointInputTypeChrono.TORQUE,
        JointInputType.VELOCITY: JointInputTypeChrono.VELOCITY,
        JointInputType.POSITION: JointInputTypeChrono.POSITION,
        JointInputType.UNCONTROL: JointInputTypeChrono.UNCONTROL
    }
    return convert_dict[joint_input_type]
