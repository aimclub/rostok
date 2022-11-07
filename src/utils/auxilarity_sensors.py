from engine.robot import Robot
import pychrono as chrono
from engine.node_render import ChronoBody, ChronoRevolveJoint
class RobotSensor:
    
    # FIXME: Change to correct method
    @staticmethod
    def mean_center(in_robot: Robot) -> chrono.ChVectorD:
        blocks = in_robot.block_map.values()
        body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        sum_cog_coord = chrono.ChVectorD(0,0,0) 
        bodies = list(body_block)
        for body in bodies:
            sum_cog_coord += body.body.GetFrame_COG_to_abs().GetPos()
        mean_center: chrono.ChVectorD = sum_cog_coord / len(bodies)
        return mean_center
    
    @staticmethod
    def sum_contact_forces_blocks(in_robot: Robot):
        blocks = in_robot.block_map
        body_block = filter(lambda x: isinstance(x[1],ChronoBody),blocks.items())
        contact_force_blocks = map(lambda x: (x[0], sum(x[1].list_n_forces)), body_block)
        return dict(contact_force_blocks)
    
    @staticmethod
    def abs_coord_COG_blocks(in_robot: Robot) -> dict[int, chrono.ChVectorD]:
        blocks = in_robot.block_map
        body_block = filter(lambda x: isinstance(x[1],ChronoBody),blocks.items())
        def cog_from_tuple(tupled): pos = tupled[1].body.GetPos(); return(tupled[0], [pos.x, pos.y, pos.z])
        coord_COG_blocks = map(cog_from_tuple, body_block)
        return dict(coord_COG_blocks)
    
    # FIXME: Current method return bool of contact, not number of contact surfaces
    @staticmethod
    def amount_contact_surfaces_blocks(in_robot: Robot) -> dict[int, int]:
        blocks = in_robot.block_map
        body_block = filter(lambda x: isinstance(x[1],ChronoBody),
                            blocks.items())
        
        num_contact_surfaces = map(lambda x: (x[0], int(not (sum(x[1].list_n_forces) == 0))),
                                   body_block)
        
        return dict(num_contact_surfaces)
    
    @staticmethod
    def joints_angle(in_robot: Robot):
        blocks = in_robot.block_map
        joint_blocks = filter(lambda x: isinstance(x[1],ChronoRevolveJoint),blocks.items())
        joints_angle_block = map(lambda x: (x[0], x[1].joint.GetMotorRot()), joint_blocks)
        return dict(joints_angle_block)