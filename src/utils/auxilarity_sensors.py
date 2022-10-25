from engine.robot import Robot
import pychrono as chrono
from engine.node_render import ChronoBody

class RobotSensor:
    
    # TODO: Change to correct method
    @staticmethod
    def mean_center(in_robot: Robot):
        blocks = in_robot.block_map.values()
        body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        sum_cog_coord = chrono.ChVectorD(0,0,0) 
        bodies = list(body_block)
        for body in bodies:
            sum_cog_coord += body.body.GetFrame_COG_to_abs().GetPos()
        mean_center: chrono.ChVectorD = sum_cog_coord / len(bodies)
        return mean_center
    