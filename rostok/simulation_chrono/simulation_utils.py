import pychrono as chrono
from rostok.block_builder_chrono.block_classes import ChronoEasyShapeObject


def calculate_covering_sphere(obj: ChronoEasyShapeObject):
    v1 = chrono.ChVectorD(0, 0, 0)
    v2 = chrono.ChVectorD(0, 0, 0)
    obj.body.GetTotalAABB(bbmin=v1, bbmax=v2)
    local_center = (v1 + v2) * 0.5
    radius = ((v2 - v1).Length()) * 0.5
    visual = chrono.ChSphereShape(radius)
    visual.SetOpacity(0.3)
    obj.body.AddVisualShape(visual, chrono.ChFrameD(local_center))
    if isinstance(obj.body, chrono.ChBodyAuxRef):
        cog_center = obj.body.GetFrame_REF_to_COG().TransformPointLocalToParent(local_center)
    else:
        cog_center = local_center

    return cog_center, radius

def calculate_covering_ellipsoid(obj: ChronoEasyShapeObject):
    v1 = chrono.ChVectorD(0, 0, 0)
    v2 = chrono.ChVectorD(0, 0, 0)
    obj.body.GetTotalAABB(bbmin=v1, bbmax=v2)
    local_center = (v1 + v2) * 0.5
    axis_x =  v2.x - v1.x
    axis_y =  v2.y - v1.y
    axis_z =  v2.z - v1.z
    visual = chrono.ChEllipsoidShape(axis_x, axis_y, axis_z)
    visual.SetOpacity(0.3)
    obj.body.AddVisualShape(visual, chrono.ChFrameD(local_center))
    if isinstance(obj.body, chrono.ChBodyAuxRef):
        cog_center = obj.body.GetFrame_REF_to_COG().TransformPointLocalToParent(local_center)
    else:
        cog_center = local_center

    return cog_center, (axis_x, axis_y, axis_z)


def set_covering_sphere_based_position(obj: ChronoEasyShapeObject,
                                       reference_point: chrono.ChVectorD = chrono.ChVectorD(
                                           0, 0, 0),
                                       direction: chrono.ChVectorD = chrono.ChVectorD(0, 1, 0)):
    center, radius = calculate_covering_sphere(obj)
    direction.Normalize()
    desired_position = reference_point + direction * radius
    shift = desired_position - obj.body.GetCoord().TransformPointLocalToParent(center)
    current_cog_pos = obj.body.GetPos()
    obj.body.SetPos(current_cog_pos + shift)

def set_covering_ellipsoid_based_position(obj: ChronoEasyShapeObject,
                                       reference_point: chrono.ChVectorD = chrono.ChVectorD(
                                           0, 0, 0),
                                       direction: chrono.ChVectorD = chrono.ChVectorD(0, 1, 0)):
    center, axis = calculate_covering_ellipsoid(obj)
    direction.Normalize()
    desired_position = reference_point + direction * axis[1] /2
    shift = desired_position - obj.body.GetCoord().TransformPointLocalToParent(center)
    current_cog_pos = obj.body.GetPos()
    obj.body.SetPos(current_cog_pos + shift)
