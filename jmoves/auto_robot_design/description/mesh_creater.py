import manifold3d as m3d
import trimesh
import numpy as np

# Helper to convert a Manifold into a Trimesh
def manifold2trimesh(manifold):
  mesh = manifold.to_mesh()

  if mesh.vert_properties.shape[1] > 3:
    vertices = mesh.vert_properties[:, :3]
    colors = (mesh.vert_properties[:, 3:] * 255).astype(np.uint8)
  else:
    vertices = mesh.vert_properties
    colors = None

  return trimesh.Trimesh(
    vertices=vertices, faces=mesh.tri_verts, vertex_colors=colors
  )


# Helper to display interactive mesh preview with trimesh
def showMesh(mesh):
  scene = trimesh.Scene()
  scene.add_geometry(mesh)
  # scene.add_geometry(trimesh.creation.axis())
  display(scene.show())

if __name__ == "__main__":
    from auto_robot_design.generator.topologies.bounds_preset import (
        get_preset_by_index_with_bounds,
    )
    from auto_robot_design.description.mechanism import JointPoint2KinematicGraph, KinematicGraph
    from auto_robot_design.description.builder import (
        ParametrizedBuilder,
        URDFLinkCreater3DConstraints,
        jps_graph2pinocchio_robot_3d_constraints,
    )
    import meshcat
    from pinocchio.visualize import MeshcatVisualizer
    from auto_robot_design.pinokla.closed_loop_kinematics import (
        closedLoopProximalMount,
    )

    builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)

    gm = get_preset_by_index_with_bounds(5)
    x_centre = gm.generate_central_from_mutation_range()
    graph_jp = gm.get_graph(x_centre)

    kinematic_graph = JointPoint2KinematicGraph(graph_jp)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    kinematic_graph.define_link_frames()
    
    links = kinematic_graph.nodes()
    
    for link in links: 
        in_joints = [j for j in link.joints if j.link_in == link]
        out_joints = [j for j in link.joints if j.link_out == link]
        
        num_joint = len(link.joints)
        
        if num_joint == 1:
            
        elif num_joint == 2:
            
        elif num_joint > 2:
    

