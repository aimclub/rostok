from pychrono import ChTriangleMeshConnected, ChVectorD
import open3d as o3d
from numpy import asarray

def o3d_to_chrono_trianglemesh(mesh: o3d.geometry.TriangleMesh) -> ChTriangleMeshConnected:
    """Converting the spatial mesh format from O3D to ChTriangleMeshConnected to create
    ChBodyEasyMesh simulation object
    Returns:
        chrono.ChTriangleMeshConnected: The spatial mesh for describing ChBodyEasyMesh
    """
    triangles = asarray(mesh.triangles)
    vertices = asarray(mesh.vertices)
    ch_mesh = ChTriangleMeshConnected()

    for item in triangles:
        vert1_index, vert2_index, vert3_index = item[0], item[1], item[2]
        ch_mesh.addTriangle(ChVectorD(vertices[vert1_index][0],
                                      vertices[vert1_index][1],
                                      vertices[vert1_index][2]),
                            ChVectorD(vertices[vert2_index][0],
                                      vertices[vert2_index][1],
                                      vertices[vert2_index][2]),
                            ChVectorD(vertices[vert3_index][0],
                                      vertices[vert3_index][1],
                                      vertices[vert3_index][2]))
    return ch_mesh
