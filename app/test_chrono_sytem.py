import pychrono as chrono
import pychrono.irrlicht as chronoirr

mysystem1 = chrono.ChSystemNSC()
mysystem1.Set_G_acc(chrono.ChVectorD(0,-0.5,0))


# Define block types
mat = chrono.ChMaterialSurfaceNSC()
mat.SetFriction(0.5)
mat.SetDampingF(0.1)

# Add object to grab
obj = chrono.ChBodyEasyBox(0.2,0.2,0.6,1000,True,True,mat)
obj.SetCollide(True)
obj.SetPos(chrono.ChVectorD(0,1.2,0))
mysystem1.Add(obj)
sis1 = obj.GetSystem()

vis1 = chronoirr.ChVisualSystemIrrlicht()
vis1.AttachSystem(mysystem1)
vis1.SetWindowSize(1024,768)
vis1.SetWindowTitle('Grab demo')
vis1.Initialize()
vis1.AddCamera(chrono.ChVectorD(8, 8, -6))
vis1.AddTypicalLights()
print("1")
while vis1.Run():
    mysystem1.Update()
    mysystem1.DoStepDynamics(1e-3)
    vis1.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis1.Render()
    vis1.EndScene()


mysystem2 = chrono.ChSystemNSC()
mysystem2.Set_G_acc(chrono.ChVectorD(0,1.5,0))

obj.SetPos(chrono.ChVectorD(0,1.2,0))
mysystem2.Add(obj)
sis2 = obj.GetSystem()

vis2 = chronoirr.ChVisualSystemIrrlicht()
vis2.AttachSystem(mysystem2)
vis2.SetWindowSize(1024,768)
vis2.SetWindowTitle('Grab demo')
vis2.Initialize()
vis2.AddCamera(chrono.ChVectorD(8, 8, -6))
vis2.AddTypicalLights()

print("2")
while vis2.Run():
    mysystem2.Update()
    mysystem2.DoStepDynamics(1e-3)
    vis2.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis2.Render()
    vis2.EndScene()
    
    
print("1 again")
while vis1.Run():
    mysystem1.Update()
    mysystem1.DoStepDynamics(1e-3)
    vis1.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis1.Render()
    vis1.EndScene()
    print("end")