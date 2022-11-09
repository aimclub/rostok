import pychrono as chrono
import pychrono.irrlicht as pchirr

length1 = 1
length2 = 2

mass = 1
inertia_matrix = chrono.ChVectorD(0.1,0.1,0.2)

sys = chrono.ChSystemNSC()

block1 = chrono.ChBody()
block3 = chrono.ChBody()
block2 = chrono.ChBody()

block1.SetMass(mass)
block3.SetMass(mass)
block2.SetMass(mass)

block1.SetInertiaXX(inertia_matrix)
block3.SetInertiaXX(inertia_matrix)
block2.SetInertiaXX(inertia_matrix)

box_asset1 = chrono.ChBoxShape()
box_asset1.GetBoxGeometry().Size = chrono.ChVectorD(0.1,0.1,length1)
box_asset2 = chrono.ChBoxShape()
box_asset2.GetBoxGeometry().Size = chrono.ChVectorD(0.1,0.1,length2)
box_asset3 = chrono.ChBoxShape()
box_asset3.GetBoxGeometry().Size = chrono.ChVectorD(0.1,0.1,0.1)

block1.AddVisualShape(box_asset1)
block3.AddVisualShape(box_asset3)
block2.AddVisualShape(box_asset2)

block1.GetVisualShape(0).SetColor(chrono.ChColor(0.1,0.2,0.3))
block3.GetVisualShape(0).SetColor(chrono.ChColor(0,0.5,0.2))
block2.GetVisualShape(0).SetColor(chrono.ChColor(0.1,0.5,0.7))

block1.SetBodyFixed(True)
block3.SetBodyFixed(True)
block2.SetBodyFixed(True)

block2.SetPos(chrono.ChVectorD(0.2,0,1))
block3.SetPos(chrono.ChVectorD(0.2,0,0))

sys.Add(block1)
sys.Add(block3)
sys.Add(block2)


vis = pchirr.ChVisualSystemIrrlicht()
vis.AttachSystem(sys)
vis.SetWindowSize(1024, 768)
vis.SetWindowTitle('test')
vis.Initialize()


vis.AddCamera(chrono.ChVectorD(2, 2, -3))
vis.AddTypicalLights()


while vis.Run():
    sys.Update()
    sys.DoStepDynamics(5e-3)
    vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
    vis.Render()

    vis.EndScene()
