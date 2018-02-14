#TODO: Read and plot several orbits at once
import numpy as np
import vtk

# Setup Input
a = 0.99999  # black hole angular momentum
fname = "spinrenderdata.npy" # filename of .npy of ray trajectory

# Visual Preferences
radius_bh = 1+np.sqrt(1-a*a) # radius to draw the blackhole sphere
color_ray = [0,0,0] # light ray color (r,g,b values 0 to 1)
color_bg = [1,1,1] # background color
color_bh = [1,1,1] # black hole color
opacity_bh = 1.
window_size = [800,800]
rays = np.load(fname)
points = vtk.vtkPoints()
lines = vtk.vtkCellArray()

for ray in rays:
    for point in ray:
        points.InsertNextPoint(point)
        
# NOTE: This is the main bottleneck (use vtk c++ if too slow)
for i in range(rays.shape[0]*rays.shape[1]-1):
    if (i+1)%(rays.shape[1]) == 0: # do not connect distinct rays
        continue
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, i)
    line.GetPointIds().SetId(1, i + 1)
    lines.InsertNextCell(line)
    
# Configure Sources
linesPolyData = vtk.vtkPolyData()
linesPolyData.SetPoints(points)
linesPolyData.SetLines(lines)

sphere = vtk.vtkSphereSource()
sphere.SetCenter(0,0,0)
sphere.SetRadius(radius_bh)
sphere.SetPhiResolution(32)
sphere.SetThetaResolution(32)

# Configure Mappers
linesMapper = vtk.vtkPolyDataMapper()
linesMapper.SetInputData(linesPolyData)
sphereMapper = vtk.vtkPolyDataMapper()
sphereMapper.SetInputConnection(sphere.GetOutputPort())

# Configure Actors
linesActor = vtk.vtkActor()
linesActor.SetMapper(linesMapper)
linesActor.GetProperty().SetColor(color_ray)

sphereActor = vtk.vtkActor()
sphereActor.SetMapper(sphereMapper)
sphereActor.GetProperty().SetOpacity(opacity_bh)
sphereActor.GetProperty().SetColor(color_bh)

# Configure Renderer
ren = vtk.vtkRenderer()
window = vtk.vtkRenderWindow()
window.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(window)
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera()) 

ren.AddActor(linesActor)
ren.AddActor(sphereActor)

ren.SetBackground(color_bg)
window.SetSize(window_size)

# Start
iren.Initialize()
iren.Start()