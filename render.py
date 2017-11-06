#!/usr/bin/env python

import numpy as np
import vtk

# Setup

a = 0.5  # black hole angular momentum
fname = "renderdata.npy"
color_ray = [1,1,1] # r,g,b values 0 to 1
color_bg = [0,0,0]

rays = np.load(fname)
points = vtk.vtkPoints()
lines = vtk.vtkCellArray()

for ray in rays:
    for point in ray:
        points.InsertNextPoint(point)
        
# NOTE: Main bottleneck (use vtk c++ if too slow)
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
sphere.SetRadius(1+np.sqrt(1-a*a))
sphere.SetPhiResolution(16)
sphere.SetThetaResolution(16)

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
sphereActor.GetProperty().SetOpacity(1.)
sphereActor.GetProperty().SetColor([0.,0.,0.])

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
window.SetSize(800, 800)

# Start
iren.Initialize()
iren.Start()