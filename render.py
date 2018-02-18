import numpy as np
import vtk

# Setup Input
a = 0.0  # black hole angular momentum

# Visual Preferences

radius_star = 1000 			# radius to draw stars
radius_bh = radius_star*2 	# radius to draw the blackhole sphere
# event horizon radius of bh: 1+np.sqrt(1-a*a)
color_ray = [0.5,0.5,1] 	# light ray color (r,g,b values 0 to 1)
color_bg = [0,0,0] 			# background color
color_bh = [1,1,1] 			# black hole color
opacity_bh = 0.7
opacity_ray = 0.1
window_size = [800,800]

ani_stepsize = 10		# only sample every <stepsize> time steps
ani_framerate = 1		# real time between rendering frames (1/fps)
ani_recordsteps = 1000 	# total number of steps to record
ani_recordfps = 30		# fps of output video

def draw_trajectories(fname="orbitsxyz.npy"):
	# fname:	file name of .npy with (x,y,z) of all 
	# 			points for each trajectory
	if not fname.endswith(".npy"): fname+=".npy"
	rays = np.load(fname)
	points = vtk.vtkPoints()
	lines = vtk.vtkCellArray()

	for ray in rays:
	    for point in ray:
	        points.InsertNextPoint(point)
	        
	# NOTE: This will be the main bottleneck (use vtk c++ if too slow)
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
	linesActor.GetProperty().SetOpacity(opacity_ray)

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
	return

def ani_orbits(fname="orbitsani.npy",record=False,recordfname="movie.ogv"):
	# fname:	file name of .npy with (t,x,y,z) of all 
	# 			points for each trajectory
	if not fname.endswith(".npy"): fname+=".npy"
	data = np.load(fname)
	rays = data[:,:,1:]
	times = data[:,:,0]
	points = vtk.vtkPoints()
	lines = vtk.vtkCellArray()

	
	for ray in rays:
	    for point in ray:
	        points.InsertNextPoint(point)
	        
	# NOTE: This will be the main bottleneck (use vtk c++ if too slow)
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
	linesActor.GetProperty().SetOpacity(opacity_ray)

	sphereActor = vtk.vtkActor()
	sphereActor.SetMapper(sphereMapper)
	sphereActor.GetProperty().SetOpacity(opacity_bh)
	sphereActor.GetProperty().SetColor(color_bh)

	# All of the above, but for stars
	stars = [Star(rays[i],times[i]) for i in range(rays.shape[0])]
	# Configure Renderer
	ren = vtk.vtkRenderer()
	window = vtk.vtkRenderWindow()
	window.AddRenderer(ren)
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(window)
	iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

	# add actors
	ren.AddActor(linesActor)
	ren.AddActor(sphereActor)
	for s in stars: ren.AddActor(s.actor)
	ren.SetBackground(color_bg)
	window.SetSize(window_size)
	
	iren.Initialize()
	# set up timer events
	star_iter = StarIterator(stars,iren,times.shape[1],ani_stepsize,record,recordfname)
	iren.AddObserver('TimerEvent', star_iter.StepForward)
	iren.CreateRepeatingTimer(ani_framerate)

	iren.Start()
	return

class Star():
	def __init__(self,xyz,time,color=[1,1,0]):
		self.xyz = xyz
		self.time = time # NOTE: UNUSED!
		self.source = vtk.vtkSphereSource()
		self.source.SetCenter(xyz[0])
		self.source.SetRadius(radius_star)
		self.source.SetPhiResolution(32)
		self.source.SetThetaResolution(32)
		
		self.mapper = vtk.vtkPolyDataMapper()
		self.mapper.SetInputConnection(self.source.GetOutputPort())

		self.actor = vtk.vtkActor()
		self.actor.SetMapper(self.mapper)
		self.actor.GetProperty().SetOpacity(1)
		self.actor.GetProperty().SetColor(color)

class StarIterator():
	def __init__(self,stars,iren,maxstep,stepsize=1,record=False,fname="movie.ogv"):
		self.stars = stars
		self.step = 0
		self.maxstep = maxstep
		self.stepsize = stepsize
		self.recording = record
		if record:
			if not fname.endswith(".ogv"): fname+=".ogv"
			print ("WRITING NEW VID "+fname)
			self.w2if = vtk.vtkWindowToImageFilter()
			self.w2if.SetInput(iren.GetRenderWindow())
			self.writer = vtk.vtkOggTheoraWriter()
			self.writer.SetInputConnection(self.w2if.GetOutputPort())
			self.writer.SetFileName(fname)
			self.writer.Start()
			self.writer.SetRate(ani_recordfps)
			self.writer.SetQuality(2)

	def StepForward(self, iren, event):
		for star in self.stars:
			star.source.SetCenter(star.xyz[self.step])
		iren.GetRenderWindow().Render()
		self.step += self.stepsize
		if self.recording:
			self.w2if.Modified()
			self.w2if.Update()
			self.writer.Write()
			if self.step>1000:
				print("FINISHED WRITING VID")
				self.recording = False
				self.writer.End()
		if self.step > self.maxstep: # loop
			self.step = self.step%self.maxstep

	def SaveFrame(self,iren):
		# screenshot code:
		w2if = vtk.vtkWindowToImageFilter()
		w2if.SetInput(iren.GetRenderWindow())
		w2if.Update()
		writer = vtk.vtkAVIWriter()
		writer.SetFileName("movie/screenshot%i.png"%self.step)
		writer.SetInputConnection(w2if.GetOutputPort())
		writer.Write()
		return

#eg render orbits stored in orbitsani10.npy, generated by orbits.py
ani_orbits(fname="orbitsani10",record=True)