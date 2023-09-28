# script-version: 2.0
# Catalyst state generated using paraview version 5.11.1

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [1280, 824]
renderView2.InteractionMode = 'Selection'
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.CenterOfRotation = [10.002041462808847, 9.978474140167236, 10.046973206102848]
renderView2.HiddenLineRemoval = 1
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [42.41949355014824, 42.39592622750662, 42.46442529344225]
renderView2.CameraFocalPoint = [10.002041462808847, 9.978474140167236, 10.046973206102848]
renderView2.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 14.53234620584578
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView2.AxesGrid.Visibility = 1
renderView2.AxesGrid.ShowGrid = 1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView2)
layout1.SetSize(1280, 824)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView2)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
ippl_particle = PVTrivialProducer(registrationName='ippl_particle')

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from ippl_particle
ippl_particleDisplay = Show(ippl_particle, renderView2, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'charge'
chargeTF2D = GetTransferFunction2D('charge')

# get color transfer function/color map for 'charge'
chargeLUT = GetColorTransferFunction('charge')
chargeLUT.TransferFunction2D = chargeTF2D
chargeLUT.RGBPoints = [-0.15625, 0.231373, 0.298039, 0.752941, -0.1562347412109375, 0.865003, 0.865003, 0.865003, -0.156219482421875, 0.705882, 0.0156863, 0.14902]
chargeLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'charge'
chargePWF = GetOpacityTransferFunction('charge')
chargePWF.Points = [-0.15625, 0.0, 0.5, 0.0, -0.156219482421875, 1.0, 0.5, 0.0]
chargePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
ippl_particleDisplay.Representation = 'Point Gaussian'
ippl_particleDisplay.ColorArrayName = ['POINTS', 'charge']
ippl_particleDisplay.LookupTable = chargeLUT
ippl_particleDisplay.SelectTCoordArray = 'None'
ippl_particleDisplay.SelectNormalArray = 'None'
ippl_particleDisplay.SelectTangentArray = 'None'
ippl_particleDisplay.OSPRayScaleArray = 'charge'
ippl_particleDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
ippl_particleDisplay.SelectOrientationVectors = 'None'
ippl_particleDisplay.ScaleFactor = 1.9956522745753908
ippl_particleDisplay.SelectScaleArray = 'None'
ippl_particleDisplay.GlyphType = 'Arrow'
ippl_particleDisplay.GlyphTableIndexArray = 'None'
ippl_particleDisplay.GaussianRadius = 0.09978261372876954
ippl_particleDisplay.SetScaleArray = ['POINTS', 'charge']
ippl_particleDisplay.ScaleTransferFunction = 'PiecewiseFunction'
ippl_particleDisplay.OpacityArray = ['POINTS', 'charge']
ippl_particleDisplay.OpacityTransferFunction = 'PiecewiseFunction'
ippl_particleDisplay.DataAxesGrid = 'GridAxesRepresentation'
ippl_particleDisplay.PolarAxes = 'PolarAxesRepresentation'
ippl_particleDisplay.ScalarOpacityFunction = chargePWF
ippl_particleDisplay.ScalarOpacityUnitDistance = 1.3403262389439892
ippl_particleDisplay.OpacityArrayName = ['POINTS', 'charge']
ippl_particleDisplay.SelectInputVectors = [None, '']
ippl_particleDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
ippl_particleDisplay.ScaleTransferFunction.Points = [-0.15625, 0.0, 0.5, 0.0, -0.156219482421875, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
ippl_particleDisplay.OpacityTransferFunction.Points = [-0.15625, 0.0, 0.5, 0.0, -0.156219482421875, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for chargeLUT in view renderView2
chargeLUTColorBar = GetScalarBar(chargeLUT, renderView2)
chargeLUTColorBar.Title = 'charge'
chargeLUTColorBar.ComponentTitle = ''

# set color bar visibility
chargeLUTColorBar.Visibility = 1

# show color legend
ippl_particleDisplay.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView2, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView2_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1720, 1139]
pNG1.Writer.Format = 'PNG'
pNG1.Writer.ResetDisplay = 1

# ----------------------------------------------------------------
# restore active source
SetActiveSource(pNG1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GenerateCinemaSpecification = 1
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
