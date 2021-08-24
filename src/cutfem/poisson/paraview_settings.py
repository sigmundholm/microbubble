"""
This is (most of) the Paraview settings used when generating figures of the solutions for
the report.

This code can just be copy-pasted into the Paraview Python Shell, after the solution vtk's
are imported into the program. Use 'File->Export Scene' to save a PDF figure.
"""

from paraview.simple import *

rv = GetActiveView()
ag = rv.AxesGrid
ag.Visibility = 1

ag.XTitle = ''
ag.YTitle = ''

# ag.XTitleColor = [0.6, 0.6, 0.0]
# ag.XAxisLabels = [-0.5, 0.5, 2.5, 3.5]

ag.XLabelBold = 0
ag.YLabelBold = 0
ag.XLabelFontSize = 24
ag.YLabelFontSize = 24

source = GetActiveSource()
view = GetActiveView()

display = GetDisplayProperties(source, view)
display.SetScalarBarVisibility(view, True)

colorMap = GetColorTransferFunction('Plasma')

# get the scalar bar in a view (akin to GetDisplayProperties)
scalarBar = GetScalarBar(colorMap, view)

# Now, you can change properties on the scalar bar object.
scalarBar.TitleFontSize = 20

scalarBar.LabelFontSize = 20
