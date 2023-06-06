#API outline


#Pre-Processing
model_to_analyze = read_model('cad_filename.ext')
primitives1d,primitives2d = get_geometry_primitives(model_to_analyze)
mat_primitives = get_material_primitives(model_to_analyze)

mesh1D = generate1Dmesh(primitives1d)
mesh2D,material_labels = generate2Dmesh(primitives2d)

material_tags = assign_material()

[Efield,Gfield,nufield,rhofield] = 

materialProps = [Efield,Gfield,nufield,rhofield]

#2D Analysis
StiffMat, Recovery = BAT_CAVE(mesh2D,materialProps)

#1D Analysis
Disp1D = BAT_WIINGS(mesh1D,loads,bcs)