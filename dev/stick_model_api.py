

Wingbeam = BeamModel(1Dmesh,)
Strutbeam = BeamModel(...)
Jurybeam = BeamModel(...)

Wingbeam.add_clamped_point(point)
Strutbeam.add_clamped_point(point2)

#
Wingbeam.add_connection(Strutbeam)

Tbw = StickModel([Wingbeam,Strutbeam,Jurybeam])

# Tbw.add_rigid_joint(Wingbeam,wingbeam_pt,Strutbeam,strutbeam_pt)

Tbw.add_rigid_joint([Parent:pt,children:pts])