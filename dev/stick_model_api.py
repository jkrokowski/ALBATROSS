from ALBATROSS.beam_model import BeamModel
from ALBATROSS.frame import Frame

Wingbeam = BeamModel(1Dmesh,)
Strutbeam = BeamModel(...)
Jurybeam = BeamModel(...)

Wingbeam.add_clamped_point(point)
Strutbeam.add_clamped_point(point2)

#
Wingbeam.add_connection(Strutbeam)

Tbw = Frame([Wingbeam,Strutbeam,Jurybeam])

# Tbw.add_rigid_joint(Wingbeam,wingbeam_pt,Strutbeam,strutbeam_pt)

Tbw.add_rigid_joint([Parent:pt,children:pts])