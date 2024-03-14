#!/usr/bin/env python

##Copyright 2009-2014 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array1OfPnt2d
from OCC.Core.Geom2dAPI import Geom2dAPI_PointsToBSpline
from OCC.Core.Geom2d import Geom2d_OffsetCurve

from OCC.Display.SimpleGui import init_display

display, start_display, add_menu, add_function_to_menu = init_display()


def curves2d_from_offset(event=None):
    """
    @param display:
    """
    pnt2d_array = TColgp_Array1OfPnt2d(1, 5)
    pnt2d_array.SetValue(1, gp_Pnt2d(-1,0))
    pnt2d_array.SetValue(2, gp_Pnt2d(0, 1))
    pnt2d_array.SetValue(3, gp_Pnt2d(1, 1.1))
    pnt2d_array.SetValue(4, gp_Pnt2d(3, .6))
    pnt2d_array.SetValue(5, gp_Pnt2d(5, 0))

    spline_1 = Geom2dAPI_PointsToBSpline(pnt2d_array).Curve()

    pnt2d_array1 = TColgp_Array1OfPnt2d(1, 5)
    pnt2d_array1.SetValue(1, gp_Pnt2d(-1,0))
    pnt2d_array1.SetValue(2, gp_Pnt2d(0, -1))
    pnt2d_array1.SetValue(3, gp_Pnt2d(1, -1.1))
    pnt2d_array1.SetValue(4, gp_Pnt2d(3, -.6))
    pnt2d_array1.SetValue(5, gp_Pnt2d(5, 0))

    spline_2 = Geom2dAPI_PointsToBSpline(pnt2d_array1).Curve()

    dist = .1
    offset_curve1 = Geom2d_OffsetCurve(spline_1, dist)
    offset_curve2 = Geom2d_OffsetCurve(spline_2, -dist)
    result = offset_curve1.IsCN(2)
    print("Offset curve yellow is C2: %r" % result)
    
    display.DisplayShape(spline_1)
    display.DisplayShape(offset_curve1, color="YELLOW")
    
    display.DisplayShape(spline_2)
    display.DisplayShape(offset_curve2, color="BLUE")


if __name__ == "__main__":
    curves2d_from_offset()
    start_display()
