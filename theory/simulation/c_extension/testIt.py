from optomechanics.theory.gaussian_beam import strongly_focussed
from theory.simulation.c_extension import fields_00

# print('\n',i00(10E-9,5E-9,0.9,4E-12, 0.8))
# print('\n', integral_00(10E-9,5E-9,0.9,4E-12, 0.8))
#
# print('\n',i01(10E-9,5E-9,0.9,4E-12, 0.8))
# print('\n', integral_01(10E-9,5E-9,0.9,4E-12, 0.8))
#
# print('\n',i02(10E-9,5E-9,0.9,4E-12, 0.8))
# print('\n', integral_02(10E-9,5E-9,0.9,4E-12, 0.8))
#
# print('\n',i10(10E-9,5E-9,0.9,4E-12, 0.8))
#
# print('\n',i11(10E-9,5E-9,0.9,4E-12, 0.8))
#
# print('\n',i12(10E-9,5E-9,0.9,4E-12, 0.8))
#
# print('\n',i13(10E-9,5E-9,0.9,4E-12, 0.8))
#
# print('\n',i14(10E-9,5E-9,0.9,4E-12, 0.8))


x = 10e-9
y = 5e-9
z_= 10e-9
focal_distance = 1e-3
NA = 0.8
e_field = -1
power = 70e-3
jones_vector_py = [1, 0]
wavelength = 1550e-9
n_1 = 1
n_2 = 1
filling_factor_ = 1
aperture_radius = 1e-3
width_inc = 1e-3
field = 0
print(strongly_focussed(x, y, z_, focal_distance, NA, None, power, jones_vector_py, wavelength, n_1, n_2, filling_factor_, aperture_radius, width_inc, False))
print(fields_00(x, y, z_, focal_distance, NA, e_field, power, jones_vector_py, wavelength, n_1, n_2, filling_factor_, aperture_radius, width_inc, field))
