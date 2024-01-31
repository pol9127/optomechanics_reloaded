from optomechanics.theory.abcd_matrices.abcd_gaussian_beam import LensSystem

def add_LD2297_C(lens_system, position):
    lens_system.add_curved_surface(1.778, -39.6e-3, absolute_pos=position)
    lens_system.add_curved_surface(1, 39.6e-3, absolute_pos=position + 3e-3)


def add_AC254_100_C(lens_system, position):
    lens_system.add_curved_surface(1.67, 32.1e-3, absolute_pos=position)
    lens_system.add_curved_surface(1.8052, -38e-3, absolute_pos=position + 6.5e-3)
    lens_system.add_curved_surface(1, 93.5e-3, absolute_pos=position + 8.3e-3)


def add_AC254_075_C(lens_system, position):
    lens_system.add_curved_surface(1.67, 26.4e-3, absolute_pos=position)
    lens_system.add_curved_surface(1.8052, -29.4e-3, absolute_pos=position + 7.6e-3)
    lens_system.add_curved_surface(1, 84.9e-3, absolute_pos=position + 9.4e-3)


def add_LA1509_C(lens_system, position):
    lens_system.add_curved_surface(1.5168, 51.5e-3, absolute_pos=position)
    lens_system.add_flat_surface(1, absolute_pos=position + 3.6e-3)


def add_LBF254_040_C(lens_system, position):
    lens_system.add_curved_surface(1.5168, 24.0e-3, absolute_pos=position)
    lens_system.add_curved_surface(1, -134.6e-3, absolute_pos=position + 6.5e-3)


def add_LBF254_050_C(lens_system, position):
    lens_system.add_curved_surface(1.5168, 30.1e-3, absolute_pos=position)
    lens_system.add_curved_surface(1, -172.0e-3, absolute_pos=position + 6.5e-3)


def add_LBF254_075_C(lens_system, position):
    lens_system.add_curved_surface(1.5168, 44.5e-3, absolute_pos=position)
    lens_system.add_curved_surface(1, -289.0e-3, absolute_pos=position + 5.0e-3)


def add_LBF254_100_C(lens_system, position):
    lens_system.add_curved_surface(1.5168, 60.0e-3, absolute_pos=position)
    lens_system.add_curved_surface(1, -353.3e-3, absolute_pos=position + 4.0e-3)


def add_LBF254_150_C(lens_system, position):
    lens_system.add_curved_surface(1.5168, 89.4e-3, absolute_pos=position)
    lens_system.add_curved_surface(1, -570.5e-3, absolute_pos=position + 4.0e-3)


def add_LBF254_200_C(lens_system, position):
    lens_system.add_curved_surface(1.5168, 121.5e-3, absolute_pos=position)
    lens_system.add_curved_surface(1, -684.5e-3, absolute_pos=position + 4.0e-3)
