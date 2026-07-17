"""CoppeliaSim helpers for rendering numbers, fractions, and ratios."""

# ============================================================
# 7-SEGMENT DISPLAY — Build visible numbers in CoppeliaSim
# ============================================================
# Each digit is built from thin cuboids arranged like a 7-segment LED.
# Segment layout:
#    _0_
#  |5   |1
#    _6_
#  |4   |2
#    _3_
#
# Segment definitions: (x_offset, z_offset, is_horizontal)
SEGMENT_MAP = {
    0: (0, 1, True),  # top
    1: (0.5, 0.5, False),  # top-right
    2: (0.5, -0.5, False),  # bottom-right
    3: (0, -1, True),  # bottom
    4: (-0.5, -0.5, False),  # bottom-left
    5: (-0.5, 0.5, False),  # top-left
    6: (0, 0, True),  # middle
}

# Which segments are ON for each digit
DIGIT_SEGMENTS = {
    "0": [0, 1, 2, 3, 4, 5],
    "1": [1, 2],
    "2": [0, 1, 3, 4, 6],
    "3": [0, 1, 2, 3, 6],
    "4": [1, 2, 5, 6],
    "5": [0, 2, 3, 5, 6],
    "6": [0, 2, 3, 4, 5, 6],
    "7": [0, 1, 2],
    "8": [0, 1, 2, 3, 4, 5, 6],
    "9": [0, 1, 2, 3, 5, 6],
}


def create_digit(sim, digit_char, position, scale=0.03, color=None):
    """
    Render a single digit (0-9) at position using 7-segment cuboids.

    Args:
        sim: CoppeliaSim sim object
        digit_char: string '0'-'9'
        position: [x, y, z] center of the digit
        scale: size multiplier
        color: [r, g, b] default white

    Returns: list of shape handles
    """
    if color is None:
        color = [1.0, 1.0, 1.0]

    segments = DIGIT_SEGMENTS.get(digit_char, [])
    handles = []

    h_len = scale * 1.0  # horizontal segment length
    h_thick = scale * 0.2  # segment thickness
    v_len = scale * 1.0  # vertical segment length

    for seg_id in segments:
        dx, dz, is_horiz = SEGMENT_MAP[seg_id]

        size = [h_len, h_thick, h_thick] if is_horiz else [h_thick, h_thick, v_len]

        handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
        sx = position[0] + dx * scale
        sy = position[1]
        sz = position[2] + dz * scale
        sim.setObjectPosition(handle, -1, [sx, sy, sz])
        sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
        handles.append(handle)

    return handles


def create_slash(sim, position, scale=0.03, color=None):
    """Render a '/' symbol as a thin tilted cuboid."""
    if color is None:
        color = [1.0, 1.0, 1.0]
    handle = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid, [scale * 0.2, scale * 0.2, scale * 2.2], 0
    )
    sim.setObjectPosition(handle, -1, position)
    sim.setObjectOrientation(handle, -1, [0, 0.5, 0])  # tilt
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
    return [handle]


def create_colon(sim, position, scale=0.03, color=None):
    """Render a ':' symbol as two small cubes."""
    if color is None:
        color = [1.0, 1.0, 1.0]
    handles = []
    for dz in [0.4, -0.4]:
        h = sim.createPrimitiveShape(
            sim.primitiveshape_cuboid, [scale * 0.25, scale * 0.25, scale * 0.25], 0
        )
        sim.setObjectPosition(h, -1, [position[0], position[1], position[2] + dz * scale])
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        handles.append(h)
    return handles


def create_equals(sim, position, scale=0.03, color=None):
    """Render '=' as two horizontal bars."""
    if color is None:
        color = [1.0, 1.0, 1.0]
    handles = []
    for dz in [0.3, -0.3]:
        h = sim.createPrimitiveShape(
            sim.primitiveshape_cuboid, [scale * 1.2, scale * 0.2, scale * 0.2], 0
        )
        sim.setObjectPosition(h, -1, [position[0], position[1], position[2] + dz * scale])
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        handles.append(h)
    return handles


def render_fraction(sim, numerator, denominator, position, scale=0.03, color=None):
    """
    Render a fraction like "3/6" at the given position.

    Args:
        numerator: int (e.g., 3)
        denominator: int (e.g., 6)
        position: [x, y, z]
        scale: size
        color: [r, g, b]

    Returns: list of all shape handles
    """
    handles = []
    num_str = str(numerator)
    den_str = str(denominator)

    # Calculate total width
    char_width = scale * 1.5
    total_chars = len(num_str) + 1 + len(den_str)  # digits + slash + digits
    start_x = position[0] - (total_chars - 1) * char_width / 2

    x = start_x
    # Numerator digits
    for ch in num_str:
        handles += create_digit(sim, ch, [x, position[1], position[2]], scale, color)
        x += char_width

    # Slash
    handles += create_slash(sim, [x, position[1], position[2]], scale, color)
    x += char_width

    # Denominator digits
    for ch in den_str:
        handles += create_digit(sim, ch, [x, position[1], position[2]], scale, color)
        x += char_width

    return handles


def render_equivalence(sim, num1, den1, num2, den2, position, scale=0.03, color=None):
    """
    Render "num1/den1 = num2/den2" at position.
    Returns list of all handles.
    """
    handles = []
    char_width = scale * 1.5

    # First fraction
    x_offset = -char_width * 4
    handles += render_fraction(
        sim,
        num1,
        den1,
        [position[0] + x_offset, position[1], position[2]],
        scale,
        color,
    )

    # Equals sign
    handles += create_equals(sim, position, scale, color)

    # Second fraction
    x_offset = char_width * 4
    handles += render_fraction(
        sim,
        num2,
        den2,
        [position[0] + x_offset, position[1], position[2]],
        scale,
        color,
    )
    return handles


def render_ratio(sim, a, b, position, scale=0.03, color=None):
    """Render a ratio like "8:2" at the given position."""
    handles = []
    a_str = str(a)
    b_str = str(b)
    char_width = scale * 1.5
    total_chars = len(a_str) + 1 + len(b_str)
    start_x = position[0] - (total_chars - 1) * char_width / 2

    x = start_x
    for ch in a_str:
        handles += create_digit(sim, ch, [x, position[1], position[2]], scale, color)
        x += char_width
    handles += create_colon(sim, [x, position[1], position[2]], scale, color)
    x += char_width
    for ch in b_str:
        handles += create_digit(sim, ch, [x, position[1], position[2]], scale, color)
        x += char_width
    return handles


def render_ratio_equivalence(sim, a1, b1, a2, b2, position, scale=0.03, color=None):
    """Render "a1:b1 = a2:b2" at position."""
    handles = []
    char_width = scale * 1.5
    x_offset = -char_width * 4
    handles += render_ratio(
        sim, a1, b1, [position[0] + x_offset, position[1], position[2]], scale, color
    )
    handles += create_equals(sim, position, scale, color)
    x_offset = char_width * 4
    handles += render_ratio(
        sim, a2, b2, [position[0] + x_offset, position[1], position[2]], scale, color
    )
    return handles


# ============================================================
# COMMON HELPERS
# ============================================================
def make_static(sim, handle):
    """Make an object static (no physics)."""
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)


def create_floor_plane(sim, size=2.0, color=None):
    """Create a large flat plane as the floor."""
    if color is None:
        color = [0.95, 0.95, 0.95]
    floor = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [size, size, 0.002], 0)
    sim.setObjectAlias(floor, "Floor")
    sim.setObjectPosition(floor, -1, [0, 0, 0.001])
    sim.setShapeColor(floor, None, sim.colorcomponent_ambient_diffuse, color)
    make_static(sim, floor)
    return floor
