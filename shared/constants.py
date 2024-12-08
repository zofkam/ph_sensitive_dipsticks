# RGB calculation methods
METHOD_CENTER = 'center_point'
METHOD_MEDIAN = 'median'
METHOD_MEAN = 'mean'
METHOD_INNER_MEDIAN = 'median_inner'
METHOD_OUTER_MEDIAN = 'median_outer'
# Aruco marker IDs
TOP_LEFT_MARKER_ID = 303
TOP_RIGHT_MARKER_ID = 269
BOTTOM_LEFT_MARKER_ID = 441
BOTTOM_RIGHT_MARKER_ID = 190
# real distances of aruco markers
ARUCO_MARKER_SQUARE_SIDE = 9.525
# edge distance + marker size
HORIZONTAL_DISTANCE = 42.475 + ARUCO_MARKER_SQUARE_SIDE
# edge distance + marker size
VERTICAL_DISTANCE = 35.475 + ARUCO_MARKER_SQUARE_SIDE
# real size of the strip square
REAL_OBJECT_SIZE = 5.125
# default fallback value
DEFAULT_OBJECT_SIZE = 32
# HSV parameters
MIN_HSV_1 = (0, 130, 0)
MAX_HSV_1 = (180, 255, 255)
MIN_HSV_2 = (20, 20, 30)
MAX_HSV_2 = (110, 150, 100)
