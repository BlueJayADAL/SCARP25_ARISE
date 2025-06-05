# exercise_forms.py
# Utility functions to check exercise form based on coordinates and angles

BOTH = 0
LEFT = 1
RIGHT = 2

def check_back_straight(coords, angles):
    global WIDTH, HEIGHT
    # Check if back is straight by comparing angles of shoulders and hips
    if angles['left_shoulder'] is not None and angles['right_shoulder'] is not None and \
       angles['left_hip'] is not None and angles['right_hip'] is not None:
        shoulder_angle = (angles['left_shoulder'] + angles['right_shoulder']) / 2
        hip_angle = (angles['left_hip'] + angles['right_hip']) / 2
        if abs(shoulder_angle - hip_angle) > 20:  # Adjust threshold as needed
            return False
    return True

# Returns True if elbows are close to body, otherwise False
def check_elbows_close_to_body(coords, angles, side=RIGHT, direction='vertical'):
    global WIDTH, HEIGHT
    # Check if elbows are close to body by comparing elbow and shoulder positions
    if direction == 'vertical':
        if side != RIGHT:
            if coords[5] is not None and coords[7] is not None:
                if abs(coords[5][0] - coords[7][0]) > 50:
                    return False
        if side != LEFT:
            if coords[6] is not None and coords[8] is not None:
                if abs(coords[6][0] - coords[8][0]) > 50:
                    return False
    elif direction == 'horizontal':
        if side != RIGHT:
            if coords[5 ] is not None and coords[7] is not None:
                if abs(coords[5][1] - coords[7][1]) > 50:
                    return False
        if side != LEFT:
            if coords[6] is not None and coords[8] is not None:
                if abs(coords[6][1] - coords[8][1]) > 50:
                    return False
    return True

# Returns True if arms are straight, otherwise False
def check_arms_straight(coords, angles, side=RIGHT):
    global WIDTH, HEIGHT
    # Check if arms are straight by comparing elbow angles
    if side != RIGHT:
        if angles['left_elbow'] is not None and angles['left_elbow'] < 135:
            return False
    if side != LEFT:
        if angles['right_elbow'] is not None and angles['right_elbow'] < 135:
            return False
    return True

# Returns True if head is up, otherwise False
def check_head_up(coords, angles):
    global WIDTH, HEIGHT
    # Check if head is up by comparing neck angle
    if angles['neck'] is not None and angles['neck'] < 20:  # Adjust threshold as needed
        return False
    return True

# Returns True if hips are back behind ankles (for squats), otherwise False
def check_hips_back(coords, angles, side=RIGHT):
    global WIDTH, HEIGHT
    # Check if hips are back by comparing hip angles
    if side == LEFT:
        if coords[11] is not None and coords[15] is not None:
            if coords[11][0] - coords[15][0] < 20:  # Adjust threshold as needed
                return False
    elif side == RIGHT:
        if coords[12] is not None and coords[16] is not None:
            if coords[12][0] - coords[16][0] > -20:  # Adjust threshold as needed
                return False
    return True

# Returns True if knees are over toes, otherwise False
def check_knees_over_toes(coords, angles, side=RIGHT):
    global WIDTH, HEIGHT
    # Check if knees are over toes by comparing knee and ankle positions
    if side != RIGHT:
        if coords[13] is not None and coords[15] is not None:
            if abs(coords[13][0] < coords[15][0]) - 50:  # Adjust threshold as needed
                return False
    elif side != LEFT:
        if coords[14] is not None and coords[16] is not None:
            if abs(coords[14][0] < coords[16][0]) - 50:  # Adjust threshold as needed
                return False
    return True

# Returns True if elbows are under shoulders, otherwise False
def check_elbows_under_shoulders(coords, angles, side=RIGHT):
    global WIDTH, HEIGHT
    # Check if elbows are under shoulders by comparing elbow and shoulder positions
    if side != RIGHT:
        if coords[5] is not None and coords[7] is not None:
            if abs(coords[5][0] - coords[7][0]) > 50:  # Adjust threshold as needed
                return False
    if side != LEFT:
        if coords[6] is not None and coords[8] is not None:
            if abs(coords[6][0] - coords[8][0]) > 50:  # Adjust threshold as needed
                return False
    return True

# Returns True if arms are level, otherwise False
def check_arms_level(coords, angles):
    global WIDTH, HEIGHT
    # Check if arms are level by comparing elbow positions
    if coords[7] is not None and coords[8] is not None:
        if abs(coords[7][1] - coords[8][1]) > 50:  # Adjust threshold as needed
            return False
    return True

# Returns True if feet are shoulder width apart, otherwise False
def check_feet_shoulder_width(coords, angles):
    global WIDTH, HEIGHT
    # Check if feet are shoulder width apart by comparing ankle positions
    if coords[5] is not None and coords[6] is not None and \
       coords[15] is not None and coords[16] is not None:
        if abs(coords[5][0] - coords[15][0]) < 50 or abs(coords[6][0] - coords[16][0]) < 50:
            return False
    return True

# Returns True if shoulders are level, otherwise False
def check_shoulders_level(coords, angles):
    global WIDTH, HEIGHT
    # Check if shoulders are level by comparing shoulder positions
    if coords[5] is not None and coords[6] is not None:
        if abs(coords[5][1] - coords[6][1]) > 50:  # Adjust threshold as needed
            return False
    return True

# Returns True if shoulders are above hips, otherwise False
def check_shoulders_above_hips(coords, angles, side=RIGHT):
    global WIDTH, HEIGHT
    # Check if shoulders are above hips by comparing shoulder and hip positions
    if side != RIGHT:
        if coords[5] is not None and coords[11] is not None:
            if abs(coords[5][0] - coords[11][0]) > 50:  # Adjust threshold as needed
                return False
    if side != LEFT:
        if coords[6] is not None and coords[12] is not None:
            if abs(coords[6][0] - coords[12][0]) > 50:  # Adjust threshold as needed
                return False
    return True

# Returns True if knees are pointed out (for squats), otherwise False
# Assumes head-on view
def check_knees_pointed_out(coords, angles):
    global WIDTH, HEIGHT
    # Check if knees are pointed out by comparing knee positions
    if coords[11] is not None and coords[12] is not None and \
    coords[13] is not None and coords[14] is not None:
        if coords[11][0] > coords[13][0] or coords[12][0] < coords[14][0]:
            return False
    return True

# Retusns a list of bad form warnings, or an empty list if form is good
def check_bad_form(current_exercise, coords, angles, dims, side=RIGHT):
    '''
    possible warnings:

    KEEP_BACK_STRAIGHT
    KEEP_ELBOWS_CLOSE_TO_BODY
    KEEP_ARMS_STRAIGHT
    KEEP_HEAD_UP
    KEEP_HIPS_BACK
    KEEP_KNEES_OVER_TOES
    KEEP_ELBOWS_UNDER_SHOULDERS
    KEEP_ARMS_LEVEL
    KEEP_FEET_SHOULDER_WIDTH
    KEEP_SHOULDERS_LEVEL
    KEEP_SHOULDERS_ABOVE_HIPS
    KEEP_KNEES_POINTED_OUT

    MOVE_INTO_CAMERA_FRAME
    MOVE_AWAY_FROM_CAMERA
    FACE_CAMERA
    '''
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = dims

    bad_form_list = []

    if current_exercise == 'bicep curl':
        # Check if body is in camera frame
        for coord in coords[5:11]:
            if coord[0] < 10 or coord[1] < 10 or coord[0] > WIDTH-10 or coord[1] > HEIGHT-10:
                bad_form_list.append("MOVE_INTO_CAMERA_FRAME")
                break
        # Check if too close to camera
        if coords[9][0] < 10 and coords[10][0] > WIDTH-10 or \
            coords[10][0] < 10 and coords[9][0] > WIDTH-10 or \
            (coords[5][1] < 10 or coords[6][1] < 10) and (coords[7][1] > HEIGHT-10 or coords[8][1] > HEIGHT-10 or coords[9][1] > HEIGHT-10 or coords[10][1] > HEIGHT-10) or \
            coords[9][0] < 10 and (coords[5][0] > WIDTH-10 or coords[7][0] > WIDTH-10) or \
            (coords[6][0] < 10 or coords[8][0] < 10) and coords[10][0] > WIDTH-10:
            bad_form_list.append("MOVE_AWAY_FROM_CAMERA")
        # Check elbows below shoulders
        if not check_elbows_under_shoulders(coords, angles, side):
            bad_form_list.append("KEEP_ELBOWS_UNDER_SHOULDERS")
        # Check back straight
        if not check_back_straight(coords, angles):
            bad_form_list.append("KEEP_BACK_STRAIGHT")
        # Check elbows close to body
        if not check_elbows_close_to_body(coords, angles, side, direction='vertical'):
            bad_form_list.append("KEEP_ELBOWS_CLOSE_TO_BODY")
        # Checks shoulders level
        if not check_shoulders_level(coords, angles):
            bad_form_list.append("KEEP_SHOULDERS_LEVEL")

    elif current_exercise == 'squat':
        # Check if body is in camera frame
        for coord in coords[5:17]:
            if coord[0] < 10 or coord[1] < 10 or coord[0] > WIDTH-10 or coord[1] > HEIGHT-10:
                bad_form_list.append("MOVE_INTO_CAMERA_FRAME")
                break
        # Check if too close to camera
        if coords[15][0] < 10 and coords[16][0] > WIDTH-10 or \
            coords[16][0] < 10 and coords[15][0] > WIDTH-10 or \
            (coords[5][0] < 10 or coords[6][0] < 10) and (coords[15][0] > HEIGHT-10 or coords[16][0] > HEIGHT-10) or \
            (coords[15][0] < 10 or coords[16][0] < 10) and (coords[5][0] > HEIGHT-10 or coords[6][0] > HEIGHT-10) or \
            (coords[5][1] < 10 or coords[6][1] < 10) and (coords[15][1] > HEIGHT-10 or coords[16][1] > HEIGHT-10):
            bad_form_list.append("MOVE_AWAY_FROM_CAMERA")
        # Check hips back behind ankles
        if not check_hips_back(coords, angles, side):
            bad_form_list.append("KEEP_HIPS_BACK")
        # Check knees over toes
        if not check_knees_over_toes(coords, angles, side):
            bad_form_list.append("KEEP_KNEES_OVER_TOES")
        # Check shoulders level
        if not check_shoulders_level(coords, angles):
            bad_form_list.append("KEEP_SHOULDERS_LEVEL")
        # Check feet shoulder width apart
        if not check_feet_shoulder_width(coords, angles):
            bad_form_list.append("KEEP_FEET_SHOULDER_WIDTH")
        # Check knees pointed out
        if not check_knees_pointed_out(coords, angles):
            bad_form_list.append("KEEP_KNEES_POINTED_OUT")
        
    elif current_exercise == 'arm raise':
        # Check if body is in camera frame
        for coord in coords[5:11]:
            if coord[0] < 10 or coord[1] < 10 or coord[0] > WIDTH-10 or coord[1] > HEIGHT-10:
                bad_form_list.append("MOVE_INTO_CAMERA_FRAME")
                break
        # Check if not facing camera
        if coords[5][0] >= coords[6][0] or coords[11][0] >= coords[12][0]:
            bad_form_list.append("FACE_CAMERA")
        # Check if too close
        if coords[9][0] < 10 and coords[10][0] > WIDTH-10 or \
           (coords[9][1] < 10 or coords[10][1] < 10) and (coords[5][1] > HEIGHT-10 or coords[6][1] > HEIGHT-10) or \
            coords[0][1] < 10 and (coords[9][1] > HEIGHT-10 or coords[10][1] > HEIGHT-10):
            bad_form_list.append("MOVE_AWAY_FROM_CAMERA")
        # Check arms straight 
        if not check_arms_straight(coords, angles, side):
            bad_form_list.append("KEEP_ARMS_STRAIGHT")
        # Check shoulders above hips
        if not check_shoulders_above_hips(coords, angles, side):
            bad_form_list.append("KEEP_SHOULDERS_ABOVE_HIPS")
        # Check head up
        if not check_head_up(coords, angles):
            bad_form_list.append("KEEP_HEAD_UP")
        # Check if arms are level
        if not check_arms_level(coords, angles):
            bad_form_list.append("KEEP_ARMS_LEVEL")
        # Check shoulders above hips
        if not check_shoulders_above_hips(coords, angles, side):
            bad_form_list.append("KEEP_SHOULDERS_ABOVE_HIPS")
        
    elif current_exercise == 'lunge':
        # Check if body is in camera frame
        for coord in coords[5:17]:
            if coord[0] < 10 or coord[1] < 10 or coord[0] > WIDTH-10 or coord[1] > HEIGHT-10:
                bad_form_list.append("MOVE_INTO_CAMERA_FRAME")
                break
        # Check if too close to camera
        if coords[15][0] < 10 and coords[16][0] > WIDTH-10 or \
            coords[16][0] < 10 and coords[15][0] > WIDTH-10 or \
            (coords[5][1] < 10 or coords[6][1] < 10) and (coords[15][1] > HEIGHT-10 or coords[16][1] > HEIGHT-10):
            bad_form_list.append("MOVE_AWAY_FROM_CAMERA")
        # Check shoulders level
        if not check_shoulders_level(coords, angles):
            bad_form_list.append("KEEP_SHOULDERS_LEVEL")
        # Check shoulders above hips
        if not check_shoulders_above_hips(coords, angles, side):
            bad_form_list.append("KEEP_SHOULDERS_ABOVE_HIPS")
        # Check elbows under shoulders
        if not check_elbows_under_shoulders(coords, angles, side):
            bad_form_list.append("KEEP_ELBOWS_UNDER_SHOULDERS")
        # Check arms level
        if not check_arms_level(coords, angles):
            bad_form_list.append("KEEP_ARMS_LEVEL")
        
    return bad_form_list