# exercise_forms.py
# Utility functions to check exercise form based on coordinates and angles

BOTH = 0
LEFT = 1
RIGHT = 2
EITHER = 3

def check_back_straight(coords, angles):
    # # Check if back is straight by comparing angles of shoulders and hips
    # if angles['left_shoulder'] is not None and angles['right_shoulder'] is not None and \
    #    angles['left_hip'] is not None and angles['right_hip'] is not None:
    #     shoulder_angle = (angles['left_shoulder'] + angles['right_shoulder']) / 2
    #     hip_angle = (angles['left_hip'] + angles['right_hip']) / 2
    #     if abs(shoulder_angle - hip_angle) > 20:  # Adjust threshold as needed
    #         return False
    # return True
    print("check_back_straight is not implemented correctly, may scrap")
    return True

# Returns True if elbows are close to body, otherwise False
def check_elbows_close_to_body(coords, angles, side=RIGHT, direction='vertical'):
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
    # Check if head is up by comparing neck angle
    if angles['neck'] is not None and abs(angles['neck']) > 20:  # Adjust threshold as needed
        return False
    return True

# Returns True if hips are back behind ankles (for squats), otherwise False
def check_hips_back_squat(coords, angles, side=RIGHT):
    # Check if hips are back by comparing hip angles
    if side == LEFT or side == EITHER:
        if coords[11] is not None and coords[15] is not None and angles['left_knee'] is not None:
            # Only check while in lowered position
            if angles['left_knee'] < 110 and coords[11][0] - coords[15][0] < 20:  # Adjust threshold as needed
                return False
    elif side == RIGHT or side == EITHER:
        if coords[12] is not None and coords[16] is not None and angles['right_knee'] is not None:
            # Only check while in lowered position
            if angles['right_knee'] < 110 and coords[12][0] - coords[16][0] > -20:  # Adjust threshold as needed
                return False
    return True

# Returns True if knees are over toes, otherwise False
def check_knees_over_toes_squat(coords, angles, side=RIGHT):
    # Check if knees are over toes by comparing knee and ankle positions
    if side == LEFT or side == EITHER:
        if coords[13] is not None and coords[15] is not None:
            if abs(coords[13][0] - coords[15][0]) > 75:  # Adjust threshold as needed
                return False
    elif side == RIGHT or side == EITHER:
        if coords[14] is not None and coords[16] is not None:
            if abs(coords[14][0] - coords[16][0]) > 75:  # Adjust threshold as needed
                return False
    return True

# Returns True if elbows are under shoulders, otherwise False
def check_elbows_under_shoulders(coords, angles, side=RIGHT):
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
    # Check if arms are level by comparing elbow positions
    if coords[7] is not None and coords[8] is not None:
        if abs(coords[7][1] - coords[8][1]) > 50:  # Adjust threshold as needed
            return False
    return True

# Returns True if feet are shoulder width apart, otherwise False
def check_feet_shoulder_width(coords, angles):
    # Check if feet are shoulder width apart by comparing ankle positions
    if coords[5] is not None and coords[6] is not None and \
       coords[15] is not None and coords[16] is not None:
        if abs(coords[5][0] - coords[15][0]) > 50 or abs(coords[6][0] - coords[16][0]) > 50:
            return False
    return True

# Returns True if shoulders are level, otherwise False
def check_shoulders_level(coords, angles):
    # Check if shoulders are level by comparing shoulder positions
    if coords[5] is not None and coords[6] is not None:
        if abs(coords[5][1] - coords[6][1]) > 50:  # Adjust threshold as needed
            return False
    return True

# Returns True if shoulders are above hips, otherwise False
def check_shoulders_above_hips(coords, angles, side=RIGHT):
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
def check_knees_pointed_out(coords, angles, side=BOTH):
    # Check if knees are pointed out by comparing knee positions
    if side != BOTH:
        return True
    if coords[11] is not None and coords[12] is not None and \
    coords[13] is not None and coords[14] is not None:
        if coords[13][0] < coords[15][0] - 20 or coords[14][0] > coords[16][0] + 20:
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
    KEEP_HIPS_BACK_SQUAT
    KEEP_KNEES_OVER_TOES_SQUAT
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
    WIDTH, HEIGHT = dims
    SCREEN_MARGIN = 20

    bad_form_list = []

    if current_exercise == 'bicep curl':
        # Check if body is in camera frame
        for coord in coords[5:13]:
            if coord[0] < SCREEN_MARGIN or coord[1] < SCREEN_MARGIN or coord[0] > WIDTH-SCREEN_MARGIN or coord[1] > HEIGHT-SCREEN_MARGIN:
                bad_form_list.append("MOVE_INTO_CAMERA_FRAME")
                break
        # Check if too close to camera
        if coords[9][0] < SCREEN_MARGIN and coords[10][0] > WIDTH-SCREEN_MARGIN or \
            coords[10][0] < SCREEN_MARGIN and coords[9][0] > WIDTH-SCREEN_MARGIN or \
            (coords[5][1] < SCREEN_MARGIN or coords[6][1] < SCREEN_MARGIN) and (coords[9][1] > HEIGHT-SCREEN_MARGIN or coords[10][1] > HEIGHT-SCREEN_MARGIN or coords[11][1] > HEIGHT-SCREEN_MARGIN or coords[12][1] > HEIGHT-SCREEN_MARGIN) or \
            coords[9][0] < SCREEN_MARGIN and (coords[5][0] > WIDTH-SCREEN_MARGIN or coords[7][0] > WIDTH-SCREEN_MARGIN) or \
            (coords[6][0] < SCREEN_MARGIN or coords[8][0] < SCREEN_MARGIN) and coords[10][0] > WIDTH-SCREEN_MARGIN:
            bad_form_list.append("MOVE_AWAY_FROM_CAMERA")
        # Check elbows below shoulders
        if not check_elbows_under_shoulders(coords, angles, side):
            bad_form_list.append("KEEP_ELBOWS_UNDER_SHOULDERS")
        # Check shoulders above hips
        if not check_shoulders_above_hips(coords, angles, side):
            bad_form_list.append("KEEP_SHOULDERS_ABOVE_HIPS")
        # Check elbows close to body
        if not check_elbows_close_to_body(coords, angles, side, direction='vertical'):
            bad_form_list.append("KEEP_ELBOWS_CLOSE_TO_BODY")
        # Checks shoulders level
        if not check_shoulders_level(coords, angles):
            bad_form_list.append("KEEP_SHOULDERS_LEVEL")

    elif current_exercise == 'squat':
        # Check if body is in camera frame
        for coord in coords[5:17]:
            if coord[0] < SCREEN_MARGIN or coord[1] < SCREEN_MARGIN or coord[0] > WIDTH-SCREEN_MARGIN or coord[1] > HEIGHT-SCREEN_MARGIN:
                bad_form_list.append("MOVE_INTO_CAMERA_FRAME")
                break
        # Check if too close to camera
        if coords[15][0] < SCREEN_MARGIN and coords[16][0] > WIDTH-SCREEN_MARGIN or \
            coords[16][0] < SCREEN_MARGIN and coords[15][0] > WIDTH-SCREEN_MARGIN or \
            (coords[5][0] < SCREEN_MARGIN or coords[6][0] < SCREEN_MARGIN) and (coords[15][0] > HEIGHT-SCREEN_MARGIN or coords[16][0] > HEIGHT-SCREEN_MARGIN) or \
            (coords[15][0] < SCREEN_MARGIN or coords[16][0] < SCREEN_MARGIN) and (coords[5][0] > HEIGHT-SCREEN_MARGIN or coords[6][0] > HEIGHT-SCREEN_MARGIN) or \
            (coords[5][1] < SCREEN_MARGIN or coords[6][1] < SCREEN_MARGIN) and (coords[15][1] > HEIGHT-SCREEN_MARGIN or coords[16][1] > HEIGHT-SCREEN_MARGIN):
            bad_form_list.append("MOVE_AWAY_FROM_CAMERA")
        # Check hips back behind ankles
        if not check_hips_back_squat(coords, angles, side):
            bad_form_list.append("KEEP_HIPS_BACK_SQUAT")
        # Check knees over toes
        if not check_knees_over_toes_squat(coords, angles, side):
            bad_form_list.append("KEEP_KNEES_OVER_TOES_SQUAT")
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
            if coord[0] < SCREEN_MARGIN or coord[1] < SCREEN_MARGIN or coord[0] > WIDTH-SCREEN_MARGIN or coord[1] > HEIGHT-SCREEN_MARGIN:
                bad_form_list.append("MOVE_INTO_CAMERA_FRAME")
                break
        # Check if not facing camera
        if coords[5][0] <= coords[6][0]+30 or coords[11][0] <= coords[12][0]+30:
            bad_form_list.append("FACE_CAMERA")
        # Check if too close
        if coords[9][0] < SCREEN_MARGIN and coords[10][0] > WIDTH-SCREEN_MARGIN or \
           (coords[9][1] < SCREEN_MARGIN or coords[10][1] < SCREEN_MARGIN) and (coords[5][1] > HEIGHT-SCREEN_MARGIN or coords[6][1] > HEIGHT-SCREEN_MARGIN) or \
            coords[0][1] < SCREEN_MARGIN and (coords[9][1] > HEIGHT-SCREEN_MARGIN or coords[10][1] > HEIGHT-SCREEN_MARGIN):
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
        if not check_arms_level(coords, angles) and side == BOTH:
            bad_form_list.append("KEEP_ARMS_LEVEL")
        
    elif current_exercise == 'lunge':
        # Check if body is in camera frame
        for coord in coords[5:17]:
            if coord[0] < SCREEN_MARGIN or coord[1] < SCREEN_MARGIN or coord[0] > WIDTH-SCREEN_MARGIN or coord[1] > HEIGHT-SCREEN_MARGIN:
                bad_form_list.append("MOVE_INTO_CAMERA_FRAME")
                break
        # Check if too close to camera
        if coords[15][0] < SCREEN_MARGIN and coords[16][0] > WIDTH-SCREEN_MARGIN or \
            coords[16][0] < SCREEN_MARGIN and coords[15][0] > WIDTH-SCREEN_MARGIN or \
            (coords[5][1] < SCREEN_MARGIN or coords[6][1] < SCREEN_MARGIN) and (coords[15][1] > HEIGHT-SCREEN_MARGIN or coords[16][1] > HEIGHT-SCREEN_MARGIN):
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