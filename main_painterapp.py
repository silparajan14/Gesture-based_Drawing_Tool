import mediapipe as mp
import cv2
import numpy as np
import time

# Constants
ml = 150
max_x, max_y = 250+ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0,0

# Color palette
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]

# Initialize color index
color_idx = 0

def next_color():
    global color_idx
    color_idx = (color_idx + 1) % len(colors)
    return colors[color_idx]

def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

def index_raised(yi, y9):
    return (y9 - yi) > 40

hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils


# Modify tools appearance

tools = np.zeros((2 * max_y + 5, 2 * max_x + 100, 3), dtype="uint8")

# Define positions for each tool with increased gap
tool_positions = [
    (8, 10),     # Line
    (190, 10),    # Rectangle (Increased x-coordinate)
    (430, 10),    # Draw (Increased x-coordinate)
    (600, 10),    # Circle (Increased x-coordinate)
    (750, 10)     # Erase (Increased x-coordinate)
]


# Define tool names
tool_names = ["Line", "Rectangle", "Draw", "Circle", "Erase"]

# Draw each tool at its respective position
tool_size = 120  # Increase the tool size as needed

for i, pos in enumerate(tool_positions):
    # Adjust the size of the individual tool boxes
    if i == 1:  # Index 1 corresponds to "Rectangle"
        cv2.rectangle(tools, (pos[0], pos[1]), (pos[0] + tool_size + 100, pos[1] + tool_size), (0, 0, 255), 2)
    else:
        cv2.rectangle(tools, (pos[0], pos[1]), (pos[0] + tool_size, pos[1] + tool_size), (0, 0, 255), 2)

    # Calculate the position to center the tool name within the box
    text_x = pos[0] + (tool_size - len(tool_names[i]) * 10) // 2
    text_y = pos[1] + (tool_size + 20) // 2 # Adjusted for better alignment

    cv2.putText(tools, tool_names[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


# Add tools labels with increased font size and positions (changed the font to black)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(tools, "Line", (50, 70), font, 1.1, (0,0,0), 2)
cv2.putText(tools, "Rectangle", (210, 70), font, 1.1, (0,0,0), 2)
cv2.putText(tools, "Draw", (450, 70), font, 1.1, (0,0,0), 2)
cv2.putText(tools, "Circle", (620, 70), font, 1.1, (0,0,0), 2)
cv2.putText(tools, "Erase", (770, 70), font, 1.1, (0,0,0), 2)



# Initialize mask
mask = np.ones((480, 640))*255
mask = mask.astype('uint8')

cap = cv2.VideoCapture(0)

while True:
	_, frm = cap.read()
	frm = cv2.flip(frm, 1)

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = hand_landmark.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
			x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)

			if x < max_x and y < max_y and x > ml:
				if time_init:
					ctime = time.time()
					time_init = False
				ptime = time.time()

				cv2.circle(frm, (x, y), rad, (0,255,255), 2)
				rad -= 1

				if (ptime - ctime) > 0.8:
					curr_tool = getTool(x)
					print("your current tool set to : ", curr_tool)
					time_init = True
					rad = 40

			else:
				time_init = True
				rad = 40

			if curr_tool == "draw":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
					prevx, prevy = x, y

				else:
					prevx = x
					prevy = y



			elif curr_tool == "line":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.line(frm, (xii, yii), (x, y), (50,152,255), thick)

				else:
					if var_inits:
						cv2.line(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "rectangle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)

				else:
					if var_inits:
						cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "circle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

				else:
					if var_inits:
						cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
						var_inits = False

			elif curr_tool == "erase":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.circle(frm, (x, y), 30, (0,0,0), -1)
					cv2.circle(mask, (x, y), 30, 255, -1)



	op = cv2.bitwise_and(frm, frm, mask=mask)
	frm[:, :, 1] = op[:, :, 1]
	frm[:, :, 2] = op[:, :, 2]
	tools_resized = cv2.resize(tools,(max_x-ml,max_y))


	frm[max_y:max_y+max_y, ml:max_x] = tools_resized

	cv2.putText(frm, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("paint app", frm)
	

	if cv2.waitKey(1) & 0XFF==ord('q'):
		cv2.destroyAllWindows()
		cap.release()
		break
 