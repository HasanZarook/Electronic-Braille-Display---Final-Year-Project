# ======================= Imported Libraries =======================
import cv2
import re
import time                                                                
import os
import datetime
from picamera2 import Picamera2
import pytesseract
import RPi.GPIO as GPIO
import PyPDF2
import pygame
import string
# ======================= Global Variables =======================
braille_map = {
'CAP' : [1,4],
'DEC' : [1,5],
'NUM' : [4,0],
'0': [3, 7],
'1': [2, 1],
'2': [7, 1],
'3': [2, 2],
'4': [2, 7],
'5': [2, 3],
'6': [7, 2],
'7': [7, 7],
'8': [7, 3],
'9': [3, 2],
' ': [1, 1],
'.': [3, 6],
',': [3, 1],
'?': [6, 4],
'!': [6, 3],
':': [3, 3],
';': [6, 1],
'-': [4, 4],
'a': [2, 1],
'b': [7, 1],
'c': [2, 2],
'd': [2, 7],
'e': [2, 3],
'f': [7, 2],
'g': [7, 7],
'h': [7, 3],
'i': [3, 2],
'j': [3, 7],
'k': [5, 1],
'l': [0, 1],
'm': [5, 2],
'n': [5, 7],
'o': [5, 3],
'p': [0, 2],
'q': [0, 7],
'r': [0, 3],
's': [6, 2],
't': [6, 7],
'u': [5, 4],
'v': [0, 4],
'w': [3, 0],
'x': [5, 5],
'y': [5, 0],
'z': [5, 6]
}

folder = "captured_images"
picam2 = None
# Navigation data
PDF_Files = None
No_of_pdfs = 0
current_Pdf = 0
current_Page_No = 0
current_page_text = None
char_No = 0
No_Pages_Pdf = 0
# Motor Values
current_pos_Motor1 = 0
current_pos_Motor2 = 0
Motor_Step = 256
# 0 for Picture mode 1 for PDF mode 
Mode_Toggle = 1
# ======================= Button Configuration =======================
Mode_Btn = 40
Img_Capture_Btn = 38
Next_Character = 36
Previous_Character = 37
Next_Page = 32
Prev_Page = 22
Next_PDF = 18
Prev_PDF = 16

def Configure_Button():
    
    GPIO.setup(Mode_Btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(Img_Capture_Btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(Next_Character, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(Previous_Character, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(Next_Page, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(Prev_Page, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(Next_PDF, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(Prev_PDF, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
# ======================= Sounds =======================
pygame.mixer.init()
start_Up = 'Sounds/Start_Up.wav'
Capture = 'Sounds/Capture_camera.wav'
Error = 'Sounds/error.wav'
Next_character_S = 'Sounds/Next_character.wav'
Next_page_S = 'Sounds/Next_page.wav'
Next_PDF_S = 'Sounds/Next_PDF.wav'
Pdf_Mode = 'Sounds/Pdf_Mode.wav'
Picture_Mode = 'Sounds/Picture_Mode.wav'
Prev_Character_S = 'Sounds/Prev_Character.wav'
Prev_Page_S = 'Sounds/Prev_Page.wav'
Prev_PDF_S = 'Sounds/Prev_PDF.wav'
def Play_Sound(path):
    sound = pygame.mixer.Sound(path)
    sound.play()
    
# ======================= Camera Configuration =======================
def setup_camera():
    global picam2
    if picam2 is None:
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (1920, 1080)
        picam2.preview_configuration.sensor.output_size = (1920, 1080)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.preview_configuration.align()
        picam2.configure("video")
#         picam2.start()
    return picam2

# ======================= Image Processing Functions =======================
def get_GrayScale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def remove_Noise(img):
    return cv2.medianBlur(img, 3)

def thresholding(img, limit):
    return cv2.threshold(img, limit, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)[1]

def thresholding_adaptive(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    sharp = cv2.addWeighted(image, 3.5, gaussian, -2.5, -10)
    return sharp
# ======================= Read Text From PDF =======================

def Get_USB_Name(path):
    if os.path.exists(path):
        files = os.listdir(path)
        print(len(files))
        if files:
            return os.path.join(path, files[0]) 
        else:
            return "No files found in the directory."
    else:
        return "Directory does not exist."
do_Once = True
def Get_USB_filePath(usb_path):
    global PDF_Files
    path = Get_USB_Name(usb_path)
    print(path)
    global No_of_pdfs
    global do_Once
    if os.path.exists(path):
        pdf_files = [f for f in os.listdir(path) if f.lower().endswith(".pdf")]
        if pdf_files:
            print("PDF files found on USB:")
            for pdf in pdf_files:
                print(pdf)
                if do_Once:
                    No_of_pdfs +=1
            PDF_Files = pdf_files
            do_Once = False
            return path
        else:
            print("No PDF files found on USB.")
            return None
    else:
        print("USB directory not found. Please check if it's mounted.")
        return None
        
def extract_text_from_pdf(file_path, page_num=0):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        if page_num < len(pdf_reader.pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            return text
        else:
            return ""
def get_pdf_page_count(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return len(pdf_reader.pages)
# ======================= Motor Setup and Control =======================
motor1_pins = [7,11, 13, 15]
motor2_pins = [29, 31, 33, 35]

step_sequence = [
    [1,0,0,1],
    [1,0,0,0], 
    [1,1,0,0],
    [0,1,0,0],
    [0,1,1,0],
    [0,0,1,0],
    [0,0,1,1],
    [0,0,0,1]
]

def setup_motors():
    for pin in motor1_pins + motor2_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)
control = 0
def rotate_Motor(pins,steps,direction=1,delay=0.001):
    print(pins)
    print(steps)
    print(direction)
    for _ in range(steps):
        steps_seq = step_sequence if direction == 1 else step_sequence[::-1]
        for step in steps_seq:
            for pin, val in zip(pins, step):
                GPIO.output(pin, val)
            time.sleep(delay)
            
def rotate_Motor2(pins, steps, direction=1, delay=0.002):
    steps_seq = step_sequence if direction == 1 else step_sequence[::-1]
    seq_len = len(steps_seq)
    step_count = 0
    current_step = 0

    while step_count < steps:
        for pin, val in zip(pins, steps_seq[current_step]):
            GPIO.output(pin, val)
        current_step = (current_step + 1) % seq_len
        step_count += 1
        time.sleep(delay)

def stop_Motor(pins):
    for pin in pins:
        GPIO.output(pin, 0)

# ======================= Auixilary Functions =======================
def Goto_MeanPosition():
    global current_pos_Motor1
    global current_pos_Motor2
    global motor1_pins
    global motor2_pins
    global Motor_Step
    motor1_dir = get_direction(current_pos_Motor1,0)
    print('Motor Dir')
    print(-motor1_dir)
    motor_2_dir = get_direction(current_pos_Motor2,0)
    print(motor_2_dir)
    motor_1_steps = get_steps(current_pos_Motor1,0)
    print('Motor Steps')
    print(motor_1_steps)
    motor_2_steps = get_steps(current_pos_Motor2,0)
    print(motor_2_steps)
    if current_pos_Motor1 ==0:
        print('Same_Position')
    else:
        rotate_Motor2(motor1_pins,motor_1_steps, direction= -motor1_dir)
        stop_Motor(motor1_pins)
        current_pos_Motor1 = 0
    if current_pos_Motor2 == 0:
        print('Same_Position')
    else:
        rotate_Motor2(motor2_pins,motor_2_steps, direction= motor_2_dir)
        stop_Motor(motor2_pins)
        current_pos_Motor2 = 0
def Startup_Setup():
    global folder
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    setup_motors()
    Configure_Button()
    setup_camera()
    Add_Button_Events()
    os.makedirs(folder, exist_ok=True)
displayed =0
def Display_Text(text_to_display = current_page_text):
#     current_page_text ='Osama'
    global braille_map
    global current_pos_Motor1
    global current_pos_Motor2
    global motor1_pins
    global motor2_pins
    global Motor_Step
    global char_No
    global displayed
#     Goto Next Page if no text
    if char_No >= len (text_to_display):
        char_No = 0
    display_char = text_to_display[char_No]
    if display_char.isupper()and displayed == 0:
        displayed = 1
        char_No-=1
        motors_positions = braille_map.get('CAP')
        print('capital')
    elif display_char.isdigit() and displayed == 0:
        displayed = 1
        char_No-=1
        motors_positions = braille_map.get('NUM')
        print("digit")
    elif display_char in string.punctuation and displayed == 0:
        displayed = 1
        char_No-=1
        motors_positions = braille_map.get('DEC')
        print('punk')
    else:
        displayed = 0
        motors_positions = braille_map.get(display_char.lower())
    if motors_positions is None:
        char_No+=1
        Display_Text(text_to_display)
    print(motors_positions)
    print(display_char)
    print("---------")
    motor1_dir = get_direction(current_pos_Motor1,motors_positions[0])
#     print('Motor Dir')
#     print(-motor1_dir)
    motor_2_dir = get_direction(current_pos_Motor2,motors_positions[1])
#     print(motor_2_dir)
    motor_1_steps = get_steps(current_pos_Motor1,motors_positions[0])
#     print('Motor Steps')
#     print(motor_1_steps)
    motor_2_steps = get_steps(current_pos_Motor2,motors_positions[1])
#     print(motor_2_steps)
    if current_pos_Motor1 == motors_positions[0]:
        print('Same_Position')
    else:
        rotate_Motor2(motor1_pins,motor_1_steps, direction= -motor1_dir)
        stop_Motor(motor1_pins)
        current_pos_Motor1 = motors_positions[0]
    if current_pos_Motor2 == motors_positions[1]:
        print('Same_Position')
    else:
        rotate_Motor2(motor2_pins,motor_2_steps, direction= motor_2_dir)
        stop_Motor(motor2_pins)
        current_pos_Motor2 = motors_positions[1]
    
def get_steps(current,target):
    forward_steps = (target - current) % 8
    backward_steps = (current - target) % 8
    if forward_steps <= backward_steps:
        return forward_steps * Motor_Step
    else:
        return backward_steps * Motor_Step
    
def get_direction(current, target):
    forward_steps = (target - current) % 8
    backward_steps = (current - target) % 8

    if forward_steps <= backward_steps:
        return 1
    else:
        return -1

def Image_Processing_Pipeline(frame):
    gray = get_GrayScale(frame)
    sharp = sharpen_image(gray)
    cv2.imwrite("captured_images/GrayScale.jpg", gray)
    cv2.imwrite("captured_images/Sharpened.jpg", sharp)
    return sharp

def clean_string(input_string):
    cleaned_string = input_string.lstrip()
    cleaned_string = re.sub(r' +', '  ', cleaned_string)
    return cleaned_string

# ======================= Event Functions =======================   
def toggle_mode_Event(channel):
    global char_No
    char_No  = 0
    Goto_MeanPosition()
    global Mode_Toggle
    global Pdf_Mode
    global Picture_Mode
    Mode_Toggle = 1 - Mode_Toggle
    print(Mode_Toggle)
    if (Mode_Toggle == 1):
        Play_Sound(Pdf_Mode)
        PDF_Mode()
    else:
        Play_Sound(Picture_Mode)
        

def Capture_image_Event(channel):
#     rotate_Motor2(motor1_pins,256, direction=-1)   # clockwise
#     stop_Motor(motor1_pins)
    global picam2
    global current_page_text
    if  Mode_Toggle == 0:
        picam2.start()
        print('Capture')
        time.sleep(1)
        global Capture
        Play_Sound(Capture)
        frame = picam2.capture_array()
        picam2.stop()
        #cv2.imshow("Camera Feed", frame)
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = os.path.join(folder, f"image_{timestamp}.jpg")
        processed_Image = Image_Processing_Pipeline(frame)
        current_page_text = pytesseract.image_to_string(processed_Image)
        char_No = 0
        print("Detected Text:")
        print(current_page_text)
        print(f"Text Length: {len(current_page_text)}")
        Display_Text(current_page_text)
        

def Next_Character_Event(channel):
#     rotate_Motor2(motor1_pins,512, direction=-1)   # clockwise
#     stop_Motor(motor1_pins)
    global char_No
    global current_Page_No
    global Next_character_S
    Play_Sound(Next_character_S)
    char_No += 1
    if char_No < len(current_page_text):
        Display_Text(current_page_text)
    else:
        char_No = 0
        Display_Text(current_page_text)
  #         NextPage
    print("NCE")
def Previous_Character_Event(channel):
    
#     rotate_Motor2(motor2_pins,256, direction=1)   # clockwise
#     stop_Motor(motor2_pins)

    global char_No
    global current_Page_No
    global Prev_Character_S
    char_No -= 1
    Play_Sound(Prev_Character_S)
    if char_No >= 0:
        Display_Text(current_page_text)
    else:
        char_No = 0
        print('start of the page')
    print("PCE")
    
def Next_Page_Event(channel):
    
    if Mode_Toggle == 0:
        return
    global Next_page_S
    Play_Sound(Next_page_S)
    global current_Page_No
    global No_Pages_Pdf
    current_Page_No +=1
    print(current_Page_No)
    print(No_Pages_Pdf)
    if(current_Page_No >= No_Pages_Pdf):
        Next_PDF_Event(32)
    else:
        PDF_Mode()
        print('PDF CAleed')
    print("NPE")
     
def Prev_Page_Event(channel):
    
#     rotate_Motor(motor1_pins,200,1)
#     time.sleep(2)
#     rotate_Motor(motor1_pins,200,-1)
    if Mode_Toggle == 0:
        return
    global Prev_Page_S
    Play_Sound(Prev_Page_S)
    global current_Page_No
    global No_Pages_Pdf
    current_Page_No -=1
#     print(current_Page_No)
#     print(No_Pages_Pdf)
    if(current_Page_No < 0 ):
        print('Stay at the same page because its page 1')
        return
    else:
        PDF_Mode()
        print('PDF CAleed')
    print("NPE")
     

    
def Next_PDF_Event(channel):
    global current_Pdf
    global No_of_pdfs
    global Next_PDF_S
    print(current_Pdf)
    print(No_of_pdfs)
    if Mode_Toggle == 0:
        rotate_Motor2(motor1_pins,50, direction=1)   # clockwise
        stop_Motor(motor1_pins)
        return
    Play_Sound(Next_PDF_S)
    current_Pdf +=1
    if current_Pdf > No_of_pdfs:
        current_Pdf = 0
        PDF_Mode()
    else:
        print('-----------------')
        print(current_Pdf)
        print(No_of_pdfs)
        print('-----------------')
        PDF_Mode()
    print("NextPDF")
    
def Prev_PDF_Event(channel):
    global current_Pdf
    global No_of_pdfs
    global Prev_PDF_S
    print(current_Pdf)
    print(No_of_pdfs)
    if Mode_Toggle == 0:
        rotate_Motor2(motor2_pins,50, direction=-1)   # clockwise
        stop_Motor(motor2_pins)
        return
    Play_Sound(Prev_PDF_S)
    current_Pdf -=1
    if current_Pdf < 0:
        current_Pdf = No_of_pdfs-1 
        PDF_Mode()
    else:
        PDF_Mode()
    print("PrevPDF")  

def Add_Button_Events():
    GPIO.add_event_detect(Mode_Btn, GPIO.RISING, callback=toggle_mode_Event, bouncetime=200)
    GPIO.add_event_detect(Img_Capture_Btn, GPIO.RISING, callback= Capture_image_Event, bouncetime=200)
    GPIO.add_event_detect(Next_Character, GPIO.RISING, callback= Next_Character_Event, bouncetime=200)
    GPIO.add_event_detect(Previous_Character, GPIO.RISING, callback= Previous_Character_Event, bouncetime=200)
    GPIO.add_event_detect(Next_Page, GPIO.RISING, callback= Next_Page_Event, bouncetime=200)
    GPIO.add_event_detect(Prev_Page, GPIO.RISING, callback= Prev_Page_Event, bouncetime=200)
    GPIO.add_event_detect(Next_PDF, GPIO.RISING, callback= Next_PDF_Event, bouncetime=200)
    GPIO.add_event_detect(Prev_PDF, GPIO.RISING, callback= Prev_PDF_Event, bouncetime=200)

def Picture_Mode():
    return

def PDF_Mode():
    global current_page_text
    global No_Pages_Pdf
    print('PDF')
    path = Get_USB_filePath("/media/pi")
    if path is None:
        return
    if os.path.exists(path):
        path = os.path.join(path,PDF_Files[current_Pdf])
        current_page_text = extract_text_from_pdf(path , current_Page_No)
        current_page_text = clean_string(current_page_text)
        No_Pages_Pdf = get_pdf_page_count(path)
        print(No_Pages_Pdf)
        print(current_page_text)
        Display_Text(current_page_text)
    

key ='d'
# ======================= Main OCR and Motor Control Routine =======================  
def main():
    global start_Up
    Startup_Setup()
    global No_of_pdfs
    Play_Sound(start_Up)
    PDF_Mode()
# #     print('---------')
# #     No_of_pdfs = len(PDF_Files)
    print(No_of_pdfs)
    print('System Setup')
    global picam2
    
# # # #     rotate_Motor(motor2_pins,200, direction=1)   # clockwise
# # # #     stop_Motor(motor2_pins)
#     try:
#         while True:
# #             if GPIO.input(Mode_Btn) == GPIO.HIGH:
# #                 if (Mode_Toggle == 1):
# #                     Mode_Toggle =0
# #                 else:
# #                     Mode_Toggle = 1
# #                 time.sleep(0.05)
# #             if (Mode_Toggle == 0):
# #                 frame = picam2.capture_array()
# #                 print('work')
# #                 cv2.imshow("Camera Feed", frame)
# #                 time.sleep(0.05)
# #             
#             if key == ord('m'):
#                 print("Rotating motors...")
#                 
#                 print("Motors moved.")
# 
#             elif key == ord('q'):
#                 print("Exiting program...")
#                 break
# 
#     except KeyboardInterrupt:
#         print("Program interrupted.")
# 
#     finally:
#         pygame.mixer.quit()
#         picam2.stop()
#         GPIO.cleanup()
#         cv2.destroyAllWindows()
#         print("Camera stopped, motors cleaned up, and windows closed.")

# ======================= Program Entry Point =======================
if __name__ == "__main__":
    main()