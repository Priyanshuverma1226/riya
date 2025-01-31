from datetime import datetime
from playsound import playsound

# Get alarm time from user
alarm_time = input("Enter the time of alarm to be set (HH:MM:SS AM/PM):\n")

# Extract hour, minute, second, and period (AM/PM) from the input
alarm_hour = alarm_time[0:2]
alarm_minute = alarm_time[3:5]
alarm_seconds = alarm_time[6:8]
alarm_period = alarm_time[9:].upper()

print("Setting up alarm..")

while True:
    now = datetime.now()
    
    # Format current time
    current_hour = now.strftime("%I")
    current_minute = now.strftime("%M")
    current_seconds = now.strftime("%S")
    current_period = now.strftime("%p")

    # Check if it's time to sound the alarm
    if alarm_period == current_period:
        if alarm_hour == current_hour:
            if alarm_minute == current_minute:
                if alarm_seconds == current_seconds or alarm_seconds == "SS":
                    print("Wake Up!")
                    playsound('audio.mp3')
                    break
