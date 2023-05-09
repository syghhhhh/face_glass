import os
import subprocess
# command = "ffmpeg -ss 20 -to 80 -i test.mp4 -c:v copy outputVideo.mp4 -loglevel quiet"
# print('command -> ' + command)
# os.system(command)

# subprocess.run(['ffmpeg', '-ss', '20', '-to', '80', '-i', 'test.mp4', '-c:v', 'copy', 'outputVideo1.mp4', '-loglevel', 'quiet'])

result = subprocess.getoutput('ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 test.mp4 -loglevel quiet')
print(result)
