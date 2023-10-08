# TrackmanAI
Framework for capturing and training an AI to play Trackmania 2020

## record.py
Record is a simple script to capture the current frame from Trackmania as well as the vehicle state and save it into
paired files for later training. The script is designed to be run in the background while playing Trackmania. It will
capture the current frame and vehicle state at 20Hz and save to a bitmap and json file respectively.

## compile.py
Compile is a script to take the recorded data, convert image files to png for less intensive storage requirements,
calculate the reward for each frame, and compile this data into a final single script containing the info for every
frame in the run.  
The reward is based off of the distance traveled over the next several seconds.