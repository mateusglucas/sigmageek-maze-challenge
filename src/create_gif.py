#!/usr/bin/env python3

# Generate animated .gif
#
# Before running this script, the script solver.py should be executed 
# with plot enabled (plot_flag = True), to generate the necessary images
# to create the .gif

import imageio
import os

n_challenge =  1    # desired challenge number (1-4)
frame_period = 0.1  # frame timestep
                    # sugestion: 0.1 for challenges 1 2 e 3; 0.2 for 
                    # challenge 4

frame_cnt=0

with imageio.get_writer('../images/challenge'+str(n_challenge)+'.gif', mode='I', duration = frame_period) as writer:
        while os.path.isfile('../images/challenge'+str(n_challenge)+'_img_'+str(frame_cnt)+'.png'):
            filename = '../images/challenge'+str(n_challenge)+'_img_'+str(frame_cnt)+'.png'
            image = imageio.imread(filename)
            writer.append_data(image)

            print(frame_cnt)
            frame_cnt+=1
