import numpy as np
import cv2
#import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.ticker as ticker

#global variables
column = 1280     # this is the farthest right pixel in our window
column_scan = []  # list used as a global variable to store the column location during scan
film = []
BLACK = 0
WHITE = 255

# If i make changes where does it go to
# function to scan the left side of the frame for the fiber and measure the number of pixels width
def FiberWidthTest(frame, scalebar):

    start = [] # initializing empty lists
    end = []
    width = []
    fiber_in_microns = 125  # the actual width of the fiber is known to be 125 microns
    fiber_width = 0     # number of pixels

    for j in range (1,6):                           #scans 5 columns, this can be increased to improve accuracy
        for i in range(1,1024):
            px = frame[i,j]
            if i != 1023:                           # if the for loop is not on the last pixel, save the upcoming pixel
                pxplusone = frame[i+1, j]
                #FOR TESTING print(px, end='')      # print the column where 0 is black, 255 is white
            if px == 0:                             # if we are on a black pixel
                if pxplusone == 255:                # and the next one is white
                    start.append(i + 1)             # add the pixel location to the 'start' list
            if px == 255:                           # if the for loop is on a white pixel
                if pxplusone == 0:                  # and the next is black
                    end.append(i)                   # add the value of the white pixel to 'end' list
        width.append(end[len(end)-1]-start[0])      # adds the width a column of pixels measured to a list
        #print ('Fiber Width: ', width)              # FOR TESTING to see the different values measured
        start.clear()                               #clears list to measure the next column
        end.clear()

    width_sum = 0                               # sum of the width measurements
    for x in range (len(width)):                # runs through the length of the width list, currently should be 5 items
        width_sum = width_sum + width[x]

    fiber_width = width_sum / len(width)        # averages the column measurements
    print('FIBER WIDTH: ', fiber_width)         # FOR TESTING
    scalebar = fiber_in_microns / fiber_width   # divides 125 by the pixel average to give a micron/pixel scale
    return scalebar

# function for determing the scanning region on the widest part of the device
def ScanningRegion(frame):
    start = []                                  # initialize list

    # we scan from left to right across each row, then move downwards looking for the column that has the first white pixel
    for i in range(1,600):                     # 1024 rows of pixels
        for j in range(400,1000):                 # 1280 columns of pixels
            px = frame[i,j]                     # pixels are [row, column] so basically [y, x], yes it's inverted
            if j != 999:                       # if we are not on the last column
                pxplusone = frame[i,j+1]        # look ahead at the upcoming pixel in the scan
            if px == WHITE and pxplusone == BLACK:    # if current pixel in scan is black and upcoming is white
                start.append(j+1)               # save the white pixel column location
                break
            elif j == 999:                      # if no white pixel is detected in the row
                start.append(0)                 # submit 0 into the list
                break

    global column                               # we use a global variable 'column' and this list for 'row' which will be returned from the function
    global column_scan
    row = []                                    # these variables will be the coordinate of what we measure against

    for i in range(0, len(start)):              # cycle through the list (array) that contains coordinates for when it changes from black to white
        if start[i] != 0 and start[i] < column: # excludes zeros, finds the minimum index and its value compared to 1280
            column = start[i]                   # column becomes the smaller value until the smallest is found (leftmost edge)


    tolerance = 3                               # find each row that is within 3 pixels of the widest point
    for i in range(0, len(start)):              # cycle through the list (array) that contains coordinates for when it changes from black to white
        if (column - tolerance) < start[i] < (column + tolerance): # column plus or minus 3
            row.append(i + 1)                   # since indexes always start at 0, the actual row needs to be the index value plus one
            column_scan.append(start[i])        # save the column value that correpsonds to each row
    #print('COLUMNS: ', column_scan)             # FOR TESTING
    return row


# must put the path to where you have the videos saved here
camera = cv2.VideoCapture("/Users/matthewsimmons/Desktop/College/Research/ImageProcessing/test_trimmed.mov")                     # start a connection to the file
endofvideo = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))  # the frame count is found using this function
print('NUMBER OF FRAMES: ', endofvideo)
count = 0            # frames in a video
scalebar = 0         # ratio of microns/pixels
scan_range = []      # the rows that meet tolerance level near wide part of device
thresholdvalue = 170 # optimum value for binary thresholding
jumpback = 120
viewimage = 1

# this block of code is for finding the width of the fiber
camera.set(cv2.CAP_PROP_POS_FRAMES, endofvideo-2)                # go to one of the last frames in the video when the fiber is in the frame
ret, frame = camera.read()                                       # get the fram
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                   # turn it gray
ret, thresh = cv2.threshold(gray, thresholdvalue, 255, cv2.THRESH_BINARY) # threshold it
if viewimage == 1:
    cv2.imshow('Fiber', thresh)
scalebar = FiberWidthTest(thresh, scalebar)
print('SCALEBAR: ', scalebar)
thresholdvalue = 158
cv2.waitKey(0)

# this block of code is for finding the widest points on the device according to our tolerance
camera.set(cv2.CAP_PROP_POS_FRAMES, endofvideo-jumpback)
ret, frame = camera.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                   # turn it gray
ret, thresh = cv2.threshold(gray, thresholdvalue, 255, cv2.THRESH_BINARY)
if viewimage == 1:
    cv2.imshow('Dry Device Binary', thresh)                                                # draw the result
scan_range = ScanningRegion(thresh)                                             # calls a function to find the scanning range
if viewimage == 1:
    for x in range(len(column_scan)):                                               # for the length of the scanning list plot each coordinate with a blue pixel
        cv2.line(frame, (column_scan[x],scan_range[x]), (column_scan[x],scan_range[x]), (178, 34, 34), 1)
        cv2.imshow('Dry Device Edited', frame)                                          # print a color frame to show the pixels being measured
    print('COLUMNS: ', column_scan, 'SCANNING RANGE ROWS: ', scan_range)            # FOR TESTING prints our found scanning range
thresholdvalue = 158                                                           # optimal thresh value for the wet device
cv2.waitKey(0)


neck_evaporation = 630
neck_column = 0
neck_scan_begin = 560
neck_scan_end = 800


while True:
    var = input("Enter an intger or 'q' to continue: ")
    try:                                # the try-except throws an error if the expected integer is anything else but an integer
        var_int = int(var)
    except ValueError:
        var_int = 0
    if var == "q":                      # this is looking for a lowercase q to break out of the cycle of inputs
        break
    elif var_int >= 0 and var_int <= 1024:  # as long as the inputed number is within the range designated by the screen resolution
        cv2.line(thresh, (neck_scan_begin,var_int), (neck_scan_end,var_int), (178, 34, 34), 1) # then we draw a line arbitrary in length at the designated row
    else:
        print("That is out of range or not an integer")     # if something outside the range or not an integer is inputed, this error message is displayed
        continue
    cv2.imshow('Neck', thresh)       # as we draw lines we will print the updated image to see the result
    cv2.waitKey(10)                        # waits 10 ms before going to the next input. This is mandatory so the image is displayed properly

ret, frame = camera.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                   # turn it gray
ret, thresh = cv2.threshold(gray, thresholdvalue, 255, cv2.THRESH_BINARY)

for i in range(neck_scan_begin, neck_scan_end):        # scan 50 pixels in front of the dry device
    px = thresh[neck_evaporation, i]                       # scanning location
    pxplusone = thresh[neck_evaporation, i + 1]            # look one pixel ahead of scanning location
    if px == WHITE and pxplusone == BLACK:                    # if the scanning location is black and the next pixel is white
        neck_column = i + 1   # save the difference between the ERROR HERE
        break
cv2.line(thresh, (neck_scan_begin,neck_evaporation), (neck_scan_end,neck_evaporation), (178, 34, 34), 1)
cv2.imshow('Line', thresh)
print('neck_column: ', neck_column)


camera.set(cv2.CAP_PROP_POS_FRAMES, 0) # set the video back to the initial frame for comparison


raw_data = [[] for i in range(len(scan_range))]
neck_evap_meas = []

while True:
    count = count + 1                                                # frame counter
    ret, frame = camera.read()                                       # read a frame
    cv2.imshow('Original', frame)                                    # draw the result
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                   # turn it gray
    ret, thresh = cv2.threshold(gray, thresholdvalue, 255, cv2.THRESH_BINARY)   # binary thresholding 127 and 255 are the standard values used

    cv2.imshow('Binary Threshold', thresh)                           # draw the result - this is the video playing

    scan_tolerance = 50     # 50 pixel tolerance
    microns = 0             # depth of film in microns
    measurements = []       # list holds the measurements of the scan
    measure_sum = 0         # sum of the said list
    measure_avg = 0         # avg value of depth in pixels

    for i in range(len(scan_range)):                            # scan each of the rows that were within 3 pixels of the widest point
        for j in range(column - scan_tolerance, column+10):        # scan 50 pixels in front of the dry device
            px = thresh[scan_range[i], j]                       # scanning location
            pxplusone = thresh[scan_range[i], j + 1]            # look one pixel ahead of scanning location
            if px == WHITE and pxplusone == BLACK:                    # if the scanning location is black and the next pixel is white
                measurements.append(column_scan[i] - (j + 1))   # save the difference between the ERROR HERE
                raw_data[i].append((column_scan[i] - (j + 1))*scalebar)
                break
    for j in range(neck_scan_begin, neck_scan_end):        # scan 50 pixels in front of the dry device
        px = thresh[neck_evaporation, j]                       # scanning location
        pxplusone = thresh[neck_evaporation, j + 1]            # look one pixel ahead of scanning location
        if px == WHITE and pxplusone == BLACK:                    # if the scanning location is black and the next pixel is white
            neck_evap_meas.append((neck_column - (j + 1))*scalebar)   # save the difference between the ERROR HERE
            break

    for x in range(len(measurements)):                          # for all of the measurements
        measure_sum = measure_sum + measurements[x]             # sum each in the list
    if len(measurements) != 0:                                  # if the measurement list actually has values
        measure_avg = measure_sum / len(measurements)           # take the average by dividing the sum by the number of measurements
        microns = measure_avg * scalebar                        # convert the pixel depth of film to microns using the scalebar found previously
        film.append(microns)                                    # add each frame's measurement to a list called film
        #print('film width in pixels: ', measure_avg, 'film width in microns: ', microns)
                                                                    #FOR TESTING print('MEASUREMENTS: ', measurements
    # if count == endofvideo-600:
    #     thresholdvalue = 135

    if count == endofvideo-jumpback:
        cv2.destroyAllWindows()
        print('END OF THE VIDEO')
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):                             # wait for 1 ms or until q is pressed
        cv2.destroyAllWindows()
        print('END OF THE VIDEO')
        break

camera.release()                                                     # release the camera

average_over = 9
averaged_microns = []   # initializing lists within a list

counter = 0                                                 # int for counting the number of frames
summation = 0                                               # int that takes the temporary sums
for x in range(int(len(film)/average_over)*average_over):        # if there are 15789 frames analyzed, and the average is 10, this rounds it down. e.g. 15780
    counter = counter + 1                                           # a counter
    summation = film[x] + summation                                 # a summation
    if counter == average_over:                                     # when the counter reaches the averaging amount
        counter = 0                                                 # reset the counter
        averaged_microns.append(summation/average_over)             # store the averaged value
        summation = 0                                               # reset the sum for the next batch of x measurements



fps = 10                                # 10 frames per second based on the THOR documentation
seconds = 60

time1 = np.arange(len(averaged_microns))
time2 = np.arange(len(raw_data[0]))
scale_time = (len(averaged_microns)*average_over)/(seconds*fps)

fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(time1, averaged_microns, '.')
for i in range(len(scan_range)):
    ax2.plot(time2, raw_data[i], '.')


#Just some print statements to make sure we're converting the frames to minutes correctly
print('# Frames analyzed: ', len(film))
print('Length of average_microns:', len(averaged_microns))
print('Length of time:', len(time1))
print('Time in Minutes:', scale_time)
print('Length of Time2 ', len(time2))

ticks_x1 = ticker.FuncFormatter(lambda time1, pos: '{0:g}'.format(time1*average_over/600)) #adjust the x-axis to be the correct time in minutes
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes")
ax1.set_ylabel("Film Thickness (um)")
ax1.set_title("Averaged Stability of Film Thickness Over Time")

ticks_x2 = ticker.FuncFormatter(lambda time2, pos: '{0:g}'.format(time2/600)) #adjust the x-axis to be the correct time in minutes
ax2.xaxis.set_major_formatter(ticks_x2)
ax2.set_xlabel("Time in Minutes")
ax2.set_ylabel("Film Thickness (um)")
ax2.set_title("Raw Data Stability of Film Thickness Over Time")

plt.subplots_adjust(hspace=.7)

time3 = np.arange(len(neck_evap_meas))

fig2 = plt.figure(2)
ax3 = fig2.add_subplot(111)
ax3.plot(time3, neck_evap_meas, '.')

ticks_x3 = ticker.FuncFormatter(lambda time3, pos: '{0:g}'.format(time3/600)) #adjust the x-axis to be the correct time in minutes
ax3.xaxis.set_major_formatter(ticks_x3)
ax3.set_xlabel("Time in Minutes")
ax3.set_ylabel("Film Thickness at Neck (um)")
ax3.set_title("Film Thickness at the Neck of the Device")

plt.show()

cv2.waitKey(0)
