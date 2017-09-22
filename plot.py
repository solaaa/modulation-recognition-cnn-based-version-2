import similarity
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Polygon
simi = np.load('similarity.npy')
Y_test = np.load('test_label.npy')
Z_test = np.load('test_snr.npy')
Y_test = Y_test[0:5000]
Z_test = Z_test[0:5000]

'''
g1: -18
g2: -12
g3: -6
g4: 0(dB)
'''
g1 = []
g2 = []
g3 = []
g4 = []     

print(Z_test[2])


for i in range(len(Y_test)):
    if Z_test[i] == -18:
        g1.append(i)
    elif Z_test[i] == -12:
        g2.append(i)
    elif Z_test[i] == -6:
        g3.append(i)
    else:
        g4.append(i)
xlabel = ['-18dB', '-12dB', '-6dB', '0dB']

data = [simi[g1, 0], simi[g1, 1], simi[g1, 2],
        simi[g2, 0], simi[g2, 1], simi[g2, 2],
        simi[g3, 0], simi[g3, 1], simi[g3, 2],
        simi[g4, 0], simi[g4, 1], simi[g4, 2]]

print(simi[g1].shape)
print(simi[g2].shape)
print(simi[g3].shape)
print(simi[g4].shape)

fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.set_window_title('Boxplot for similarity')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)


plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting


ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Comparison of similarity')
ax1.set_xlabel('SNR')
ax1.set_ylabel('similarity')

# Now fill the boxes with desired colors
boxColors = ['darkkhaki', 'gold', 'green']
numDists = 4
numBoxes = numDists*3
medians = list(range(numBoxes))

for i in range(numBoxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    
    #print(boxX)
    # Alternate between Dark Khaki and Royal Blue
    k = i % 3
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    ax1.add_patch(boxPolygon)
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
             color='w', marker='*', markeredgecolor='k')

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, numBoxes + 0.5)
top = 1
bottom = 0
ax1.set_ylim(bottom, top)
xtickNames = plt.setp(ax1, xticklabels=np.repeat(xlabel, 3))
plt.setp(xtickNames, rotation=45, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(numBoxes) + 1
upperLabels = [str(np.round(s, 3)) for s in medians]
weights = ['bold', 'semibold']
for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
    k = tick % 2
    ax1.text(pos[tick], top - (top*0.05), upperLabels[tick],
             horizontalalignment='center', size='x-small', weight=weights[k],
             color=boxColors[0])

# Finally, add a basic legend
plt.figtext(0.80, 0.13, 'raw                       ',
            backgroundcolor=boxColors[0], color='black', weight='roman',
            size='x-small')
plt.figtext(0.80, 0.1,  'wavelet denoising',
            backgroundcolor=boxColors[1],
            color='black', weight='roman', size='x-small')
plt.figtext(0.80, 0.065, 'ResNet denoising ',
            backgroundcolor=boxColors[2],
            color='black', weight='roman', size='x-small')


plt.figtext(0.80, 0.015, '*', color='white', backgroundcolor='silver',
            weight='roman', size='medium')
plt.figtext(0.815, 0.013, ' Average Value', color='black', weight='roman',
            size='x-small')

plt.show()