#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:14:30 2018

@author: MONTALDO - WU
"""

from Tkinter import *
import numpy as np
from PIL import Image
from skimage import io as skio # necessite scikit-image
import itertools
import time
import datetime

#
#
def getStringFromRGB(imgMat, y, x):
    global imgMatUint
    colorString = '#'
    if B == 3: 
        for b in range(B):
            colorString += hex(imgMatUint[b][y][x])[2:].rjust(2, '0')
    #manage weird case B=4 like this
    #for debugging purpose
    elif B == 4:
        for b in range(3):
            colorString += hex(imgMatUint[b][y][x])[2:].rjust(2, '0')
    else:
        for b in range(3):
            colorString += hex(imgMatUint[0][y][x])[2:].rjust(2, '0')
    return colorString

#Draw image in canvas
def drawImage():
    global imgMat
    for h in range(0, H):
        for w in range(0, W):
            widget.create_rectangle(w*pixS, h*pixS, (w+1)*pixS, (h+1)*pixS, fill=getStringFromRGB(imgMat, h, w), outline="")

#Draw target
def drawTarget():
    global canvas
    global targMat
    
    for h in range(0, H):
        for w in range(0, W):
            if targMat[h][w]==0:
                strCol = '#000000'
            elif targMat[h][w]==1:
                strCol = '#ffffff'
            else:
                strCol = '#3f3f3f'
            widget.create_rectangle(w*pixS, h*pixS, (w+1)*pixS, (h+1)*pixS, fill=strCol, outline="")
    
    #save target in file
    outMat = np.zeros((H, W), dtype = 'uint8') #will hold target area
    for h in range(0, H):
        for w in range(0, W):
            if targMat[h][w]==1:
                outMat[h][w] = 255
    skio.imsave('data/targetArea.tif', outMat)

#mouse click happened
def callback(event):
    #pixel indexes
    ezerx = event.x / pixS - 2 
    ezery = event.y / pixS - 2
    
    #check if inpaint button has been clicked
    if event.x > (canvas_width/2)-50 and event.x < (canvas_width/2)+50 and event.y > canvas_height-10-150 and event.y < canvas_height+10-150: 
        inpaint()
    #check if show target button has been clicked
    elif event.x > (canvas_width/2)-50 and event.x < (canvas_width/2)+50 and event.y > canvas_height-10-50 and event.y < canvas_height+10-50:
        drawTarget()
        
    #check if show inpainted image button has been clicked
    #elif event.x > (canvas_width/2)-50 and event.x < (canvas_width/2)+50 and event.y > canvas_height-10-20 and event.y < canvas_height+10-20:
    #    drawImage()
    #check if the click is inside the image
    elif event.x > 0 and event.x < pixS*W and event.y > 0 and event.y < pixS*H:
        #print('yes')
        #addpixel(ezerx,ezery)
        drawBrush(ezery, ezerx) #Draw big brush pixel
        updateMap(ezery, ezerx)

#mouse dragged happened
def onLeftDrag(event):
    ezerx = event.x/pixS-2
    ezery = event.y/pixS-2
    #addpixel(ezerx,ezery)
    #check if the click is inside the image
    if event.x > 0 and event.x < pixS*W and event.y > 0 and event.y < pixS*H:
        drawBrush(ezery, ezerx) #Draw big brush pixel
        updateMap(ezery, ezerx)

#draw windows
def drawcanvas(canvas_height, canvas_width):
    #for i in range(L+2):
    #    widget.create_line(pixS, i * pixS, L*pixS + pixS, i * pixS, width=1)
    #    widget.create_line(i * pixS, pixS, i * pixS, L*pixS + pixS, width=1)
    drawImage()
    
    #inpaint button
    widget.create_rectangle((canvas_width/2)-50, canvas_height-10-150, (canvas_width/2)+50,  canvas_height+10-150, fill='#111111')
    widget.create_text((canvas_width/2, canvas_height-150), text="inpaint", fill="white") #position is the center of text
    
    #label
    global text
    text = widget.create_text(canvas_width/2, canvas_height-120, text="Select target area")
    
    #show target area button
    widget.create_rectangle((canvas_width/2)-50, canvas_height-10-50, (canvas_width/2)+50,  canvas_height+10-50, fill='#3f3f3f')
    widget.create_text((canvas_width/2, canvas_height-50), text="show target", fill="white") #position is the center of text
    
    #show result img button
    #widget.create_rectangle((canvas_width/2)-50, canvas_height-10-20, (canvas_width/2)+50,  canvas_height+10-20, fill='#3f3f3f')
    #widget.create_text((canvas_width/2, canvas_height-20), text="result image", fill="white") #position is the center of text
    
def reset():
    global targMat
    targMat = np.zeros((H, W), dtype = 'uint8')
    for pixel in drawedpixels:
        widget.delete(pixel)

#Draw Brush
def drawBrush(imgy, imgx):
    imgy -= 1 #pass from pixel number to indexing zero based
    imgx -= 1
    
    #draw brush
    for x in range(-brushSize, +brushSize+1):
        for y in range(-brushSize, +brushSize+1):
            drawedpixels.append(widget.create_rectangle((imgx+x)*pixS, (imgy+y)*pixS, (imgx+x+1)*pixS, (imgy+y+1)*pixS, fill='#ff0000', outline=""))

#Update target area map
def updateMap(imgy, imgx):
    #global mapFlag
    
    #mapflag = 1
    imgy -= 1 #pass from pixel number to indexing zero based
    imgx -= 1
    global targMat
    #canvas.create_rectangle(imgx*pixS, imgy*pixS, (imgx+1)*pixS, (imgy+1)*pixS, fill='#ff0000', outline="")
    for x in range(-brushSize, +brushSize+1):
        for y in range(-brushSize, +brushSize+1):
            targMat[imgy+y][imgx+x]=1

#find point on the frontier            
def findFrontierPoints():
    global targMat
    
    x=[] #list of list of coordinates points
    
    oldVal = 0 #initialize with 0 (source area)
    
    #vertically
    for w in range (0,W):
        for h in range (1,H-1): #skip first and last line
            if oldVal == 0: #we are in source area and look for ones
                if targMat[h][w] == 1: 
                    x.append([w,h])
                    oldVal = 1
            else: #we are in target area and look for zeros
                if targMat[h][w] == 0: 
                    x.append([w,h-1])
                    oldVal = 0  
    oldVal = 0 #reset flag
    #horizzontally
    for h in range (0,H): 
        for w in range (1,W-1): #skip first and last column
            if oldVal == 0: #we are in source area and look for ones
                if targMat[h][w] == 1: 
                    x.append([w,h])
                    oldVal = 1
            else: #we are in target area and look for zeros
                if targMat[h][w] == 0: 
                    x.append([w-1,h])
                    oldVal = 0           
    x.sort()
    return list(x for x,_ in itertools.groupby(x))

def gradx(im):
    "renvoie le gradient dans la direction x"
    imt=np.float32(im)
    gx=0*imt
    gx[:,:-1]=imt[:,1:]-imt[:,:-1]
    return gx

def grady(im):
    "renvoie le gradient dans la direction y"
    imt=np.float32(im)
    gy=0*imt
    gy[:-1,:]=imt[1:,:]-imt[:-1,:]
    return gy

def inpaint():
    global targMat
    global confMat
    global imgMat
    global imgMatUint
    global mapFlag

    #if no target has been given import the last one
    if targMat.sum() == 0:
        targetPath = 'data/targetArea.tif'
        #---tif format
        im1=skio.imread(targetPath)
        
        print('target image imported')
        
        #fill target matrix
        for h in range(0, H):
            for w in range(0, W):
                #---tif format
                    if im1[h][w] == 255:
                        targMat[h][w] = 1 #gray image

    #Debug
    #delete pixel values in the image corresponding to target area
    for h in range(0, H):
        for w in range(0, W):
            if targMat[h][w]==1:
                    for b in range(B):
                        imgMat[b][h][w] = 0
    
    start_time = time.time() #start timer
    
    confMat = np.zeros([H, W])+1 #will hold confidence values
    selectionOrderMat = np.zeros([H, W]) #will hold the number of selection for pixel in target area
    orderCounter = 1
    
    #initialize confidence matrix
    #1 in the source, 0 in the target
    for h in range(0, H):
        for w in range(0, W):
            if targMat[h][w]==1:
                confMat[h][w] = 0
    
    tmpConfMat = confMat #use for store temporary confidences
    
    frontierPoints = [1]
    iterationCounter = 0
    #---Repeat until there are no more frontier points
    while frontierPoints != []:
        priorMat = np.zeros([H, W]) #will hold priority values
        
        #find point on the frontier
        frontierPoints = [] #empty out list
        frontierPoints = findFrontierPoints()
        
        #compute modulus of the gradient for every band
        modGradList = [] #list of mod Grad for every band
        
        for b in range(B):
            im = np.zeros([H, W])
            
            for h in range(0, H):
                for w in range(0, W):
                    im[h][w] = imgMat[b][h][w]
            #grad on x
            imgx=gradx(im)
            #grad on y
            imgy=gradx(im)
            modGrad = np.square(imgx) + np.square(imgy)
            modGradList.append(np.sqrt(modGrad))
        
        #--compute priorities for each frontier point
        #for every point in list of frontieer list, with x,y coordinates
        for point in frontierPoints:
            x = point[0]
            y = point[1]
            
            #compute C(p) for each of them using definition
            sumConf = 0
            for h in range(-patchHalfDim, +patchHalfDim+1):
                for w in range(-patchHalfDim, +patchHalfDim+1):
                    sumConf += confMat[h+y][w+x]
            tmpConfMat[y][x] = sumConf / ((2*patchHalfDim+1)**2)
        
            #compute the D(p)
            #the computation is done as the local average of the values of the 
            #pixel that stay in the intersection of the source area and the patch
            dataTerm = 0
            for b in range(B):
                count = 0
                for h in range(-patchHalfDim, +patchHalfDim+1):
                    for w in range(-patchHalfDim, +patchHalfDim+1):
                        if targMat[y+h][x+w] == 0: #if we are in the source area
                            dataTerm += modGradList[b][y+h][x+w]
                            count += 1
                dataTerm /= count
            dataTerm /= B
                        
            #compute priority P(p)
            #priorMat[y][x] = tmpConfMat[y][x]
            priorMat[y][x] = tmpConfMat[y][x]*(dataTerm/255)
            
        #--find point with maximum priority
        #xMax and yMax are the center of the patch that must be substituted
        maxVal = 0
        for h in range(0, H):
            for w in range(0, W):
                if priorMat[h][w] > maxVal:
                    maxVal = priorMat[h][w]
                    xMax = w
                    yMax = h
        
        print('max priority point')
        print(xMax)
        print(yMax)
        
        #--find examplar that minimizes patches distance
        minDist = float("inf")
        for hPatch in range(patchHalfDim, H-patchHalfDim):
            for wPatch in range(patchHalfDim, W-patchHalfDim):
                
                #look that examplar does not belong to target area
                flag = 0
                for h in range(-patchHalfDim, +patchHalfDim+1):
                    for w in range(-patchHalfDim, +patchHalfDim+1):
                        if targMat[hPatch+h][wPatch+w]==1:
                            flag = 1
                if flag==0:
                    #compute distance
                    sumVal = 0
                    for h in range(-patchHalfDim, +patchHalfDim+1):
                        for w in range(-patchHalfDim, +patchHalfDim+1):
                            if targMat[h+yMax][w+xMax]==0: #if the position correspond to the source area
                                for b in range(B):  
                                        sumVal += abs(imgMat[b][h+yMax][w+xMax] - imgMat[b][h+hPatch][w+wPatch])
                    if sumVal < minDist:
                        minDist = sumVal
                        xBestPatch = wPatch
                        yBestPatch = hPatch
        
        print('best match found')
        print(xBestPatch)
        print(yBestPatch)
        
        #--for every pixel in the intersection of target area and the patch
        for h in range(-patchHalfDim, +patchHalfDim+1):
            for w in range(-patchHalfDim, +patchHalfDim+1):
                if targMat[h+yMax][w+xMax]==1: #if the position correspond to the target area
                    #--copy examplar found in the image
                    if B == 1:
                        imgMat[0][h+yMax][w+xMax] = imgMat[0][h+yBestPatch][w+xBestPatch] #inpaint image
                    else:
                        for b in range(B):
                            imgMat[b][h+yMax][w+xMax] = imgMat[b][h+yBestPatch][w+xBestPatch] #inpaint image
                    #update confidence matrix
                    
                    sumConf = 0
                    for h2 in range(-patchHalfDim, +patchHalfDim+1):
                        for w2 in range(-patchHalfDim, +patchHalfDim+1):
                            sumConf += confMat[h2+h+yMax][w2+w+xMax]
                    confMat[h+yMax][w+xMax] = sumConf / ((2*patchHalfDim+1)**2)
                    
                    #modify selectionOrderMat
                    selectionOrderMat[h+yMax][w+xMax] = orderCounter
                    
                    #update target region
                    targMat[h+yMax][w+xMax] = 0
        
        orderCounter += 1
        iterationCounter += 1
    
    #results
    print('inpainting done with: ', iterationCounter, ' iterations')
    print('in: ', time.time() - start_time, ' sec')
    
    #save result image in unique name file
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    filePath = 'data/output/' + st + 'output.tif'
    imgMatUint = imgMat.astype('uint8')
    skio.imsave(filePath, imgMatUint)
    
    #process selectionOrderMat for being displayed in image and save to unique file
    for h in range(0, H):
        for w in range(0, W):
            if selectionOrderMat[h][w] != 0:
                a = int(255/iterationCounter) - 1
                if a > 1:
                    selectionOrderMat[h][w] *= a
            if selectionOrderMat[h][w] > 255:
                selectionOrderMat[h][w] = 255
    ''' BETTER DISPLAYING - TO BE TESTED
    M = selectionOrderMat.max()
    L = 50
    for h in range(0, H):
        for w in range(0, W):
            if selectionOrderMat[h][w] != 0:
                inp = selectionOrderMat[h][w]
                selectionOrderMat[h][w] = ((L-255)*inp)/(M-1)
                selectionOrderMat[h][w] += (L-255)*((-1/(M-1))+(255/(L-255)))
    ''' 
    filePath = 'data/output/' + st + 'orderSelection.tif'
    selectionOrderMatUint = selectionOrderMat.astype('uint8')
    skio.imsave(filePath, selectionOrderMatUint)
    
    #GUI updates
    string = 'results exported in data/output.tif'
    widget.itemconfig(text,text=string) #example change font
    #widget.itemconfig(text,text="cambio",font=("Purisa", 20)) #with font
    #clear canvas and target area
    reset()
    #display image result
    drawImage()
                                                
#-----------------------------------------------------
#-----------------------------------------------------
#-----------------------------------------------------
        
global B                #num of bands
global patchHalfDim     #num of bands
global imgMat           #matrix value for images pixel
global imgMatUint
global widget
global brushSize

# Constants       
pixS = 2            #pixel size in the canvas
patchHalfDim = 2    #patch dimension is 2*patchDim+1
brushSize = 3       #dimension of the brush for drow the target area
imagePath = 'data/image4_2.tiff'    #image to be used

#some path for other examples:
#imagePath = 'data/image2_V2.tiff'

#color:  'data/lena_color.tiff'
#        'data/fleur.tif'
#gray:   'data/dice1.tif'
#        'data/lena_petit.tif'         
#gif:    'data/pianeta.gif'  STILL TO BE FIXED THE GIF

#variables
drawedpixels = [] #list of drawed pixels for marking target area

#-----Import image
#---gif format
#im = Image.open('data/pianeta.gif') 
#rgbimage = im.convert('RGB')
#W, H = im.size
#---tif format
im=skio.imread(imagePath)
#get number of bands
if(len(im.shape) == 3): #color image
    B = im.shape[2]
else: #grey image
    B = 1
#get dimensions
H = im.shape[0]
W = im.shape[1]

print('image imported with: W ', W, ' H ', H, ' B ', B)

imgMat = np.empty([B, H, W]) #will hold image values
targMat = np.zeros((H, W), dtype = 'uint8') #will hold target area
                                            #obs: 0->not belonging, 1->belonging
#fill image matrix with image values
for b in range(0, B):
    for h in range(0, H):
        for w in range(0, W):
            #---gif format
            #imgMat[b][h][w] = rgbimage.getpixel((w,h))[b]
            #---tif format
            if B == 1:
                imgMat[b][h][w] = im[h][w] #gray image
            else:
                imgMat[b][h][w] = im[h][w][b] #color image

imgMatUint = imgMat.astype('uint8')

#GUI
tkroot = Tk()
canvas_width = (W + 2)*pixS
canvas_height = (H + 2)*pixS + 180 #180 for contents below
widget = Canvas(tkroot,width=canvas_width,height=canvas_height)
widget.pack(expand=YES, fill=BOTH)

drawcanvas(canvas_height, canvas_width)
widget.bind('<B1-Motion>', onLeftDrag)
widget.bind("<Button-1>", callback)
widget.focus()
tkroot.title('Inpainting project')
tkroot.mainloop()























