import math
import random


HEIGHT = 500
WIDTH = 500
SIZE = (HEIGHT*WIDTH)

screen = ['0'] * SIZE

def cls():
    for row in range(HEIGHT):
        for col in range(WIDTH):
            screen[row*WIDTH+col] = 0

def clip_data(x):
    if x == 1:
        return 0.8
    return 0.1


def plot(x,y,c):
    offset = (HEIGHT/2-y)*WIDTH + (WIDTH/2+x)
    if offset > 0 and offset < SIZE:
        screen[int(offset)] = c

def plotbig(x,y,c):
    #plot(x,y,c)
    #return
    
    for dx in range(-1,1):
        for dy in range(-1,1):
            plot(x+dx,y+dy,c)

def dump():
    for row in range(HEIGHT):
        for col in range(WIDTH):
            offset = row*WIDTH + col
            print(screen[offset],end='')
        print()

def dump_pair(output):
    print("[[",end="")
    for i in range(SIZE-1):
        print(f"{clip_data(screen[i])},",end="")
    print(f"{clip_data(screen[i])}",end="")
    output1 = [clip_data(x) for x in output]
    print(f"],{output1}",end="")
    print("]",end="")



def gauss(A,x0,sigma,offset):
    x = -HEIGHT/2
    while x < HEIGHT/2:
        y = int(offset) + int(A*math.exp(-((x-x0)**2)/(2*math.pi*sigma**2)))
        plotbig(x,y,1)
        x += 0.1

def square(A,x0,width,offset):
    x = -WIDTH/2
    while x < WIDTH/2:
        if x <= x0 or x >= (x0 + width):
            y = int(offset)
        else:
            y = int(offset + A)
        plotbig(x,y,1)
        x += 1

    for y in range(int(offset),int(A + offset)+1):
        plotbig(x0,y,1)
        plotbig(x0+width,y,1)
    
def triangle(A,x0,width,offset):
    x = -WIDTH/2
    y = offset
    m = 0.5
    while x < WIDTH/2:
        if x % int(width) == 0:
            m *= -1
        y += m * x
        plotbig(x,y,1)
        x += 1

#f=1..10 seems to work best
def osc(A,f,offset):
    x = -WIDTH/2
    offset = 0
    while x < WIDTH/2:
        y = offset - int(A*math.sin(x/20))
        plotbig(x,y,1)
        x += 0.1


def get_params():
    A = random.uniform(10,300)
    x0 = random.uniform(-300,300)
    sd = random.uniform(1,50)
    width = random.uniform(2,100)
    offset = random.uniform(-20,200) #0
    return A,x0,sd,width,offset


stats = {"gauss": 0, "square": 0, "triangle": 0}

print("[") # open json


N = 100
for i in range(0,N):
    cls()
    A,x0,sd,width,offset = get_params()
    n = random.randint(0,2)

    if n == 0:
        gauss(A,x0,sd,offset)
        output = [0,0,1]
        stats['gauss'] += 1
    if n == 1:
        square(A,x0,width,offset)
        output = [0,1,0]
        stats['square'] += 1
    if n == 2:
        #triangle(A,x0,5*width,offset)
        osc(A,2,offset)
        output = [1,0,0]
        stats['triangle'] += 1

    dump_pair(output)
    if i < N-1:
        print(",")


print("\n]") # close json

#print(stats)
