import math
import random
import numpy as np
import matplotlib.pyplot as plt

WIDTH = 500
HEIGHT = 500


def get_params():
    A = random.uniform(5,500)
    x0 = random.uniform(-300,300)
    sd = random.uniform(1,50)
    width = random.uniform(2,100)
    offset = random.uniform(-200,200-A) #0
    freq = random.uniform(0.01,0.25)
    B = random.uniform(1,5)
    tw = random.uniform(50,500)
    return A,x0,sd,width,offset,freq,B,tw


stats = {"gauss": 0, "square": 0, "triangle": 0,"sin":0}


def xy(x1,y1,x2,y2):
    return [x1,x2],[y1,y2]

x = np.linspace(-250,250,1000)
N = 1000
for i in range(0,N):
    A,x0,sd,width,offset,f,B,tw = get_params()
    n = random.randint(0,3)
    plt.xlim([-250,250])
    plt.ylim([-250,250])
    plt.axis('off')
    plt.gray()
    
    if n == 0:
        plt.plot(x,A*np.exp(-((x-x0)**2)/(2*math.pi*sd**2))+offset,'k')
        plt.savefig(f"Data/image_{i:04d}.png")
        plt.close()
        output = [0,0,1]
        stats['gauss'] += 1
    if n == 1:
        plt.plot([-WIDTH/2,x0],[offset,offset],'k')
        plt.plot([x0,x0],[offset,offset+A],'k')
        plt.plot([x0,x0+width],[offset+A,offset+A],'k')
        plt.plot([x0+width,x0+width],[offset+A,offset],'k')
        plt.plot([x0+width,WIDTH/2],[offset,offset],'k')
        plt.savefig(f"Data/image_{i:04d}.png")
        plt.close()
        output = [0,1,0]
        stats['square'] += 1
    if n == 2:
        plt.plot(x,A/2*np.sin(f*x)+offset,'k')
        plt.savefig(f"Data/image_{i:04d}.png")
        plt.close()
        output = [1,0,0]
        stats['sin'] += 1
    if n == 3:
        plt.plot(x,1*(x % tw)+offset,'k')
        plt.savefig(f"Data/image_{i:04d}.png")
        plt.close()
        output = [1,1,0]
        stats['triangle'] += 1

    with open(f"Data/output_{i:04d}.dat","w") as f:
        f.write(f"{output}\n")

    if i % 100 == 0:   
        print(i)


print(stats)
