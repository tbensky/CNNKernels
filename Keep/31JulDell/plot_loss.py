import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("loss.csv")

plt.plot(df['t'],df['vx'],label="$v_x$")


plt.subplot(3,1,1)
plt.plot(loss_track['epoch'],loss_track['loss'])
plt.ylabel("Loss")
plt.xticks([])
plt.title("Convolutional Network Training")

plt.subplot(3,1,2)
plt.plot(loss_track['epoch'],loss_track['train_correct'])
plt.xticks([])
plt.ylabel("Train Correct")

plt.subplot(3,1,3)
plt.plot(loss_track['epoch'],loss_track['test_correct'])
plt.ylabel("Test Correct")
plt.xlabel("Epoch")
plt.xticks()

plt.savefig("loss.png",dpi=300)
plt.close()