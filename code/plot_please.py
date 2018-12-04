import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))

PATH = './test/'
for i in range(4):
    IoU = []
    epoch = []
    with open(PATH+'config_%d/testIoU.txt' % (i+1), 'r') as f:
        lines = f.readlines()
        for line in lines:
            IoU.append(float(line.split()[1]))
            epoch.append(int(line.split()[0])/1000)
    plt.plot(epoch, IoU, label='Configuration%d' % (i+1), linewidth=5.0)

plt.legend(loc='best')
plt.xlabel("epoch")
plt.ylabel("IoU")
plt.title('Intersection over Union vs Epochs')
plt.savefig('results.png')