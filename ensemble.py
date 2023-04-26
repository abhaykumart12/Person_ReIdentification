import os
import os.path as osp
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Ensemble')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='model names')
parser.add_argument('--topn',default=15, type=int, help='value of n')
parser.add_argument('--query_path',default="./", type=str, help='value of n')
opts = parser.parse_args()

def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def readfile(filename,dir):
    f=open(dir+filename,"r")
    lines=f.readlines()
    arr=[]
    for line in lines:
        line=line.strip()
        l=line.split(" ")
        arr.append((l[0],float(l[1])))
    f.close()
    return arr

s=opts.name
models=s.split(",")
N=opts.topn
k=N//len(models)
if(osp.isdir("./results/ensemble")==False):
    os.mkdir("./results/ensemble")
if(osp.isdir("./results/ensemble/m1")==False):
    os.mkdir("./results/ensemble/m1")
if(osp.isdir("./results/ensemble/m2")==False):
    os.mkdir("./results/ensemble/m2")
files=os.listdir("./results/"+models[0])
for i in files:
    campus=readfile(i,"./results/campus/")
    bus=readfile(i,"./results/bus/")
    mall=readfile(i,"./results/mall/")
    #traffic=readfile(i,"/content/Person_ReID/results/traffic/")
    #supermarket=readfile(i,"/content/Person_ReID/results/supermarket/")
    query_path=opts.query_path+"/"+i[:3]+"/"+i[:-4]+".jpg"
    #method 1
    #take top N/k from all the datasets
    result=campus[:5]+bus[:5] +mall[:5] #+traffic[:2]+supermarket[:2]
    result.sort(key=lambda x:x[1],reverse=True)
    f=open("./results/ensemble/m1/"+i,"w")
    temp=[]
    for j in result:
        temp.append(j[0])
    s="\n".join(temp)
    f.write(s)
    f.close()



    query_label=int(i.split("_")[0])
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,16,1)
    ax.axis('off')
    imshow(query_path,'query')
    c=1
    for j in temp:
        ax = plt.subplot(1,16,c+1)
        ax.axis('off')
        fname=j.split("/")[-1]
        label=int(fname.split("_")[0])
        imshow(j)
        if label == query_label:
            ax.set_title('%d'%(c), color='green')
        else:
            ax.set_title('%d'%(c), color='red')
        c+=1

    fig.savefig("./results/ensemble/m1/"+i[:-4]+".jpg")
    plt.close('all')
    #method 2
    #sort all the results and take top N
    result=campus[:N]+bus[:N]+mall[:N] #+traffic+supermarket
    result.sort(key=lambda x:x[1],reverse=True)
    temp=[]
    for j in range(N):
        temp.append(result[j][0])
    s="\n".join(temp)
    f=open("./results/ensemble/m2/"+i,"w")
    f.write(s)
    f.close()

    query_label=int(i.split("_")[0])
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,16,1)
    ax.axis('off')
    imshow(query_path,'query')
    c=1
    for j in temp:
        ax = plt.subplot(1,16,c+1)
        ax.axis('off')
        fname=j.split("/")[-1]
        label=int(fname.split("_")[0])
        imshow(j)
        if label == query_label:
            ax.set_title('T', color='green')
        else:
            ax.set_title('F', color='red')
        c+=1
    fig.savefig("./results/ensemble/m2/"+i[:-4]+".jpg")
    plt.close('all')

os.system("python evaluate_ensemble.py")
