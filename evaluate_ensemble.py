import os
import os.path as osp


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

cmc1=[0 for i in range(15)]
cmc2=[0 for i in range(15)]
def calculate_rank(result,query_label,method):
    c=0
    for i in range(15):
        if(c==1):
            if(method==1):
                cmc1[i]+=1
            else:
                cmc2[i]+=1
            continue
        img_name=result[i][0].split("/")[-1]
        g_label=int(img_name[:3])
        if(g_label==query_label):
            c=1
            if(method==1):
                cmc1[i]+=1
            else:
                cmc2[i]+=1


aps=[0.0,0.0]
def calculate_map(result,query_label,method):
    c=0
    curr=0.0
    for i in range(15):
        img_name=result[i][0].split("/")[-1]
        g_label=int(img_name[:3])
        if(g_label==query_label):
            c+=1
            curr+=c/(i+1)
    if(c==0):
        return
    curr=curr/c
    if(method==1):
        aps[0]+=curr
    else:
        aps[1]+=curr

files=os.listdir("./results/campus")
q=len(files)
for f in files:
    query_label=int(f[:3])
    campus=readfile(f,"./results/campus/")
    bus=readfile(f,"./results/bus/")
    mall=readfile(f,"./results/mall/")

    N=(len(bus)+len(campus)+len(mall))//3
    T=N//3
    #method 1
    #result=campus[:T]+bus[:T]+mall[:T]
    x1=campus[:5]+bus[:5]+mall[:5]
    x1.sort(reverse=True,key=lambda x: x[1])
    #result.sort(reverse=True,key=lambda x: x[1])
    calculate_rank(x1,query_label,1)
    calculate_map(x1,query_label,1)

    #method 2
    result=campus+bus+mall
    result.sort(reverse=True,key=lambda x: x[1])
    result=result[:15]
    calculate_rank(result,query_label,2)
    calculate_map(result,query_label,2)


print("Final Result:")
print("-----------------------------------")
print("For method 1:")
print("Rank 1: ",cmc1[0]/q)
print("Rank 5: ",cmc1[4]/q)
print("Rank 10: ",cmc1[9]/q)
print("Rank 15: ",cmc1[14]/q)
print("mAP: ",aps[0]/q)
print("-----------------------------------")
print("For method 2:")
print("Rank 1: ",cmc2[0]/q)
print("Rank 5: ",cmc2[4]/q)
print("Rank 10: ",cmc2[9]/q)
print("Rank 15: ",cmc2[14]/q)
print("mAP: ",aps[1]/q)