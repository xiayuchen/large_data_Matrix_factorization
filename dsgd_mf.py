import os
import sys
import numpy as np
import csv
import math
from pyspark import SparkContext, SparkConf
def my_fun(s):
    l=s.split(',')
    return l
def strata_id(index):
    i_i=int(index[0])
    i_j=int(index[1])
    x=(i_i-1)/(bsize.value[1]);
    y=(i_j-1)/(bsize.value[2]);
    s_index=(x-y)%bsize.value[0];
    b_index=x
    return ([s_index,b_index],index)

def block_id(s):
    x=s[0][1]
    y=s[1]
    return (x,y)

def sgd_update(W,H,iterator):
    W_local={}
    H_local={}
    L_NZ=0.0
    prev=0
    bsize_u=bsize.value[1]
    bsize_f=bsize.value[2]
    for s in iterator:
        tao=int((prev_nb.value+prev)/(total_b.value))*10
        step=pow(tao+100,-beta.value)
        prev+=1
        content=s[1]
        i=int(content[0])
        j=int(content[1])
        b_id_i=(i-1)/(bsize_u)
        b_id_j=(j-1)/(bsize_f)
        score=float(content[2])
        predict=np.dot(W.value[i],H.value[j])
        #W.value[i]=W.value[i]+0.0005*((score-predict)*H.value[j])
        #H.value[j]=H.value[j]+0.0005*((score-predict)*W.value[i])
        W.value[i]=W.value[i]+step*((score-predict)*H.value[j]+2*lamda.value/N_i.value[i]*W.value[i])
        H.value[j]=H.value[j]+step*((score-predict)*W.value[i]+2*lamda.value/N_j.value[j]*H.value[j])
        L_NZ+=(predict-score)*(predict-score)
    for j in xrange(1,bsize_u+1):
        t=bsize_u*b_id_i+j
        W_local[t]=W.value[t]
    #print "bsize_u="+str(bsize_u)
    #print "end of block i="+str(t)
    for j in xrange(1,bsize_f+1):
        t=bsize_f*b_id_j+j
        H_local[t]=W.value[t]
    #print "bsize_f="+str(bsize_f)
    #print "end of block j = "+str(t)
    #print L_NZ
    #print prev
    #print W_local
    print L_NZ/prev
    return [W_local,H_local,L_NZ,prev]

def N_distr_i(s):
    i=int(s[0])
    return [i]


def N_distr_j(s):
    j=int(s[1])
    return [j]

data_path=sys.argv[6]
block_size=int(sys.argv[2])
conf = SparkConf().setAppName("test").setMaster("local["+str(block_size)+"]")
sc = SparkContext(conf=conf)
lines=sc.textFile(data_path)
line_content=lines.map(my_fun)
max_id=line_content.reduce(lambda a,b: [max(int(a[0]),int(b[0])),max(int(a[1]),int(b[1]))])
bsize_u=int(math.ceil(float(max_id[0])/float(block_size)))
bsize_f=int(math.ceil(float(max_id[1])/float(block_size)))
bsize=sc.broadcast([block_size,bsize_u,bsize_f])
factor_n=int(sys.argv[1])
lamda=float(sys.argv[5])
beta=float(sys.argv[4])
csv_W=sys.argv[7]
csv_H=sys.argv[8]
ite_time=int(sys.argv[3])
lamda=sc.broadcast(lamda)
beta=sc.broadcast(beta)
line_strata=line_content.map(strata_id)
N_i_j=line_content.map(N_distr_i)
Ns=N_i_j.countByKey().items()
N_i=[0]*(max_id[0]+1)
for pair in Ns:
    ind=pair[0]
    value=pair[1]
    N_i[ind]=value
#for i in Ns:
    #print Ns
N_i_j=line_content.map(N_distr_j)
Ns=N_i_j.countByKey().items()
N_j=[0]*(max_id[0]+1)
for pair in Ns:
    ind=pair[0]
    value=pair[1]
    N_j[ind]=value
total=sum(N_i)
total_b=sc.broadcast(total)
W={}
H={}
for i in xrange(1,bsize_u*block_size+1):
    W[i]=np.random.rand(factor_n)
for i in xrange(1,bsize_f*block_size+1):
    H[i]=np.random.rand(factor_n)

W_b=sc.broadcast(W)
H_b=sc.broadcast(H)
N_i=sc.broadcast(N_i)
N_j=sc.broadcast(N_j)
prev_n=0
error_old=-1.0
error_new=0.0
for ite in xrange (0,ite_time):
#while(abs((error_old-error_new)/error_old)<0.001):
    error=0.0
    for i in range (0,block_size):
        prev_nb=sc.broadcast(prev_n)
        strated=line_strata.filter(lambda s : s[0][0]==i).map(block_id)
        block = strated.partitionBy(block_size)
        W_H= block.mapPartitions(lambda x: sgd_update(W_b,H_b,x))
        update_W_H=W_H.glom().reduce(lambda a,b : [dict(a[0],**b[0]),dict(a[1],**b[1]),a[2]+b[2],a[3]+b[3]])
        W_b=sc.broadcast(update_W_H[0])
        H_b=sc.broadcast(update_W_H[1])
        prev_n+=update_W_H[3]
        error+=update_W_H[2]
        #print pow(prev_n+100,-beta.value)
    error_old=error_new
    error_new=error/float(total)
    #if(abs((error_old-error_new)/error_old)<0.001):
        #break;
    print str(error/float(total))
csvfile=file(csv_W,'w')
writer=csv.writer(csvfile)
for i in xrange(1,max_id[0]+1):
    writer.writerow(update_W_H[0][i])
#writer.writerows(update_W_H[0])
csvfile.close()
csvfile=file(csv_H,'w')
writer=csv.writer(csvfile)
for i in xrange(1,max_id[1]+1):
    writer.writerow(update_W_H[1][i])
csvfile.close()
#writer.writerows(update_W_H[1])
    #print W_a
    #W=W_a.glom().reduce(lambda a,b : dict(a,**b))
    #H=H_a.reduce(lambda a,b : dict(a,** b)).collect()
    #for c in W:
    #    print c
