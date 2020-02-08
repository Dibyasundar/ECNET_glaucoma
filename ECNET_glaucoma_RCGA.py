#!/usr/bin/env python


import scipy.io as io
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Layer, Activation, MaxPool2D
import tensorflow as tf
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as in_out


if not os.path.exists('Dataset.mat'):
    data_path='/home/drago/store/Dataset_Raw/Glaucoma/Deepak'
    dirs=os.listdir(data_path)
    img_ext=['jpg','tif','png','bmp','gif']
    data_set=[];
    req_sz=(64,64);
    for folder in dirs:
        k=1;
        f_name=[fn for fn in os.listdir(data_path+'/'+folder) if any(fn.endswith(ext) for ext in img_ext)]
        for file_name in f_name:
            img=plt.imread(data_path+'/'+folder+'/'+file_name)
            img=img/255.0
            img=cv2.resize(img,req_sz,interpolation=cv2.INTER_CUBIC)
            if k==1:
                data=np.expand_dims(img,axis=0);
                k=0
            else:
                data=np.append(data,np.expand_dims(img,axis=0),axis=0)
        data_set.append(data)
    io.savemat('Dataset.mat',{'data_set':data_set})
    mat = io.loadmat('Dataset.mat')
else:
    mat = io.loadmat('Dataset.mat')



data_set=mat['data_set']
train_set=[]
test_set=[]
per=0.7
np.random.seed(100)
for i in range(data_set.shape[1]):
    n_data=data_set[0,i]
    sz = n_data.shape
    pos = np.random.permutation(sz[0])
    train_set.append(n_data[pos[0:round(per*sz[0])],:,:,:])
    test_set.append(n_data[pos[round(per*sz[0]):sz[0]],:,:,:])



plt.imshow(train_set[0][1,:,:,:])
plt.show()


class MyLayer(Layer):
    def __init__(self,dim, **kwargs):
        self.dim=dim
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(self.dim), initializer='uniform',trainable='false')
        super(MyLayer, self).build(input_shape)
    def call(self, x):
        return K.sum(tf.transpose(K.dot(tf.transpose(x,[0,3,1,2]), self.kernel),[0,2,3,1]),axis=3)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim[1])



def new_model(m,n,p):
    model=Sequential()
    model.add(Conv2D(32, kernel_size = (5,5), input_shape = (m,n,p), padding='same'))
    model.add(Activation('relu'))
    model.add(MyLayer((n,10)))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return(model)
def obj_eval(data, model):
    m = []
    v = []
    for i in range(len(data)):
        out = model.predict(data[i], steps=1)
        m.append(np.mean(out, axis=0))
        v.append(np.var(out, axis=0))
    mt=np.mean(np.array(m),axis=0)
    m_obj = 0;
    v_obj = 0
    for i in range(len(m)):
        for j in range(i+1,len(m),1):
            m_obj=m_obj+ np.sqrt(np.sum(np.square(m[i]-m[j])))
        m_obj=m_obj + np.sqrt(np.sum(np.square(mt-m[j])))
        v_obj=v_obj+np.sum(v[i])
    return m_obj, v_obj




sz = train_set[0].shape
model = new_model(sz[1],sz[2],sz[3])
model.summary()




def initialize_pop(w):
    pop=[]
    for i in range(len(w)):
        pop.append((-1+np.random.random(w[i].shape)*2)*0.01)
    return pop
def make_random(p):
    x=np.copy(p)
    for i in range(p.shape[0]):
        x[i]=-1+np.random.random(p[i].shape)*2
    return(x)



def get_feature(data,model):
    for i in range(len(data)):
        out = model.predict(data[i], steps=1)
        if i==0:
            feature = np.array(out)
            label = np.ones((out.shape[0],1))*i
        else:
            feature = np.append(feature,out,axis=0)
            label = np.append(label, np.ones((out.shape[0],1))*i,axis=0)
    return feature, label


def weight_reshape(g,w):
    print(g[0].shape)
    for i in range(len(g)):
        w[i]=np.reshape(g[i],w[i].shape)
    return(w)



if not os.path.exists('RCGA_plot.mat'):
    w = model.get_weights()
    pop_sz = 50 #population size
    cp=0.6 #crossover probability
    mp=0.01 #mutation probability
    pop = []
    vel = []
    for i in range(pop_sz):
        np.random.seed(i)
        pop.append(initialize_pop(w))
    pop=np.array(pop)
    m_obj=[]
    v_obj=[]
    for i in range(pop_sz):
        model.set_weights(pop[i])
        [m, v] = obj_eval(train_set,model)
        m_obj.append(np.copy(m))
        v_obj.append(np.copy(v))
    m_obj = np.array(m_obj)
    v_obj = np.array(v_obj)
    it=1
    MaxIter=100
    plot_g_m=[]
    plot_g_v=[]
    while it<= MaxIter:
        obj = 0.2*(m_obj/np.sum(m_obj)) -  0.8*(v_obj/np.sum(v_obj))
        g_best=np.argmax(obj)
        global_best=np.copy(pop[g_best])
        global_best_m_obj=np.copy(m_obj[g_best])
        global_best_v_obj=np.copy(v_obj[g_best])
        pos=np.argsort(-obj) #get location of sorted array in decending order
        for i in range(pop_sz):
            b_pos=np.random.permutation(10)
            npop = np.multiply(cp,pop[pos[b_pos[0]]]) + np.multiply(1-cp,pop[pos[i]])
            npop = npop + np.multiply(mp,make_random(pop[pos[i]]))
            model.set_weights(npop)
            [m, v] = obj_eval(train_set,model)
            if (m-v>m_obj[pos[i]]-v_obj[pos[i]]):
                pop[pos[i]] = np.copy(npop)
                m_obj[pos[i]] = np.copy(m)
                v_obj[pos[i]] = np.copy(v)
            npop = np.multiply(1-cp,pop[pos[b_pos[0]]]) + np.multiply(cp,pop[pos[i]])
            npop = npop + np.multiply(mp,make_random(pop[pos[i]]))
            model.set_weights(npop)
            [m, v] = obj_eval(train_set,model)
            if (m-v>m_obj[pos[i]]-v_obj[pos[i]]):
                pop[pos[i]] = np.copy(npop)
                m_obj[pos[i]] = np.copy(m)
                v_obj[pos[i]] = np.copy(v)
        plot_g_m.append(global_best_m_obj)
        plot_g_v.append(global_best_v_obj)
        print(it, global_best_m_obj,"   " ,global_best_v_obj," ",np.mean(m_obj)," ",np.mean(v_obj))
        it = it+1
    in_out.savemat('RCGA_plot.mat',mdict={'g_m':plot_g_m,'g_v':plot_g_v,'weight':global_best})
else:
    mat = io.loadmat('RCGA_plot.mat')
    global_best=mat['weight'][0]
    plot_g_m=mat['g_m'][0]
    plot_g_v=mat['g_v'][0]



plt.plot(np.array(plot_g_m)-np.array(plot_g_v))
plt.show()


w=model.get_weights()
model.set_weights(weight_reshape(global_best,w))
[m,v]=obj_eval(train_set,model)
print(m)
train_feature, train_label = get_feature(train_set,model)
test_feature, test_label = get_feature(test_set,model)



def precision(label, confusion_matrix):
    tp = confusion_matrix[label, label];
    fn = np.sum(confusion_matrix[label, :]) - tp;
    fp = np.sum(confusion_matrix[:, label]) - tp;
    tn = np.sum(confusion_matrix) - (tp+fp+fn)
    return (tp)/(tp+fp)

def recall(label, confusion_matrix):
    tp = confusion_matrix[label, label];
    fn = np.sum(confusion_matrix[label, :]) - tp;
    fp = np.sum(confusion_matrix[:, label]) - tp;
    tn = np.sum(confusion_matrix) - (tp+fp+fn)
    return (tp)/(tp+fn)

def accuracy_f(label, confusion_matrix):
    tp = confusion_matrix[label, label];
    fn = np.sum(confusion_matrix[label, :]) - tp;
    fp = np.sum(confusion_matrix[:, label]) - tp;
    tn = np.sum(confusion_matrix) - (tp+fp+fn)
    return (tp+tn)/(tp+tn+fp+fn)
def tp_f(label, confusion_matrix):
    tp = confusion_matrix[label, label];
    fn = np.sum(confusion_matrix[label, :]) - tp;
    fp = np.sum(confusion_matrix[:, label]) - tp;
    tn = np.sum(confusion_matrix) - (tp+fp+fn)
    return tp,fn,fp,tn

print('-----------------------------------------------------------------------------------------------------')
print('-------------------------------------------K-NN Results ---------------------------------------------')
print('-----------------------------------------------------------------------------------------------------')

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import confusion_matrix, classification_report
cls=knn(n_neighbors=3)
cls.fit(train_feature, train_label.ravel())
p = cls.predict(test_feature)
no_cl = np.unique(train_label).shape[0]
conf = confusion_matrix(test_label.ravel(),p)
print('confusion matrix')
print('________________\n')
print(conf)
print('_______________________________________________________\n')
print(classification_report(test_label.ravel(),p))
print('_______________________________________________________\n')
print("label     tp     fn     fp     tn   accuracy   precision   recall")
for label in range(no_cl):
    tp,fn,fp,tn = tp_f(label, conf.astype(float))
    print(f"{label+1:5d} {tp:6.0f} {fn:6.0f} {fp:6.0f} {tn:6.0f}   {accuracy_f(label, conf):8.3f}   {precision(label, conf):9.3f}   {recall(label, conf):6.3f}")

print('_______________________________________________________\n')
acc=np.sum(np.diagonal(conf))/np.sum(conf)
print('Accuracy : {}'.format(acc))

print('-----------------------------------------------------------------------------------------------------')
print('-------------------------------------------ELM  Results ---------------------------------------------')
print('-----------------------------------------------------------------------------------------------------')

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier as ELM_clf
from sklearn_extensions.extreme_learning_machines.random_layer import MLPRandomLayer as rand_l
cls=ELM_clf(hidden_layer=rand_l(n_hidden=400, activation_func='sigmoid'))
cls.fit(train_feature, train_label.ravel())
p = cls.predict(test_feature)
no_cl = np.unique(train_label).shape[0]
conf = confusion_matrix(test_label.ravel(),p)
print('confusion matrix')
print('________________\n')
print(conf)
print('_______________________________________________________\n')
print(classification_report(test_label.ravel(),p))
print('_______________________________________________________\n')
print("label     tp     fn     fp     tn   accuracy   precision   recall")
for label in range(no_cl):
    tp,fn,fp,tn = tp_f(label, conf.astype(float))
    print(f"{label+1:5d} {tp:6.0f} {fn:6.0f} {fp:6.0f} {tn:6.0f}   {accuracy_f(label, conf):8.3f}   {precision(label, conf):9.3f}   {recall(label, conf):6.3f}")

print('_______________________________________________________\n')
acc=np.sum(np.diagonal(conf))/np.sum(conf)
print('Accuracy : {}'.format(acc))


print('-----------------------------------------------------------------------------------------------------')
print('-------------------------------------------K-ELM Results --------------------------------------------')
print('-----------------------------------------------------------------------------------------------------')

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier as ELM_clf
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer as rand_l
cls=ELM_clf(hidden_layer=rand_l(n_hidden=300, activation_func='gaussian',rbf_width=0.01))
cls.fit(train_feature, train_label.ravel())
p = cls.predict(test_feature)
no_cl = np.unique(train_label).shape[0]
conf = confusion_matrix(test_label.ravel(),p)
print('confusion matrix')
print('________________\n')
print(conf)
print('_______________________________________________________\n')
print(classification_report(test_label.ravel(),p))
print('_______________________________________________________\n')
print("label     tp     fn     fp     tn   accuracy   precision   recall")
for label in range(no_cl):
    tp,fn,fp,tn = tp_f(label, conf.astype(float))
    print(f"{label+1:5d} {tp:6.0f} {fn:6.0f} {fp:6.0f} {tn:6.0f}   {accuracy_f(label, conf):8.3f}   {precision(label, conf):9.3f}   {recall(label, conf):6.3f}")

print('_______________________________________________________\n')
acc=np.sum(np.diagonal(conf))/np.sum(conf)
print('Accuracy : {}'.format(acc))


print('-----------------------------------------------------------------------------------------------------')
print('-------------------------------------------SVM   Results --------------------------------------------')
print('-----------------------------------------------------------------------------------------------------')

from sklearn.svm import SVC
cls=SVC(C=10000,gamma=10,kernel='rbf',random_state=None,tol=0.000001,coef0=0,degree=2,decision_function_shape='ovr')
cls.fit(train_feature, train_label.ravel())
p = cls.predict(test_feature)
no_cl = np.unique(train_label).shape[0]
conf = confusion_matrix(test_label.ravel(),p)
print('confusion matrix')
print('________________\n')
print(conf)
print('_______________________________________________________\n')
print(classification_report(test_label.ravel(),p))
print('_______________________________________________________\n')
print("label     tp     fn     fp     tn   accuracy   precision   recall")
for label in range(no_cl):
    tp,fn,fp,tn = tp_f(label, conf.astype(float))
    print(f"{label+1:5d} {tp:6.0f} {fn:6.0f} {fp:6.0f} {tn:6.0f}   {accuracy_f(label, conf):8.3f}   {precision(label, conf):9.3f}   {recall(label, conf):6.3f}")

print('_______________________________________________________\n')
acc=np.sum(np.diagonal(conf))/np.sum(conf)
print('Accuracy : {}'.format(acc))

print('-----------------------------------------------------------------------------------------------------')
print('------------------------------------------- MLP   Results --------------------------------------------')
print('-----------------------------------------------------------------------------------------------------')

from sklearn.neural_network import MLPClassifier as MLP
cls=MLP(solver='sgd', alpha=0.000003, hidden_layer_sizes=(400), tol=0.0000001, activation='tanh', max_iter=5000)
cls.fit(train_feature, train_label.ravel())
p = cls.predict(test_feature)
no_cl = np.unique(train_label).shape[0]
conf = confusion_matrix(test_label.ravel(),p)
print('confusion matrix')
print('________________\n')
print(conf)
print('_______________________________________________________\n')
print(classification_report(test_label.ravel(),p))
print('_______________________________________________________\n')
print("label     tp     fn     fp     tn   accuracy   precision   recall")
for label in range(no_cl):
    tp,fn,fp,tn = tp_f(label, conf.astype(float))
    print(f"{label+1:5d} {tp:6.0f} {fn:6.0f} {fp:6.0f} {tn:6.0f}   {accuracy_f(label, conf):8.3f}   {precision(label, conf):9.3f}   {recall(label, conf):6.3f}")

print('_______________________________________________________\n')
acc=np.sum(np.diagonal(conf))/np.sum(conf)
print('Accuracy : {}'.format(acc))
