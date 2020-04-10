# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:31:07 2020

@author: tarik
"""


# =============================================================================
# Gerekli kütüphaneleri içe aktaralım
# =============================================================================
import torch.nn as nn
import torch
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# =============================================================================
# DATA SETİMİZİ ÇEKELİM
# =============================================================================
x_train ,y_train =load_breast_cancer(return_X_y=True)

# =============================================================================
# verilerimizi numpy arrayından pytorch tensoruna çevirelim
# =============================================================================

x_train=torch.from_numpy(x_train).float()

y_train=torch.from_numpy(y_train).long()


# =============================================================================
# Hiper parametreli atayalım
# =============================================================================


input_size=30            #giriş boyutunu atadık
hidden_size=128          #gizli katman boyutu
output_size=2            #çıkış boyutumuzu atadık
learning_rate=0.001      #öğrenmek katsayımızı belirledik
EPOCHS= 200              #tekrarlama sayımızı atadık


_loss=[]                #grafik için list oluşturduk


# =============================================================================
# Pytorch Class modulumuzu oluşturuyoruz
# =============================================================================

class Net(nn.Module):
    def __init__(self,input_size,output_size):
        super(Net,self).__init__()    
        self.lineer1=nn.Linear(input_size,hidden_size)   #1.Lineer katmanımız oluşturduk
        self.lineer2=nn.Linear(hidden_size,output_size)  #2.Lineer katmanımız oluşturduk
        self.relu=nn.ReLU()                              #RELU aktivasyon fonksiyonumuz seçtik
    def forward(self,x):
        x=self.lineer1(x)
        x=self.relu(x)
        x=self.lineer2(x)
        return x
net=Net(input_size,output_size)

los=nn.CrossEntropyLoss()                               #loss olarak CrossEntropyLoss
optim=torch.optim.Adam(net.parameters(),lr=learning_rate)  #optimizer olarak Adam learning_rate=0.001 yaptık

for epoch in range(EPOCHS):                               #epohcs degerimiz kadar  tüm veriyi birden eğitime verdik
        output=net(x_train)                               
        loss=los(output,y_train)                         #çıkan sonuçlar ile  gerçek sonuçları karşılaştırıp hata değeri elde ettik
        optim.zero_grad()                                
        loss.backward()                                  
        optim.step()
        _loss.append(loss.item())                       #grafik için list loss değerlerini ekledik
        print ('Epoch [{}/{}] , Loss {:.4f}'.format(epoch+1,EPOCHS,loss.item()))


plt.plot(np.array(_loss))                               #plot yardımıyla loss değerlerini çizdirdik






