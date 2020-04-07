# -*- coding: utf-8 -*-


# =============================================================================
# İlk önce gerekli kütüphaneleri içe aktaralım
# =============================================================================
import torch
# =============================================================================
#  y=5*x+2 gibi bir denklem denklem tanımlıyalım
# =============================================================================
x=torch.tensor(1.0,requires_grad=True)

w=torch.tensor(5.0,requires_grad=True) #ağırlık değeri=5.0

b=torch.tensor(2.0,requires_grad=True) #bias değeri=2.0


y=w*x+b            # denklemi kurduk 

y.backward()       # gradient hesaplıyalım

print (x.grad)  # 5

print (w.grad)  # 1

print (b.grad)  # 1









