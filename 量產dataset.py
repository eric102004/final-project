#!/usr/bin/env python
# coding: utf-8

# 需要製作的函式:
# 
# **建立array**
# **輸入資料數量(ndata)**
# 
# 以下為主函式內容------------------------datamanufacture(ndata)
# -----------------------------
# 
# 1.4*3個觀測角XYZ權重---------------------------weight()
# 
# 2.4個觀測角------------------------------------four_angle(wa0,wa0p,wb0,wb0p)
# 
# 
# **for迴圈**
# 
# 1.產生密度矩陣、theta、p、pi值，return 這四個值--density_matrix()
# 
# 2.用ppt求正解----------------------------------ppt(dmatrix)
# 
# 3.4個feature----------------------------------four_feature(dmatrix,a0,a0p,b0,b0p)
# 
# 4.寫入陣列-------------------------------------write(i,f1,f2,f3,f4,label,p,theta,phi)
# 
# **for 迴圈結束**
# 
# 1.陣列輸出excel--------------------------------save(F1,F2,F3,F4,Label,P,Theta,Phi)
# 
# 2.觀測角權重寫入txt-----------------------------wnote(wa0,wa0p,wb0,wb0p)
# 
# 主函式結束:回傳complete
# ------------------------------

# In[18]:


def weight():    #建立觀測角的包立矩陣權重陣列、並輸入觀測角的包立矩陣權重
    wa0 = []
    wa0p = []
    wb0 = []
    wb0p = []
    dic = ['X','Y','Z']
    while 1:
        print('\nInput the weight of sigmaX、Y、Z for a0 respectively.\nThe sum of the three weight should equal to 1')
        for i in range(3):
            wa0.append(float(input('input the weight of sigma'+dic[i]+' for measure angle a0.')))
        if abs(wa0[0]**2+wa0[1]**2+wa0[2]**2-1)<=0.00001:
            break
        else:
            print('Error! NOT EQUAL TO 1')
    while 1:
        print('\nInput the weight of sigmaX、Y、Z for a0p respectively.\nThe sum of the three weight should equal to 1')
        for j in range(3):
            wa0p.append(float(input('input the weight of sigma'+dic[j]+' for measure angle a0p.')))
        if abs(wa0p[0]**2+wa0p[1]**2+wa0p[2]**2-1)<=0.00001:
            break
        else:
            print('Error! NOT EQUAL TO 1')
    while 1:
        print('\nInput the weight of sigmaX、Y、Z for b0 respectively.\nThe sum of the three weight should equal to 1')
        for k in range(3):
            wb0.append(float(input('input the weight of sigma'+dic[k]+' for measure angle b0.')))
        if abs(wb0[0]**2+wb0[1]**2+wb0[2]**2-1)<=0.00001:
            break
        else:
            print('Error! NOT EQUAL TO 1')
    while 1:
        print('\nInput the weight of sigmaX、Y、Z for b0p respectively.\nThe sum of the three weight should equal to 1')
        for l in range(3):
            wb0p.append(float(input('input the weight of sigma'+dic[l]+' for measure angle b0p.')))
        if abs(wb0p[0]**2+wb0p[1]**2+wb0p[2]**2-1)<=0.00001:
            break
        else:
            print('Error! NOT EQUAL TO 1')    
    return (wa0,wa0p,wb0,wb0p)


# In[19]:


def four_angle(wa0,wa0p,wb0,wb0p):
    #產生a0,a0p,b0,b0p
    a0 = wa0[0]*q.sigmax()+wa0[1]*q.sigmay()+wa0[2]*q.sigmaz()
    a0p = wa0p[0]*q.sigmax()+wa0p[1]*q.sigmay()+wa0p[2]*q.sigmaz()
    b0 = wb0[0]*q.sigmax()+wb0[1]*q.sigmay()+wb0[2]*q.sigmaz()
    b0p = wb0p[0]*q.sigmax()+wb0p[1]*q.sigmay()+wb0p[2]*q.sigmaz()
    #兩兩相乘得到四個"參數"(註:參數的dims要轉成[4,4])(用Kronecker product)
    a0b0 = q.Qobj(np.kron(a0.full(),b0.full()).reshape(4,4))
    a0b0p = q.Qobj(np.kron(a0.full(),b0p.full()).reshape(4,4))
    a0pb0 = q.Qobj(np.kron(a0p.full(),b0.full()).reshape(4,4))
    a0pb0p = q.Qobj(np.kron(a0p.full(),b0p.full()).reshape(4,4))
    return (a0b0,a0b0p,a0pb0,a0pb0p)


# In[20]:


def density_matrix():
    p = np.random.uniform(0, 1)
    theta = np.random.uniform(0,np.pi)
    phi = np.random.uniform(0,2*np.pi)
    phase = np.exp(1j*phi)
    sin = np.sin(theta/2)
    cos = np.cos(theta/2)
    ket = q.Qobj(np.array([cos,0,0,phase*sin]).reshape(4,1))
    doper = ket*ket.dag()
    dmatrix = p*doper + ((1-p)/4)*np.eye(4)
    return [dmatrix,p,theta,phi]


# In[21]:


def ppt(dmatrix):   #entangled:1 ; separable:0
    dmx = q.Qobj(dmatrix,dims=[[2, 2], [2, 2]])
    ptdmx = q.partial_transpose(dmx,[0,1])
    eigenvalues = ptdmx.eigenenergies()
    for l in eigenvalues:
        if l < 0:
            return 1
    return 0


# In[22]:


def four_feature(dmatrix,a0b0,a0b0p,a0pb0,a0pb0p):
    f1m = dmatrix*a0b0
    f2m = dmatrix*a0b0p
    f3m = dmatrix*a0pb0
    f4m = dmatrix*a0pb0p
    f1 = f1m.tr()
    f2 = f2m.tr()
    f3 = f3m.tr()
    f4 = f4m.tr()
    #註:過濾極小誤差+檢測
    f1 = round(f1.real,12)+round(f1.imag,12)*1j
    f2 = round(f2.real,12)+round(f2.imag,12)*1j
    f3 = round(f3.real,12)+round(f3.imag,12)*1j
    f4 = round(f4.real,12)+round(f4.imag,12)*1j
    if np.imag(f1) ==0 and np.imag(f2) ==0 and np.imag(f3) ==0 and np.imag(f4) ==0 :
        f1 = np.real(f1)
        f2 = np.real(f2)
        f3 = np.real(f3)
        f4 = np.real(f4)
        return (f1,f2,f3,f4)
    else:
        return "complex Feature!"


# In[23]:


def write(f1,f2,f3,f4,label,p,theta,phi):
    global F1,F2,F3,F4,Label,P,Theta,Phi
    F1.append(f1)
    F2.append(f2)
    F3.append(f3)
    F4.append(f4)
    Label.append(label)
    P.append(p)
    Theta.append(theta)
    Phi.append(phi)


# In[24]:


def save(F1,F2,F3,F4,Label,P,Theta,Phi):
    Feature = pd.DataFrame({'F1':F1,'F2':F2,'F3':F3,'F4':F4})
    LABEL = pd.DataFrame({'Label':Label})
    Reference = pd.DataFrame({'F1':F1,'F2':F2,'F3':F3,'F4':F4,'Label':Label,'P':P,'Theta':Theta,'Phi':Phi})
    #將表寫入excel檔儲存
    Feature.to_excel("E:\個別研究\程式\dataset\Feature\Feature_train.xlsx")#自訂檔案路徑+名稱
    LABEL.to_excel("E:\個別研究\程式\dataset\LABEL\LABEL_train.xlsx")
    Reference.to_excel("E:\個別研究\程式\dataset\Reference\Reference_train.xlsx")


# In[25]:


def wnote(wa0,wa0p,wb0,wb0p):
    f = open('E:\個別研究\程式\dataset\Reference\wnote.txt','w')
    f.write(str(wa0)+'\n'+str(wa0p)+'\n'+str(wb0)+'\n'+str(wb0p))
    f.close()


# In[26]:


def datamanufacture(ndata):
    weightlist = weight()
    anglelist =  four_angle(*weightlist)
    for i in range(ndata):
        dmx_info = density_matrix()
        label = ppt(dmx_info[0])
        featurelist = four_feature(dmx_info[0],*anglelist)
        write(*featurelist,label,dmx_info[1],dmx_info[2],dmx_info[3])
    save(F1,F2,F3,F4,Label,P,Theta,Phi)
    wnote(*weightlist)
    return 'complete'


# In[10]:


#運作區
if __name__ == '__main__':
    import numpy as np
    import qutip as q
    import pandas as pd
    #建立array
    F1 = []
    F2 = []
    F3 = []
    F4 = []
    Label = []
    P = []
    Theta = []
    Phi = []
    ndata = int(input('input the number of data'))
    datamanufacture(ndata)


# In[ ]:




