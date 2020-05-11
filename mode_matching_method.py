import green_function as gr
import numpy as np
import scipy as sp
import function as fu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

pos = np.array([0,0,0])


#初期条件
freq = 550
cv = 344
num_sp = 64
lambda_ = 10**-8
r_sp = 1.5

#グラフ条件
Plot_range = 2.5
space = 0.05

#計算
omega = 2 * np.pi * freq
k = omega/cv
pos_azi, pos_ele, pos_r = fu.cart2sph(pos[0],pos[1],pos[2])
vec_R = np.array([pos_azi, pos_ele, pos_r])

#スピーカー位置
sp_xyz = fu.EquiDistSphere(num_sp)
sp_xyz = sp_xyz * r_sp
sp_azi,sp_ele, sp_r = fu.cart2sph(sp_xyz[:,0],sp_xyz[:,1],sp_xyz[:,2])
sp_ele1 = np.pi/2 - sp_ele
sp_sph = np.vstack([np.vstack([sp_azi,sp_ele1]),sp_r]).T

#最適な次数決定 
N_q = int(np.floor(np.sqrt(num_sp))-1)

#一次音源
ps_xyz = np.array([1.5,1.5,0]) # スピーカーを配置してる半径をよりも外に配置
ps_azi,ps_ele, ps_r = fu.cart2sph(ps_xyz[0],ps_xyz[1],ps_xyz[2])
ps_ele1 = np.pi/2 - ps_ele
ps_sph = np.vstack([np.vstack([ps_azi,ps_ele1]),ps_r]).T

#モードマッチング法の計算
C = fu.calcCmn(N_q,k,r_sp,sp_sph);     
b = fu.calcCmn(N_q,k,ps_sph[0,2],ps_sph);     

I = np.identity(num_sp)
CC = np.conj(C.T) @ C
Cb = np.conj(C.T) @ b

d_MM = 	np.linalg.solve(CC + lambda_ * I,Cb)

#所望音場(一次音源)と再現音場を求める
X = np.arange(-Plot_range,Plot_range+space,space)
Y = np.arange(-Plot_range,Plot_range+space,space)
P_org = np.zeros((X.size,Y.size), dtype = np.complex)
MM = np.zeros((X.size,Y.size), dtype = np.complex)
for j in range(X.size):
    for i in range(Y.size):
        pos_r = np.array([X[j],Y[i],0])
        P_org[i,j] = gr.green2(1,freq,cv,ps_xyz,pos_r)
        for l in range(num_sp):
            G_transfer = gr.green2(1,freq,cv,sp_xyz[l,:],pos_r)
            MM[i,j] += G_transfer *  d_MM[l]

NE = np.zeros((X.size,Y.size), dtype = np.complex)
for i in range(X.size):
    for j in range(Y.size):
        NE[i,j] = 10*np.log10(((np.abs(MM[i,j]-P_org[i,j]))**2)/((np.abs(P_org[i,j]))**2))

#スイートスポット
rad = (N_q*cv)/(2*np.pi*freq)

#グラフ
#所望音場
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False)
r_sp_ = patches.Circle(xy=(0, 0), radius=r_sp, ec='b',fill=False)
plt.figure()
ax = plt.axes()
im = ax.imshow(np.real(P_org), interpolation='gaussian',cmap=cm.jet,origin='lower', 
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=0.1, vmin=-0.1)
ax.grid(False)
ax.add_patch(sweet_spot)
ax.add_patch(r_sp_)
plt.axis('scaled')
ax.set_aspect('equal')
plt.colorbar(im)
plt.savefig('original_sound.pdf',bbox_inches="tight",dpi = 64,
            facecolor = "lightgray", tight_layout = True) 
plt.show()

#再現音場
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False)
r_sp_ = patches.Circle(xy=(0, 0), radius=r_sp, ec='b',fill=False)
plt.figure()
ax2 = plt.axes()
im2 = ax2.imshow(np.real(MM), interpolation='gaussian',cmap=cm.jet,origin='lower', 
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=0.1, vmin=-0.1)
ax2.grid(False)
ax2.add_patch(sweet_spot)
ax2.add_patch(r_sp_)
plt.axis('scaled')
ax2.set_aspect('equal')
plt.colorbar(im2)
plt.savefig('reproduced_sound.pdf',bbox_inches="tight",dpi = 64,
            facecolor = "lightgray", tight_layout = True)
plt.show()

#正規化誤差
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False)
r_sp_ = patches.Circle(xy=(0, 0), radius=r_sp, ec='b',fill=False)
plt.figure()
ax3 = plt.axes()
im3 = ax3.imshow(np.real(NE), interpolation='gaussian',cmap=cm.pink_r,origin='lower', 
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=0, vmin=-60)
ax3.grid(False)
ax3.add_patch(sweet_spot)
ax3.add_patch(r_sp_)
plt.axis('scaled')
ax3.set_aspect('equal')
plt.colorbar(im3)
plt.savefig('normalization_error.pdf',bbox_inches="tight",dpi = 64,
            facecolor = "lightgray", tight_layout = True)
plt.show()
