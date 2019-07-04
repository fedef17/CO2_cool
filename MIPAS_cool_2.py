#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import math as mt
import spect_base_module as sbm
from subprocess import call
import pickle
import scipy.io as io
from scipy.interpolate import PchipInterpolator as spline


cart = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/MIPAS_2009/'

cmaps = ['spectral','jet','gist_ncar','gist_rainbow','hsv','nipy_spectral']
cmap = cm.get_cmap(cmaps[5])

restot = pickle.load(file=open(cart+'ssw2009_v3_okTOCO2.pyc','r'))

l1 = 75
l2 = 90
szl = 100.0

res_np = restot[(restot.latitude >= l1) & (restot.latitude <= l2) & (restot.sza < szl)]

szamin = np.min(res_np.sza)-2
szamax = np.max(res_np.sza)+2
#res_np = restot

diffs=res_np.cr_fomi_int[0]
temps=res_np.cr_fomi_int[0]
cr_mipas=res_np.cr_fomi_int[0]
cr_fomi=res_np.cr_fomi_int[0]

for alt,cr1,cr2,temp in zip(res_np.altitude,res_np.cr_mipas,res_np.cr_fomi_int,res_np.temperature):
    #pl.plot(alt,cr1,color='black')
    #pl.plot(alt,-cr2,color='red')
    cdiff = cr1 + cr2
    print(len(cdiff))
    diffs = np.vstack([diffs,cdiff])
    temps = np.vstack([temps,temp])
    cr_mipas = np.vstack([cr_mipas,cr1])
    cr_fomi = np.vstack([cr_fomi,-cr2])

#pl.show()

diffs = diffs[1:,:]
temps = temps[1:,:]
cr_mipas = cr_mipas[1:,:]
cr_fomi = cr_fomi[1:,:]

print(np.shape(diffs))

diff_day = np.mean(diffs, axis=0)
temp_day = np.mean(temps, axis=0)
cr_mipas_day = np.mean(cr_mipas, axis=0)
cr_fomi_day = np.mean(cr_fomi, axis=0)


res_np = restot[(restot.latitude >= l1) & (restot.latitude <= l2) & (restot.sza > szl)]


szamin = np.min(res_np.sza)-2
szamax = np.max(res_np.sza)+2
#res_np = restot

diffs=res_np.cr_fomi_int[0]
temps=res_np.cr_fomi_int[0]
cr_mipas=res_np.cr_fomi_int[0]
cr_fomi=res_np.cr_fomi_int[0]

for alt,cr1,cr2,temp in zip(res_np.altitude,res_np.cr_mipas,res_np.cr_fomi_int,res_np.temperature):
    #pl.plot(alt,cr1,color='black')
    #pl.plot(alt,-cr2,color='red')
    cdiff = cr1 + cr2
    print(len(cdiff))
    diffs = np.vstack([diffs,cdiff])
    temps = np.vstack([temps,temp])
    cr_mipas = np.vstack([cr_mipas,cr1])
    cr_fomi = np.vstack([cr_fomi,-cr2])

#pl.show()

diffs = diffs[1:,:]
temps = temps[1:,:]
cr_mipas = cr_mipas[1:,:]
cr_fomi = cr_fomi[1:,:]

print(np.shape(diffs))

diff_night = np.mean(diffs, axis=0)
temp_night = np.mean(temps, axis=0)
cr_mipas_night = np.mean(cr_mipas, axis=0)
cr_fomi_night = np.mean(cr_fomi, axis=0)


# Ed ora grafico
res_np = restot[(restot.latitude >= l1) & (restot.latitude <= l2)]

diffs=res_np.cr_fomi_int[0]
for alt,cr1,cr2,temp in zip(res_np.altitude,res_np.cr_mipas,res_np.cr_fomi_int,res_np.temperature):
    #pl.plot(alt,cr1,color='black')
    #pl.plot(alt,-cr2,color='red')
    cdiff = cr1 + cr2
    print(len(cdiff))
    diffs = np.vstack([diffs,cdiff])
diffs = diffs[1:,:]

fig1 = pl.figure(figsize=(8, 6), dpi=150)
ax1 = pl.subplot(111)
pl.xlabel('Cooling rate difference Mipas-Fomichev (K/day)')
pl.ylabel('Altitude (km)')
fig2 = pl.figure(figsize=(8, 6), dpi=150)
ax2 = pl.subplot(111)
pl.xlabel('Temperature (K)')
pl.ylabel('Altitude (km)')
fig3 = pl.figure(figsize=(8, 6), dpi=150)
ax3 = pl.subplot(111)
pl.xlabel('Cooling rate Mipas (K/day)')
pl.ylabel('Altitude (km)')
fig4 = pl.figure(figsize=(8, 6), dpi=150)
ax4 = pl.subplot(111)
pl.xlabel('Cooling rate Fomichev (K/day)')
pl.ylabel('Altitude (km)')

for di,temp,cr1,cr2,sza in zip(diffs,res_np.temperature,res_np.cr_mipas,res_np.cr_fomi_int,res_np.sza):
    #colo = cmap(1-(sza-szamin)/(szamax-szamin))
    if(sza > szl):
         colo = 'grey'
    else:
         colo = 'blue'
    ax1.plot(di,alt,color=colo)
    ax2.plot(temp,alt,color=colo)
    ax3.plot(cr1,alt,color=colo)
    ax4.plot(-cr2,alt,color=colo)

ax1.plot(diff_night,alt,color = 'black',linewidth=2.0,label='night')
#ax1.scatter(diff_night,alt, color = 'black')
ax1.plot(diff_day,alt,color = 'red',linewidth=2.0,label='day')
#ax1.scatter(diff_day,alt, color = 'red')

ax2.plot(temp_night,alt,color = 'black',linewidth=2.0,label='night')
ax2.plot(temp_day,alt,color = 'red',linewidth=2.0,label='day')

ax3.plot(cr_mipas_night,alt,color = 'black',linewidth=2.0,label='night')
ax3.plot(cr_mipas_day,alt,color = 'red',linewidth=2.0,label='day')

ax4.plot(cr_fomi_night,alt,color = 'black',linewidth=2.0,label='night')
ax4.plot(cr_fomi_day,alt,color = 'red',linewidth=2.0,label='day')

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

ax1.set_ylim([30,70])
ax2.set_ylim([30,70])
ax3.set_ylim([30,70])
ax4.set_ylim([30,70])

#ax1.xlim(30,70)
#ax2.xlim(30,70)
#ax3.xlim(30,70)
#ax4.xlim(30,70)

#fig.savefig(cart+'Diff_northpole_120km.eps', format='eps', dpi=150)
#pl.show()
pl.close()
pl.close()
pl.close()
pl.close()


######
### voglio vedere cosa combina al variare del SZA: ####

fig1 = pl.figure(figsize=(8, 6), dpi=150)
sza = np.linspace(30,50,5)

n=0
for i in range(len(sza)-1):
    oks = res_np[(res_np.sza > sza[i]) & (res_np.sza < sza[i+1])]
    print(len(oks))
    if(len(oks) < 1): continue
    n+=1
    colo, io = sbm.findcol(5,n)
    difs = np.mean([cr1+cr2 for cr1,cr2 in zip(oks.cr_fomi_int, oks.cr_mipas)], axis = 0)
    pl.plot(difs,res_np.altitude[0],color=colo,label='{:5.2f}'.format(sza[i]))

pl.grid()
pl.legend(loc=3)
pl.xlabel('SZA dependence')
pl.ylabel('Altitude (km)')
#pl.show()
pl.close()

fig1 = pl.figure(figsize=(8, 6), dpi=150)

tempm = np.mean([ti for ti in res_np.temperature], axis = 0)
temp_var = [ti-tempm for ti in res_np.temperature]
print(type(temp_var))
for ti in temp_var:
    pl.plot(ti,res_np.altitude[0])
pl.xlabel('Temp. oscillation')
pl.ylabel('Altitude (km)')
#pl.show()
pl.close()

fig1 = pl.figure(figsize=(8, 6), dpi=150)
#difs = np.mean([(cr1+cr2)*ti for cr1,cr2,ti in zip(res_np.cr_fomi_int, res_np.cr_mipas, temp_var)], axis = 0)
difs = [(cr1+cr2) for cr1,cr2,ti in zip(res_np.cr_fomi_int, res_np.cr_mipas, temp_var)]
for di in difs:
    pl.plot(di,res_np.altitude[0])#,color=colo,label='{:5.2f}'.format(sza[i]))

#pl.xlabel('deltaCR*deltaT')
pl.xlim(-80,80)
pl.xlabel('Cooling rate difference Mipas - Fomichev (K/day)')
pl.ylabel('Altitude (km)')
pl.grid()
pl.legend()
#pl.show()
pl.close()


## Faccio un correlation plot tra differenza nei rates e nelle temperature nella regione 70 - 100

x = []
y = []
alts = res_np.altitude[0]
for cr1,cr2,ti in zip(res_np.cr_fomi_int, res_np.cr_mipas, temp_var):
    for i in range(len(alts)):
        if(alt[i] < 100.0 and alt[i] > 70.0):
            x2 = ti[i]
            y2 = cr1[i]+cr2[i]
            x = np.append(x,x2)
            y = np.append(y,y2)

#sbm.plotcorr(x,y,cart+'corr_Dx_DT.eps',xlabel = 'Temperature variation',ylabel='Cooling rate difference')


## Faccio un correlation plot tra differenza nei rates e nelle temperature nella regione 70 - 100
# Stavolta mediando sia i cooling rates che le temperature variations

x = []
y = []
alts = res_np.altitude[0]
cond = (alts > 70.0) & (alts < 100.0)
for cr1,cr2,ti in zip(res_np.cr_fomi_int, res_np.cr_mipas, temp_var):
    x2 = np.std([cra+crb for cra,crb in zip(cr1[cond],cr2[cond])])
    y2 = np.std(ti[cond])
    x = np.append(x,x2)
    y = np.append(y,y2)

## sbm.plotcorr(x,y,cart+'corr_Dx_DT_stdstd.eps',xlabel = 'Temperature variation (std dev)',ylabel='Cooling rate difference (std dev)')


i=-2
ok = (alts > 35) & (alts < 125)

fig = pl.figure(figsize=(8, 6), dpi=150)

ax1 = pl.subplot(1,2,1)
pl.ylabel('Altitude (km)')
pl.xlabel('Temperature (K)')
ax2 = pl.subplot(1,2,2)
pl.xlabel('Cooling rate (K/day)')

ax1.set_ylim([40,120])
ax2.set_ylim([40,120])

ax1.plot(res_np.temperature[i][ok],alts[ok])
colo, iis = sbm.findcol(3,0)
ax2.plot(-res_np.cr_fomi_int[i][ok],alts[ok],color=colo,label='Fomichev')
colo, iis = sbm.findcol(3,1)
ax2.plot(res_np.cr_mipas[i][ok],alts[ok],color=colo,label='Mipas')
colo, iis = sbm.findcol(3,2)
ax2.plot(res_np.cr_fomi_int[i][ok]+res_np.cr_mipas[i][ok],alts[ok],color=colo,label='diff')
ax1.grid()
ax2.grid()
pl.legend(loc=4,fontsize='small')
pl.show()
pl.close()
pl.close()
pl.close()
sys.exit()

#
# fig = pl.figure(figsize=(8, 6), dpi=150)
# for di,temp,cr1,sza in zip(diffs,res_np.temperature,res_np.cr_mipas,res_np.sza):
#     i40 = sbm.find_near(alt,30.0)
#     if(di[i40] > diff[i40]):
#         colo = 'grey'
#     else:
#         colo = 'blue'
#     colo = cmap(1-(sza-szamin)/(szamax-szamin))
#     pl.plot(temp,alt,color=colo)
# pl.grid()
# #pl.xlabel('Cooling rate Mipas (K/day)')
# pl.ylim(30,90)
# pl.xlabel('Temperature (K)')
# pl.ylabel('Altitude (km)')
# pl.show()
#
# i=0
# for di,temp,sza in zip(diffs,res_np.temperature,res_np.sza):
#     i40 = sbm.find_near(alt,30.0)
#     if(di[i40] > diff[i40]):
#         colo = 'grey'
#     else:
#         colo = 'blue'
#     colo = cmap(1-(sza-szamin)/(szamax-szamin))
#     pl.scatter(i,sza,color=colo)
#     i+=1
# pl.xlabel('Time')
# pl.ylabel('SZA')
# pl.show()
#
#
# fig = pl.figure(figsize=(8, 6), dpi=150)
# for di,temp,cr1,sza in zip(diffs,res_np.temperature,res_np.cr_mipas,res_np.sza):
#     i40 = sbm.find_near(alt,30.0)
#     if(di[i40] > diff[i40]):
#         colo = 'grey'
#     else:
#         colo = 'blue'
#     colo = cmap(1-(sza-szamin)/(szamax-szamin))
#     pl.plot(cr1,alt,color=colo)
# pl.grid()
# pl.xlabel('Cooling rate Mipas (K/day)')
# pl.ylabel('Altitude (km)')
# pl.ylim(30,70)
# pl.xlim(-2,20)
# pl.show()
#
#
# fig = pl.figure(figsize=(8, 6), dpi=150)
# for di,temp,cr2,sza in zip(diffs,res_np.temperature,res_np.cr_fomi_int,res_np.sza):
#     i40 = sbm.find_near(alt,30.0)
#     if(di[i40] > diff[i40]):
#         colo = 'grey'
#     else:
#         colo = 'blue'
#     colo = cmap(1-(sza-szamin)/(szamax-szamin))
#     pl.plot(-cr2,alt,color=colo)
# pl.grid()
# pl.xlabel('Cooling rate Fomichev (K/day)')
# pl.ylabel('Altitude (km)')
# pl.ylim(30,70)
# pl.xlim(-2,20)
# pl.show()

#1 -> differenza day/night


#2 -> atmosfera polare: plot differenze giorno e notte (ssw)


#3 -> atmosfera non polare: plot differenze giorno e notte (gw) -> correlazione differenze/temperatura?
