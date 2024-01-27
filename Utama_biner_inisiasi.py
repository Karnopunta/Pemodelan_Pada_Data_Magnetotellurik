import numpy as np
from Fungsi_versi_biner import Inisiasi,Milih,KawinSilang,Seleksi,Mutasi,Conv_desm
import matplotlib.pyplot as plt
from fwd_MT import f_MT

#-------memasukkan model dan memanggil fungsi/def Inisiasi
npopulasi=50
n_lapis =7
n_bit = 13
rho_max = 90.0
rho_min = 0.0
thick_max = 5000.0
thick_min = 0.0
jlh_bit   = n_bit*((2*n_lapis)-1)
ind_biner = np.zeros([npopulasi,jlh_bit]) 

[ind_biner,ind_desimal] = Inisiasi(npopulasi,n_lapis,n_bit,rho_min,rho_max,thick_min,thick_max,jlh_bit)
[n_pop,n_param] = np.shape(ind_desimal)



#-------memanggil data observasi
d_obs          = np.loadtxt('Data.txt') 
[ndat,nkol]    = np.shape(d_obs)
freq           = d_obs[:,0]
rho_dat        = d_obs[:,1]

#------menyiapkan parameter perhitungan
max_iter     = 100
silang_rate  = 0.8
p_mutasi     = 0.3
#-----Inisiasi parameter perhitungan

RMS_hasil    = np.zeros(max_iter)
iterasi_arr  = np.zeros(max_iter)
m_min        = np.array([rho_min,thick_min]) 
m_max        = np.array([rho_max,thick_max])

Rho_cal = [ ]
phase_cal= [ ]
ind_terbaik=[ ]

for iterasi in range(0,max_iter):
    #-----Pemilihan individu berdasarkan probabilitas kumulatif menggunakan konsep Roulette Wheel
    [id_induk1,id_induk2,jlh_kawin]= Milih(ndat,freq,rho_dat,npopulasi,ind_desimal,silang_rate,n_lapis) #checked
    #-----Proses Cross Over
    ind_biner = KawinSilang(jlh_bit,ind_biner,id_induk1,id_induk2,jlh_kawin)
    m,n = ind_biner.shape
    ind_desimal = Conv_desm(m,n_lapis,n_bit,ind_biner,m_min,m_max)
    #-----Proses seleksi agar jumlahnya sesuai dengan jumlah populasi di generasi awal
    Id_terpilih,d_rms = Seleksi(ndat,freq,rho_dat,m,ind_desimal,n_lapis)
    Ind_terpilih = np.zeros([m,jlh_bit])
    for i in range(0,npopulasi):
        Ind_terpilih[i,:] = ind_biner[int(Id_terpilih[i,0]),:]
    ind_biner = Mutasi(Ind_terpilih,npopulasi,p_mutasi,jlh_bit)
    ind_biner = np.delete(ind_biner, np.s_[npopulasi:m],0)
    ind_desimal = Conv_desm(npopulasi,n_lapis,n_bit,ind_biner,m_min,m_max)
    #m = Rank_indi[Rank_indi[:, 1].argsort()[::1]]
    RMS_hasil[iterasi] = d_rms
    #print(RMS_hasil)
    #print(Id_terpilih)
    iterasi_arr[iterasi] = iterasi 
    #individu_terbaik =ind_desimal[int(Id_terpilih[0,0]),:]
    
#print(ind_desimal)
#print(Id_terpilih)
individu_terbaik = ind_desimal[int(Id_terpilih[0,0]),:]
#print(freq)
Rho_app_calc,Phase_calc = f_MT(freq,individu_terbaik,n_lapis)
#print(Rho_app_calc)

#print(individu_terbaik)
#print(ind_desimal)

Rho_cal = Rho_app_calc
phase_cal= Phase_calc
ind_terbaik=individu_terbaik

print('nlapis=7')


print(ind_desimal)
#    -------------------tampilkan hasil-----------------
print('rho calculasi')
print(Rho_cal)
#
print('phase')
print(phase_cal)

print('individu terbaik')
print(individu_terbaik)
#
plt.plot(np.log10(freq),Rho_app_calc,'-',color='r')
plt.plot(np.log10(freq),rho_dat,'p',color='b')
#mt.title('data observasi')
plt.xlabel('Frequensi(log 10)')
plt.ylabel('Rho (Ohm.m)')
plt.show()

plt.plot(np.log10(freq),Phase_calc)
plt.title('PHASA HASIL INVERSI')
plt.xlabel('frequensi')
plt.ylabel('phase')
plt.show()

plt.plot(iterasi_arr,RMS_hasil)
plt.title('GRAFIK RMS ERROR INVERSI')
plt.xlabel('Generasi')
plt.ylabel('RMS ERROR')
plt.show()