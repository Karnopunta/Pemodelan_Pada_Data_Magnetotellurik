import numpy as np
from fwd_MT import f_MT


# -------------(1)---------------------------------------------------(Komplit)#
def Inisiasi(n,n_lapis,n_bit,rho_min,rho_max,thick_min,thick_max,jlh_bit):
    #------Membangkitkan populasi beranggotakan n individu
    #------setiap individu diwakili oleh (n_bit x (2 x n_lapis-1)) buah bit biner
    #------jlh bit desimal per individu = n_bit x jumlah parameter model
    #------menyiapkan bit untuk setiap individu
    #------susunan parameter model : rho 1,...rho n, thick 1 ...thick n
    individu_biner   = np.zeros([n,jlh_bit])       # matriks ukuran n x (jlh bit x jlh parameter model) 
    individu_desimal = np.zeros([n,(2*n_lapis)-1]) # matriks ukuran n x (jlh parameter model)
    #---jika bilangan random di setiap jlh_bit >=0.5 maka bit di dalam individu di lokasi tersebut = 1.0
    #---lainnya = 0 
    for i in range(0,n):
        for j in range(0,jlh_bit):
            r = np.random.rand(1)
            if (r >= 0.5):
                individu_biner[i,j] = 1.0
    #--------------------Konversi ke bilangan biner
    m_min   = np.array([rho_min,thick_min]) 
    m_max   = np.array([rho_max,thick_max])
    #--------------------print(individu_biner)
    for i in range(0,n):
        for j in range(0,(2*n_lapis)-1):
            sumnya = 0
            for k in range(0,n_bit):
                jn = int(j*n_bit)
                sumnya = sumnya + (individu_biner[i,jn+k]*(2**-(k+1)))   
            if (j < (2*n_lapis)/2):
                individu_desimal[i,j] = m_min[0] + (m_max[0]-m_min[0]) * sumnya
            else:
                individu_desimal[i,j] = m_min[1] + (m_max[1]-m_min[1]) * sumnya
    return(individu_biner,individu_desimal)
    
   
#-------------(2)----------------------------------------------------(komplit)#
def Conv_desm(n,n_lapis,n_bit,individu_biner,m_min,m_max):
    individu_desimal = np.zeros([n,(2*n_lapis)-1])
    for i in range(0,n):
        for j in range(0,(2*n_lapis)-1):
            sumnya = 0
            for k in range(0,n_bit):
                jn = int(j*n_bit)
                sumnya += individu_biner[i,jn+k] * (2**(-(k+1)))   
            if (j < ((2*n_lapis)/2)):
                individu_desimal[i,j] = m_min[0] + (m_max[0]-m_min[0]) * sumnya
            else:
                individu_desimal[i,j] = m_min[1] + (m_max[1]-m_min[1]) * sumnya
    return(individu_desimal)

#-------------(1)----------------------------------------------------(Komplit)#
def Milih(ndat,freq,rho,npopulasi,ind_desimal,silang_rate,n_lapis):
    #---Menggunakan Roda Roulette
    d_cal_rho     = np.zeros([npopulasi,ndat])         
    d_sel_rho_app = np.zeros([npopulasi,ndat])
    d_sum         = np.empty(npopulasi)
    d_rms         = np.empty(npopulasi)
    fitness       = np.zeros(npopulasi)
    probl         = np.empty(npopulasi)
    prob_cum      = np.empty(npopulasi)
  # Melabeli jenis individu disesuaikan dengan indeks baris dari d_cal
    for ii in range(0,npopulasi):
        # data hasil kalkulasi
        [Rho_app_calc,Phase_calc] = f_MT(freq,ind_desimal[ii,:],n_lapis)
        #tidak ada data phasa sehingga nilai lambda = 0
        d_cal_rho[ii,:]           = np.log10(Rho_app_calc/rho)
        # kuadrat dari selisih antara data kalkulasi dan observasi
        d_sel_rho_app[ii,:]       = d_cal_rho[ii,:]*d_cal_rho[ii,:]
        # jumlah dari kuadrat selisih
        d_sum[ii]    = np.sum(d_sel_rho_app[ii,:])
        d_rms[ii]    = np.sqrt(d_sum[ii]/ndat)
        #print(ii,d_rms[ii])
    E_min = np.min(d_rms)
    for i in range(0,npopulasi):
        fitness[i]  = np.exp(-d_rms[i]/E_min)
    for i in range(0,npopulasi):
        probl[i] = fitness[i]/np.sum(fitness)
    prob_cum[0] = probl[0]
    for ii in range(1,npopulasi):
        prob_cum[ii] = prob_cum[ii-1] + probl[ii]
        
    # --------setelah bagian ini, nilai probabilitas kumulatif sudah tersedia
    id_induk  = np.zeros(1)
    jlh_kawin = np.floor(silang_rate*npopulasi)
    id_induk1 = np.zeros(int(jlh_kawin))
    id_induk2 = np.zeros(int(jlh_kawin))
    
    for i in range(0,int(jlh_kawin)):
        R_acak = np.random.rand(1)
        for j in range(0,npopulasi):
            if (R_acak<prob_cum[j]):
                id_induk = np.append(id_induk,j)
        [miy]= np.shape(id_induk)
        id_resize = np.zeros([miy-1])
        id_resize = id_induk[1:]
        id_induk1[i] = int(np.min(id_resize)) #(id_resize[i,:])
        id_induk2[i] = id_induk1[i]-1
        if (id_induk2[i]==-1):
            id_induk2[i]=npopulasi-1
        id_induk = np.zeros(1)
    return(id_induk1,id_induk2,int(jlh_kawin))

#----------------(4)----------------------------------proses cross over------#
def KawinSilang(jlh_bit,ind_biner,id_induk1,id_induk2,jlh_kawin):
    anak_1 = np.empty([jlh_kawin,jlh_bit])
    anak_2 = np.empty([jlh_kawin,jlh_bit])
    #print(anak_1)
    for ii in range(0,jlh_kawin):
        anak_1[ii,:] = ind_biner[int(id_induk1[ii]),:]
        anak_2[ii,:] = ind_biner[int(id_induk2[ii]),:]   
        R_acak = np.random.randint(jlh_bit)
        anak_1[ii,R_acak:jlh_bit] = ind_biner[int(id_induk2[ii]),R_acak:jlh_bit]
        anak_2[ii,0:R_acak] = ind_biner[int(id_induk1[ii]),0:R_acak]
    ind_biner = np.append(ind_biner,anak_1,axis =0)
    ind_biner = np.append(ind_biner,anak_2,axis=0)
    return(ind_biner) 
   
def Seleksi(ndat,freq,rho,npopulasi,ind_desimal,n_lapis):
    #---Menggunakan Roda Roulette
    d_cal_rho     = np.zeros([npopulasi,ndat])         
    d_sel_rho_app = np.zeros([npopulasi,ndat])
    d_sum         = np.empty(npopulasi)
    d_rms         = np.empty(npopulasi)
    Rank_indi     = np.zeros([npopulasi,2])
  # Melabeli jenis individu disesuaikan dengan indeks baris dari d_cal
    for ii in range(0,npopulasi):
        # data hasil kalkulasi
        [Rho_app_calc,Phase_calc] = f_MT(freq,ind_desimal[ii,:],n_lapis)
        #print(Rho_app_calc)
        d_cal_rho[ii,:]           = np.log10(Rho_app_calc/rho)
        # kuadrat dari selisih antara data kalkulasi dan observasi
        d_sel_rho_app[ii,:]       = d_cal_rho[ii,:]*d_cal_rho[ii,:]
        # jumlah dari kuadrat selisih
        d_sum[ii]    = np.sum(d_sel_rho_app[ii,:])
        d_rms[ii]    = np.sqrt(d_sum[ii]/ndat)
    for jj in range(0,npopulasi):
        Rank_indi[jj,0] = jj
        Rank_indi[jj,1] = d_rms[jj]
    m = Rank_indi[Rank_indi[:, 1].argsort()[::1]]
    min_rms = min(d_rms)
    return(m,min_rms)

#-----------------------------------------------------------------------------#
def Mutasi(ind_biner,npopulasi,p_mutasi,jlh_bit):
    jlh_ind_mut = np.floor(npopulasi*p_mutasi)
    for i in range(0,int(jlh_ind_mut)):
        R_acak1 = np.random.randint(npopulasi)
        R_acak2 = np.random.randint(jlh_bit)
        if ind_biner[R_acak1,R_acak2] == 0:
           ind_biner[R_acak1,R_acak2] = 1
        else:
           ind_biner[R_acak1,R_acak2] = 0 
    return(ind_biner)

