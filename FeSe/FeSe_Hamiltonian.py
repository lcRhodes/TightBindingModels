import numpy as np

"""""""""
Hamiltonian for FeSe
original tight binding model from Eschrig 2009
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.80.104503
parameters used for tetragonal state presented in Rhodes 2017
https://www.nature.com/articles/s41535-021-00341-6
"""""""""
class FeSe():
    def __init__(self,NEMATIC=0,SOC = True):
        #CHANGE THESE PARAMETERS ONLY
        self.SOC = SOC

        if self.SOC==True:
            self.length = 20
            self.NHam =4
        else:
            self.length = 10
            self.NHam=2
        self.halflength= 10
        self.quartlength= 5


        self.nem = NEMATIC
        if 0 <= self.nem <= 1:
            self.OO_dxzyz = 0.029*(self.nem)
            self.OO_dxy =   0.029*(self.nem)
            self.mu = 0.020 - 0.008*(1-self.nem)
            self.Phi_dxzyz_electron = 0.014*(self.nem)
            self.Phi_dxzyz_hole = 0.007*(self.nem)
            self.Phi_dxy = -0.028*(self.nem)
            self.Phi_dxy_hartree = 0.028*(self.nem)
        else:
            print("Error, NEMATIC must be between 0.0 and 1.0 (0% and 100%)")
            exit(0)


        #Load Spin Orbit Coupling matrices
        self.SO_z = 0.019
        self.SO_xy = 0.005

        self.lx = np.zeros((self.halflength,self.halflength),dtype=np.complex128)
        self.ly = np.zeros((self.halflength,self.halflength),dtype=np.complex128)
        self.lz = np.zeros((self.halflength,self.halflength),dtype=np.complex128)

        self.lx[0,2] = -1j #dxy/dxz
        self.lx[1,3] = 1j #dx2y2/dyz
        self.lx[2,0] = 1j #dxz/dxy
        self.lx[3,1] = -1j #dyz/dx2y2
        self.lx[3,4] = -np.sqrt(3)*1j #dyz/dz2
        self.lx[4,3] = np.sqrt(3)*1j #dz2/dyz

        self.ly[0,3] = 1j #dxy/dyz
        self.ly[1,2] = 1j #dx2y2/dxz
        self.ly[2,1] = -1j #dxz/dx2y2
        self.ly[2,4] = np.sqrt(3)*1j #dxz/dz2
        self.ly[3,0] = -1j #dyz/dxy
        self.ly[4,2] = -np.sqrt(3)*1j #dz2/dxz

        self.lx[5,7] = self.lx[0,2]
        self.lx[6,8] = self.lx[1,3]
        self.lx[7,5] = self.lx[2,0]
        self.lx[8,6] = self.lx[3,1]
        self.lx[8,9] = self.lx[3,4]
        self.lx[9,8] = self.lx[4,3]

        self.ly[5,8] = self.ly[0,3]
        self.ly[6,7] = self.ly[1,2]
        self.ly[7,6] = self.ly[2,1]
        self.ly[7,9] = self.ly[2,4]
        self.ly[8,5] = self.ly[3,0]
        self.ly[9,7] = self.ly[4,2]

        self.lz[0,1] = 2j #dxy/dx2y2
        self.lz[1,0] = -2j #dx2y2/dxz
        self.lz[2,3] = -1j #dxz/dyz
        self.lz[3,2] = 1j #dyz/dxz

        self.lz[5,6] = self.lz[0,1]
        self.lz[6,5] = self.lz[1,0]
        self.lz[7,8] = self.lz[2,3]
        self.lz[8,7] = self.lz[3,2]

        #using notation of Echrig and Koepernik Phys. Rev. B 80, 104503 (2009) Tight-binding models for the iron-based superconductors
        #https://journals.aps.org/prb/abstract/10.1103/PhysRevB.80.104503
        #t_10_16 corresponds to a hopping along 1 unit cell in Rx and no unit cells in Ry, (10) between orbital 1 and orbital 6
        #Note 16 corresponds to the Hamiltonian indices [0,5]
        self.hopping = {
        "t11_11":0.0181799780,#0
        "t10_16":-0.0313287569,
        "t20_11":0.000929691422,
        "t21_16":-0.00230902409,
        "t11_13":-0.01225797j,
        "t10_18":0.11516269j,#5
        "t11_15":-0.04,
        "t10_27":-0.0498818920,
        "t11_22":-0.01668514,
        "t10_29":-0.0949260978j,
        "t11_23":0.0148384369j,#10
        "t11_210":0.0596492240,
        "t11_33":0.0243279700,
        "t10_38":0.00867590303,
        "t20_33":0.000964007446,
        "t21_38":-0.00493307501,#15
        "t02_33":-0.00716813661,
        "t10_49":0.0502319069,
        "t22_33":0.00758116474,
        "t21_49":-0.0000848738783,
        "t11_34":0.0165,#20
        "t10_410":-0.00902000930j,
        "t11_35":0.00568905629j,
        "E1":0.0340544241,
        "E2":0.02,
        "E3":0.00309573361,#25
        "E5":-0.193976913,
        "t101_16":0.0027,
        "t001_11":0.00,
        "t121_16":-0.0028333,
        "t111_11":0,#30
        "t101_18":0.00149999j,
        "t201_11":0.00013333,
        "t101_19":0.003333j,
        "t201_14":0.0095j,
        "t121_19":0.0051666j,#35
        "t001_38":0.0018333,
        "t101_38":0.003,
        "t201_33":-0.001333,
        "t121_38":-0.0005,
        "t021_33":0.003333,#40
        "t101_39":0.0025,
        "t101_49":0.00216666,
        "t121_49":0.001}


        #0-4 Fe1 dxy, dx2y2, dxz,dyz,dz2
        #5-9 Fe2 dxy, dx2y2, dxz,dyz,dz2
    def Load_Hamiltonian(self,kx,ky,kz):
        k1 = kx+ky
        k2 = -kx+ky

        H = np.zeros((self.halflength,self.halflength),dtype=np.complex128)

        #Load framework
        ### H++
        H[0,0] = self.hopping["E1"] + 2*self.hopping["t11_11"]*(np.cos(k1)+np.cos(k2)) + 2*self.hopping["t20_11"]*(np.cos(2*k1)+np.cos(2*k2)) + (2*self.hopping["t001_11"] + 4*self.hopping["t111_11"]*(np.cos(k1)+np.cos(k2)) + 4*self.hopping["t201_11"]*(np.cos(2*kx)+np.cos(2*ky)))*np.cos(kz)
        H[0,1] = 0
        H[0,2] = 2j*self.hopping["t11_13"]*(np.sin(k1)-np.sin(k2)) - 4*self.hopping["t201_14"]*np.sin(2*ky)*np.sin(kz)
        H[0,3] = 2j*self.hopping["t11_13"]*(np.sin(k1)+np.sin(k2)) - 4*self.hopping["t201_14"]*np.sin(2*kx)*np.sin(kz)
        H[0,4] = 2*self.hopping["t11_15"]*(np.cos(k1)-np.cos(k2))

        H[1,0] = np.conj(H[0,1])
        H[1,1] = self.hopping["E2"] + 2*self.hopping["t11_22"]*(np.cos(k1)+np.cos(k2))
        H[1,2] = 2j*self.hopping["t11_23"]*(np.sin(k1)+np.sin(k2))
        H[1,3] = 2j*self.hopping["t11_23"]*(-np.sin(k1)+np.sin(k2))
        H[1,4] = 0

        H[2,0] = np.conj(H[0,2])
        H[2,1] = np.conj(H[1,2])
        H[2,2] = self.hopping["E3"] + 2*self.hopping["t11_33"]*(np.cos(k1)+np.cos(k2)) + 2*self.hopping["t20_33"]*np.cos(2*kx) + 2*self.hopping["t02_33"]*np.cos(2*ky) + 4*self.hopping["t22_33"]*np.cos(2*kx)*np.cos(2*ky) + (2*self.hopping["t001_38"] + 4*self.hopping["t201_33"]*np.cos(2*kx) + 4*self.hopping["t021_33"]*np.cos(2*ky))*np.cos(kz)
        H[2,3] = 2*self.hopping["t11_34"]*(np.cos(k1)-np.cos(k2))
        H[2,4] = 2j*self.hopping["t11_35"]*(np.sin(k1)+np.sin(k2))

        H[3,0] = np.conj(H[0,3])
        H[3,1] = np.conj(H[1,3])
        H[3,2] = np.conj(H[2,3])
        H[3,3] = self.hopping["E3"] + 2*self.hopping["t11_33"]*(np.cos(k1)+np.cos(k2)) + 2*self.hopping["t02_33"]*np.cos(2*kx) + 2*self.hopping["t20_33"]*np.cos(2*ky) + 4*self.hopping["t22_33"]*np.cos(2*kx)*np.cos(2*ky) + (2*self.hopping["t001_38"] + 4*self.hopping["t021_33"]*np.cos(2*kx) + 4*self.hopping["t201_33"]*np.cos(2*ky))*np.cos(kz)
        H[3,4] = 2j*self.hopping["t11_35"]*(np.sin(k1)-np.sin(k2))

        H[4,0] = np.conj(H[0,4])
        H[4,1] = np.conj(H[1,4])
        H[4,2] = np.conj(H[2,4])
        H[4,3] = np.conj(H[3,4])
        H[4,4] = self.hopping["E5"]

        ### H+-
        H[0,5] = 2*self.hopping["t10_16"]*(np.cos(kx)+np.cos(ky)) + 2*self.hopping["t21_16"]*((np.cos(k1)+np.cos(k2))*(np.cos(kx)+np.cos(ky)) - np.sin(k1)*(np.sin(kx)+np.sin(ky)) + np.sin(k2)*(np.sin(kx)-np.sin(ky))) + 4*self.hopping["t101_16"]*(np.cos(kx)+np.cos(ky))*np.cos(kz) +2*self.hopping["t121_16"]*((np.cos(k1+ky)+np.cos(k1+kx))*np.exp(1j*kz) + (np.cos(k2+ky)+np.cos(k2-kx))*np.exp(-1j*kz))
        H[0,6] = 0
        H[0,7] = 2j*self.hopping["t10_18"]*np.sin(kx) - 4*(self.hopping["t101_18"]*np.sin(kx) + self.hopping["t101_19"]*np.sin(ky))*np.sin(kz) + 2j*self.hopping["t121_19"]*(np.sin(k1+ky)*np.exp(1j*kz)-np.sin(k2+ky)*np.exp(-1j*kz))
        H[0,8] = 2j*self.hopping["t10_18"]*np.sin(ky) - 4*(self.hopping["t101_19"]*np.sin(kx) + self.hopping["t101_18"]*np.sin(ky))*np.sin(kz) + 2j*self.hopping["t121_19"]*(np.sin(k1+kx)*np.exp(1j*kz)+np.sin(k2-kx)*np.exp(-1j*kz))
        H[0,9] = 0

        H[1,5] = H[0,6]
        H[1,6] = 2*self.hopping["t10_27"]*(np.cos(kx)+np.cos(ky))
        H[1,7] = -2j*self.hopping["t10_29"]*np.sin(ky)
        H[1,8] = 2j*self.hopping["t10_29"]*np.sin(kx)
        H[1,9] = 2*self.hopping["t11_210"]*(np.cos(kx)-np.cos(ky))

        H[2,5] = H[0,7]
        H[2,6] = H[1,7]
        H[2,7] = 2*self.hopping["t10_38"]*np.cos(kx) + 2*self.hopping["t10_49"]*np.cos(ky) + 2*self.hopping["t21_38"]*((np.cos(k1)+np.cos(k2))*np.cos(kx)-(np.sin(k1)-np.sin(k2))*np.sin(kx)) + 2*self.hopping["t21_49"]*((np.cos(k1)+np.cos(k2))*np.cos(ky)-(np.sin(k1)+np.sin(k2))*np.sin(ky)) + 4*(self.hopping["t101_38"]*np.cos(kx)+ self.hopping["t101_49"]*np.cos(ky))*np.cos(kz) + 2*self.hopping["t121_38"]*(np.cos(k1+kx)*np.exp(1j*kz) + np.cos(k2-kx)*np.exp(-1j*kz)) + 2*self.hopping["t121_49"]*(np.cos(k1+ky)*np.exp(1j*kz) + np.cos(k2+ky)*np.exp(-1j*kz))
        H[2,8] = 4j*self.hopping["t101_39"]*(np.cos(kx)+np.cos(ky))*np.sin(kz)
        H[2,9] = 2j*self.hopping["t10_410"]*np.sin(ky)

        H[3,5] = H[0,8]
        H[3,6] = H[1,8]
        H[3,7] = H[2,8]
        H[3,8] = 2*self.hopping["t10_49"]*np.cos(kx) + 2*self.hopping["t10_38"]*np.cos(ky) + 2*self.hopping["t21_49"]*((np.cos(k1)+np.cos(k2))*np.cos(kx)-(np.sin(k1)-np.sin(k2))*np.sin(kx)) + 2*self.hopping["t21_38"]*((np.cos(k1)+np.cos(k2))*np.cos(ky)-(np.sin(k1)+np.sin(k2))*np.sin(ky)) + 4*(self.hopping["t101_49"]*np.cos(kx)+ self.hopping["t101_38"]*np.cos(ky))*np.cos(kz) + 2*self.hopping["t121_49"]*(np.cos(k1+kx)*np.exp(1j*kz) + np.cos(k2-kx)*np.exp(-1j*kz)) + 2*self.hopping["t121_38"]*(np.cos(k1+ky)*np.exp(1j*kz) + np.cos(k2+ky)*np.exp(-1j*kz))
        H[3,9] = 2j*self.hopping["t10_410"]*np.sin(kx)

        H[4,5] = H[0,9]
        H[4,6] = H[1,9]
        H[4,7] = H[2,9]
        H[4,8] = H[3,9]
        H[4,9] = 0

        #extract H++ and H+-
        H_sym = np.zeros((self.quartlength,self.quartlength),dtype=np.complex128)
        H_anti = np.zeros((self.quartlength,self.quartlength),dtype=np.complex128)
        for i in range(self.quartlength):
            for j in range(self.quartlength):
                H_sym[i,j] = H[i,j] #++
                H_anti[i,j] = H[i,j+self.quartlength] #+-

        #use properties H-- = H++* to calculate full matrix
        H_anticonj = np.conjugate(H_anti)
        H_symconj = np.conjugate(H_sym)

        #generate full 10x10 hamiltonian
        for i in range(self.quartlength):
            for j in range(self.quartlength):
                H[i+self.quartlength,j] = H_anticonj[i,j]
                H[i+self.quartlength,j+self.quartlength] = H_symconj[i,j]

        #change fermi surface and add in order parameter
        H = H - self.mu*np.identity(self.halflength)

        #d-wave - xz/yz electron pocket only
        H[3,8] += self.Phi_dxzyz_electron*(np.cos(kx) - np.cos(ky))
        H[8,3] += self.Phi_dxzyz_electron*(np.cos(kx) - np.cos(ky))
        H[2,7] += self.Phi_dxzyz_electron*(np.cos(kx) - np.cos(ky))
        H[7,2] += self.Phi_dxzyz_electron*(np.cos(kx) - np.cos(ky))

        #extended s-wave xz/yz hole pocket only
        H[3,8] += self.Phi_dxzyz_hole*(np.cos(kx) + np.cos(ky))
        H[8,3] += self.Phi_dxzyz_hole*(np.cos(kx) + np.cos(ky))
        H[2,7] -= self.Phi_dxzyz_hole*(np.cos(kx) + np.cos(ky))
        H[7,2] -= self.Phi_dxzyz_hole*(np.cos(kx) + np.cos(ky))

        #d-wave - xy electron pocket only
        H[0,5] += self.Phi_dxy*(np.cos(kx) - np.cos(ky))
        H[5,0] += self.Phi_dxy*(np.cos(kx) - np.cos(ky))


        #extended s-wave xy hole pocket only
        H[0,0] -= self.Phi_dxy_hartree*(np.cos(k1) + np.cos(k2))
        H[5,5] -= self.Phi_dxy_hartree*(np.cos(k1) + np.cos(k2))

        H[0,5] += self.Phi_dxy_hartree*(np.cos(kx) + np.cos(ky))
        H[5,0] += self.Phi_dxy_hartree*(np.cos(kx) + np.cos(ky))

        if self.SOC == True:
            #Generate 20x20 SOC hamilitonian
            H_SOC = np.zeros((self.length,self.length),dtype=np.complex128)
            Quad1 = H + (self.SO_z/2)*self.lz
            Quad2 = (self.SO_xy/2)*(self.lx-(self.ly*1j))
            Quad3 = (self.SO_xy/2)*(self.lx+(self.ly*1j))
            Quad4 = H - (self.SO_z/2)*self.lz

            for i in range(self.halflength):
                for j in range(self.halflength):
                    H_SOC[i,j] = Quad1[i,j]
                    H_SOC[i,j+self.halflength] = Quad2[i,j]
                    H_SOC[i+self.halflength,j] = Quad3[i,j]
                    H_SOC[i+self.halflength,j+self.halflength] = Quad4[i,j]

            return H_SOC
        else:
            return H
