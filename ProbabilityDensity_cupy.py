import numpy as np
import cupy as cp
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import xlsxwriter
import pandas as pd
import scipy.stats as st

#np.set_printoptions(threshold=sys.maxsize)


# Fills transpose of mat[N][N] in tr[N][N] 
def transpose(mat, tr, N): 
    for i in range(N): 
        for j in range(N): 
            tr[i][j] = mat[j][i] 
   
# Returns true if mat[N][N] is symmetric, else false 
def isSymmetric(mat, N): 
      
    tr = [ [0 for j in range(len(mat[0])) ] for i in range(len(mat)) ] 
    transpose(mat, tr, N) 
    for i in range(N): 
        for j in range(N): 
            if (mat[i][j] != tr[i][j]): 
                return False
    return True

#Make a matrix symmetric
def makeSymmetric(inputmatrix):
    inputmatrixT = np.transpose(inputmatrix)
    inputmatrix = (inputmatrix + inputmatrixT)
    '''
    for i in range(len(inputmatrix)):
        for j in range(len(inputmatrix[0])):
            if inputmatrix[i][j] not in [0,1]:
                inputmatrix[i][j]=1
    '''
    return inputmatrix

class PDBAtom(object):
    """ Class to represent a single atom's position and state at a frame
    
    Attributes:
        _valence_dict (dict{str: int}): A dictionary of valence electron count per element
        x (float): The x coordinate of the atom
        y (float): The y coordinate of the atom
        z (float): The z coordinate of the atom
        valence_count (int): Number of valence electrons in the atom
    """
    
    
    _valence_dict = {'C': 4,
                     'H': 1,
                     'N': 5,
                     'O': 6,
                     'S': 6}
    
    _electroneg_dict = {'C': 2.55,
                        'H': 2.2,
                        'N': 3.04,
                        'O': 3.44,
                        'S': 2.58}
    
    def __init__(self, string):
        """ Standard PDB file format
        ATOM    277  O1  LYS A  14      21.138  -0.865  -4.761  1.00  0.00           O1-
        """
#       Coordinate Parser 
        self.x = float(string[30:38].strip())
        self.y = float(string[38:46].strip())
        self.z = float(string[46:54].strip())
        
#       Element and Valence Electron Number Parser
        self.element_spec = string[77:].strip()
        mod = 0
        if self.element_spec.endswith(('-', '+')):
            self.element_sym = self.element_spec[:-2].strip()
            mod = int(self.element_spec[-2])
            mod *= (-1, 1)[self.element_spec.endswith('-')]
        else:
            self.element_sym = self.element_spec.strip()
        self.valence_count = PDBAtom._valence_dict.get(self.element_sym)
        if self.valence_count is None:
            raise TypeError('Used an element that is not in the valence dictionary')
        else:
            self.valence_count += mod
        self.electronegativity = PDBAtom._electroneg_dict.get(self.element_sym)


class Adj_Mats(object):
    """ Class to represent a series of adjacency matrices
    
    Attributes:
        file (str): The path of the pdb file to be parsed
        valence_list (Array[Array[int]]): Stores the number of valence electrons in all atoms in every frame
        distance_graphs (Array[Array[Array[int]]]): The series of distance matrices of the atoms in the evolution
        adjacenecy_graphs (Array[Array[Array[int]]]): The series of adjacency matrices of the atoms in the evolution
        elec_adjacency_graphs (Array[Array[Array[int]]]): The series of adjacency matrices of electrons in the evolution
        
    Methods:
        set_atom_dists: Used to set the distance_graphs attribute
        set_atom_adj: Used to set the adjacency_graphs attribute
        get_atom_dists: Used to parse the pdb file to create a distance_graphs object
        get_atom_adj: Used to set an adjacency threshold on the distance matrices and make adjacency matrices
    """
    
    
    def __init__(self, pdb):
        self.file = pdb
        self.valence_list = cp.zeros(1, int)
        cp.cuda.Stream.null.synchronize()
        self.distance_graphs = cp.zeros(1, int)
        cp.cuda.Stream.null.synchronize()
        self.adjacency_graphs = cp.zeros(1, int)
        cp.cuda.Stream.null.synchronize()
        self.elec_adjacency_graphs = cp.zeros(1, int)
        cp.cuda.Stream.null.synchronize()
        self.elneg_adj = cp.zeros(1, int)
        cp.cuda.Stream.null.synchronize()
        self.eigenvalues = None
        self.bin_probs = None
        self.entropy = None
        self.energy = None
        self.cont_ent = None
    
    def set_atom_dists(self, new_dists):
        self.distance_graphs = new_dists
        
    def set_atom_adj(self, new_adj):
        self.adjacency_graphs = new_adj
        
    def get_atom_dists(self):
        if os.path.isfile(self.file):
            pdb_file = open(self.file,'r')
        else:
            raise OSError('File {} does not exist'.format(self.file))

        lineno = 0
        frames = []
        atoms = []
        val_frames = []
        val_atoms = []
        
        for line in pdb_file:
            lineno += 1
            if line.startswith('ATOM'):
                try:
                    at_obj = PDBAtom(line)
                    atoms.append([at_obj.x, at_obj.y, at_obj.z])
                    val_atoms.append(at_obj.valence_count)
                except:
                    sys.stderr.write('\nProblem parsing line {} in file {}\n'.format(lineno, self.file))
                    sys.stderr.write(line)
                    sys.stderr.write('Probably ATOM entry is formatted incorrectly?\n')
                    sys.stderr.write('Please refer to - http://www.wwpdb.org/documentation/format32/sect9.html#ATOM\n\n')
                    sys.exit(1)
            elif line.startswith('END'):
                frames.append(atoms)
                atoms = []
                val_frames.append(val_atoms)
                val_atoms = []
        pdb_file.close()
    
        base = cp.zeros((len(framesindices), len(frames[0]), 3))
        for i in range(len(framesindices)):
            for j in range(len(frames[i])):
                for k in range(len(frames[i][j])):
                    base[i][j][k] = frames[i][j][k]
        dists = cp.reshape(base, (len(framesindices), 1, len(frames[0]), 3)) - cp.reshape(base, (len(framesindices), len(frames[0]), 1, 3))  
        cp.cuda.Stream.null.synchronize()
        dists = dists**2
        dists = dists.sum(3)
        dists = cp.sqrt(dists)
        cp.cuda.Stream.null.synchronize()
        
        self.valence_list = val_frames
        self.distance_graphs = dists
        
        return self.distance_graphs
    
    def get_atom_adj(self, s=1, t=4):
        if len(self.distance_graphs) == 1:
            self.get_atom_dists()

        used_valence_list = set()
        hydrogenbond_count=0

        self.adjacency_graphs = ((self.distance_graphs < t) & (self.distance_graphs > s)).astype(int)

        #Eliminating same-atom intramolecular interactions:
        for frame in range(len(self.adjacency_graphs)):
            for i in range(len(self.adjacency_graphs[frame])):
                for j in range(len(self.adjacency_graphs[frame][i])):
                    if (i//3==j//3):
                        self.adjacency_graphs[frame][i][j]=1
                    elif ((self.valence_list[frame][i]!=self.valence_list[frame][j]) & (self.adjacency_graphs[frame][i][j]==1)):
                        self.adjacency_graphs[frame][i][j]=1
                        if (i,j) not in used_valence_list:
                            hydrogenbond_count+=1
                        else: 
                            pass
                        used_valence_list.add((i,j))
                        used_valence_list.add((j,i))
                    else:
                        self.adjacency_graphs[frame][i][j]=0

        hydrogenbonds_array.append(hydrogenbond_count)

        return self.adjacency_graphs
    
    def get_elec_adj(self):
        #if len(self.adjacency_graphs) == 1:
        self.get_atom_adj(lowerlimit,upperlimit)
            
        total_val = 0
        
        for i in range(len(self.valence_list[0])):
            total_val += self.valence_list[0][i]
        valencelistframes = len(self.valence_list)
        self.elec_adjacency_graphs = cp.zeros((len(framesindices), total_val, total_val))
        cp.cuda.Stream.null.synchronize()
        curr_n, curr_m = 0, 0
        
        for i in range(len(self.adjacency_graphs)):
            for j in range(len(self.adjacency_graphs[0])):
                for b in range(self.valence_list[i][j]):
                    for k in range(len(self.adjacency_graphs[0][0])):
                        for a in range(self.valence_list[i][k]):
                            self.elec_adjacency_graphs[i][curr_n][curr_m] = self.adjacency_graphs[i][j][k]
                            curr_m += 1
                    curr_m = 0
                    curr_n += 1
            curr_n = 0  
        return self.elec_adjacency_graphs
    
    def make_eigenvalues(self, hamiltonian_iter=1000):
        #self.elec_adjacency_graphs=[]
        self.elec_adjacency_graphs=self.get_elec_adj()
        elec_count = len(self.elec_adjacency_graphs[0])
        self.eigenvalues = cp.zeros((len(self.elec_adjacency_graphs), hamiltonian_iter, elec_count))
        cp.cuda.Stream.null.synchronize()
        for frame in range(len(self.elec_adjacency_graphs)):
            frame_eigs = []
            for i in range(hamiltonian_iter):
                print(i)
                r = cp.random.normal(size=(elec_count, elec_count))
                cp.cuda.Stream.null.synchronize()
                rt = cp.transpose(r)
                cp.cuda.Stream.null.synchronize()
                h = (r + rt) / np.sqrt(2 * elec_count)          
                adj_r = self.elec_adjacency_graphs[frame] * h
                eigs = cp.ndarray.tolist(np.linalg.eigvals(adj_r))
                cp.cuda.Stream.null.synchronize()
                eigs.sort()
                #for i in range(len(eigs)):
                frame_eigs.append(cp.real(eigs))
                cp.cuda.Stream.null.synchronize()
            self.eigenvalues[frame] = frame_eigs
        return self.eigenvalues

    def get_spacings(self,types):
        allframes_spacing=[]
        eigenvalues = self.make_eigenvalues()
        #eigenvalues=[np.random.rand(1000,60)]
        eigenvalues=cp.array(eigenvalues)
        cp.cuda.Stream.null.synchronize()
        if types=='all':
            medians=[]
            nexttomedian=[]
            spacings=[]
            allspacings=cp.zeros((len(eigenvalues[0]),len(eigenvalues[0][0])-1))
            cp.cuda.Stream.null.synchronize()
            for i in range(len(eigenvalues[0])):
                for j in range(len(eigenvalues[0][0])-1):
                    allspacings[i][j]=(eigenvalues[0][i][j+1]-eigenvalues[0][i][j])
            allspacings=allspacings.transpose()
            counts, binedges = cp.histogram(allspacings)
            cp.cuda.Stream.null.synchronize()
            return [counts, allspacings]
        else:
            for frame in range(len(self.elec_adjacency_graphs)):
                medians=[]
                nexttomedian=[] 
                spacings=[]
                for i in range(len(eigenvalues[frame])):
                    #Calculating medians and converting eigenvalue array to 1xn list:
                    medians.append(np.median(eigenvalues[frame][i]))
                    if len(eigenvalues[frame][i])%2==0:
                        nexttomedian.append((eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+1]+eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+2])/2)
                    else:
                        nexttomedian.append(eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+1])
                #Calculating median and median+1 spacings, along with spacing standard deviation:
                for i in range(len(medians)):
                    spacings.append(abs(nexttomedian[i]-medians[i]))
                allframes_spacing.append(spacings)
                print(allframes_spacing)
            return allframes_spacing

    def get_stdevs(self):
        stdevs=[]
        medianvalues=[]
        eigenvalues = self.make_eigenvalues()
        for i in range(len(eigenvalues)):
            eigenvalues[i].sort()
            medianvalues.append(eigenvalues[i])
        stdevs.append(np.std(medianvalues))
        return stdevs

def createUpperLimitList(low,high,spacing):
    limitlist=[]
    for i in range(int((high-low)/spacing)+1):
        value=low+(spacing*i)
        limitlist.append(value)
    return limitlist

#Create bins
def makeBins(array,binnumber):
    minima=[]
    maxima=[]
    for z in range(len(array)):
        minima.append(min(array[z]))
        maxima.append(max(array[z]))
    minimum=min(minima)
    maximum=max(maxima)
    difference=maximum-minimum
    binsize=difference/binnumber
    bins=[]
    for i in range(numberofbins):
        bins.append(minimum+binsize*i)
    bins.append(maximum)
    return [bins,minimum,maximum]

#Check which bin each spacing lands in
def whichBin(binnumber,array):
    print(array)
    instances=cp.zeros((len(array),binnumber))
    probability=cp.zeros((len(array),binnumber))
    bininfo=makeBins(array,binnumber)
    bins=bininfo[0]
    valuemin=bininfo[1]
    valuemax=bininfo[2]
    print(len(array))
    print(len(array[0]))
    print(len(bins))
    for i in range(len(array)):
        print(i)
        for j in range(len(array)):
            for k in range(len(bins)-1):
                if k<max(range(len(bins))):
                    if (bins[k]<=array[i][j]<bins[k+1]):
                        instances[i][k]+=1
                    else:
                        pass
                else:
                    if (bins[k]<=array[i][j]<=bins[k+1]):
                        instances[i][k]+=1
                    else:
                        pass
        instances[i]=instances[i]/binnumber
        probability=instances
        probability=probability.transpose()
    return [probability,valuemin,valuemax]


def binSizes(values,binscount):
    binvalues=[]
    #if framecount==1:
    width=abs(max(values)-min(values))
    for i in range(binscount):
        print(i)
        binvalues.append(sum([min(values),i*(width/binscount)]))
    '''
    else:
        for frame in range(len(values)):
            width=abs(max(values[frame])-min(values[frame]))
            for i in range(binscount):
                print(i)
                binvalues.append(sum([min(values[frame]),i*(width/binscount)]))
    '''
    return binvalues

def findIndex(value,list_values):
    for i range(len(list_values)):
        if list_values[i]==value:
            return int(i)
        else: pass

types='all_frames'
lowerlimit=1
upperlimit_list=createUpperLimitList(lowerlimit,10,0.5)
spacinghistograms=[]
framesindices=cp.linspace(0,2500,100)
cp.cuda.Stream.null.synchronize()
hydrogenbonds_allframes=[]
hydrogenbonds_array=[]

data_folder = "/home/gemsec-user/Desktop/"
file_to_open = data_folder + "water.pdb"
file = open(file_to_open)


#for frameindex in framesindices:
hydrogenbonds=[]
allspacings=[]
numberofbins=100
histogram=[]

#naming protocol
names_allframes=[]
numbers=upperlimit_list
for j in range(len(numbers)):
    name_string=str(numbers[j]) + 'Å' 
    names_allframes.append(name_string)
#names_allframes.append(names)
print(names_allframes)

valueiterations=len(upperlimit_list)

for upperlimit in upperlimit_list:
    print(upperlimit)
    if __name__ == "__main__":
        full = Adj_Mats(file_to_open)
        spacinginfo=full.get_spacings(types)

        if (types=='all'):
            histogram.append(list(spacinginfo[0]))
            allspacings=spacinginfo[1]

        elif (types=='median'):
            spacinghistograms=spacinginfo
            
        else:
            #Creating histogram data
            spacinghistograms=spacinginfo
            x = range(len(spacinginfo))
            x_coord = cp.repeat(x, len(spacinginfo[0]))
            cp.cuda.Stream.null.synchronize()
            spacing_array = cp.array(spacinginfo)
            cp.cuda.Stream.null.synchronize()
            allhists = cp.ravel(spacing_array)
            cp.cuda.Stream.null.synchronize()

            #Saving data to excel spreadsheet for batching
            workbook = xlsxwriter.Workbook('Frech_Christian_cupyprobdensitydata.xlsx')
            worksheet = workbook.add_worksheet()

            bold = workbook.add_format({'bold'=True})
            col = findIndex(upperlimit,upperlimit_list)
            
            worksheet.write(0, col, upperlimit, bold)
            for row in range(len(stdevvalues)):
                worksheet.write((row+1), col, allhists[row])

            # Create a figure for plotting the data as a 3D histogram.
            fig, ax = plt.subplots(1,1)
            weighting = cp.ones_like(allhists) / len(spacinginfo[0])
            cp.cuda.Stream.null.synchronize()
            ax.hist2d(allhists, x_coord, bins=numberofbins, weights=weighting)
            ax.set_xlabel('Spacing')
            ax.set_ylabel('Time Elapsed (10ns)')

            filename='3DNetwork_{5}frame_probdensity3dplot_'+str(upperlimit)+'.png'
            filenames.append(filename)
            plt.savefig(filename, dpi=95)
            plt.clf()

    else:
        pass

hydrogenbonds_allframes.append(hydrogenbonds)
allspacings_list=cp.array(spacinghistograms)
cp.cuda.Stream.null.synchronize()

fig, axes = plt.subplots(1,1)
print('check')

###BIN SIZES###

if (types=='all'):
    spacingmin=min(range(len(allspacings[0])))
    spacingmax=max(range(len(allspacings[0])))
    spacingiterations=len(allspacings[0])
    valuemin=lowerlimit
    #spacinginfo[1]
    valuemax=max(upperlimit_list)
    #spacinginfo[2]
    eigenvaluearray=cp.array(valueiterations)

    #XY-plane
    yy=cp.linspace(valuemin,valuemax,valueiterations)
    cp.cuda.Stream.null.synchronize()
    xx=cp.linspace(spacingmin,spacingmax,spacingiterations)
    cp.cuda.Stream.null.synchronize()
    X,Y=cp.meshgrid(xx,yy)
    cp.cuda.Stream.null.synchronize()

    instances=[]

    #Z value Distributions
    print(allspacings_list)
    allspacings_list=cp.array(allspacings_list)
    cp.cuda.Stream.null.synchronize()
    Z = allspacings_list
    print(X.shape)
    print(Y.shape)
    print(Z.shape)

    #Make 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    #ax.set_ylim3d(0,0.005)   

    ax.set_xlabel('Eigenvalue Spacing Pair #')
    ax.set_ylabel('Spacing Value')
    ax.set_zlabel('Probability Value')
    plt.savefig('3DWater_{1}frame_hydrogenbonds_allpairs_spacingdistribution_sorted_1.55-2.2A.png', dpi=95)
    plt.show()

elif (types=='median'):
    #fig, axes = plt.subplots(1,2) 
    Z = allframes_spacing.transpose()
    fig = plt.figure(figsize=plt.figaspect(0.5))

    #Creating X-axis for 3D plots (frames)
    framemin=min(range(len(Z)))
    framemax=max(range(len(Z)))
    frameiterations=len(Z)
    xx=cp.linspace(framemin,framemax,frameiterations)
    cp.cuda.Stream.null.synchronize()

    iterations=[upperlimit_list,hydrogenbonds_array]
    xaxis_labels=['Cut-Off Distance For Interactions (Angstroms)', 'Number of Hydrogen Bonds']


    for frame in range(len(framesindices)):
        for i in range(len(allspacings_list)):
            print(frame)
            print(i)
            spacingbinvalues=binSizes(allspacings_list[frame][i], numberofbins)
            weighting = cp.ones_like(allspacings_list[frame][i]) / len(allspacings_list[frame][i])
            cp.cuda.Stream.null.synchronize()
            #ENTER WEIGHTINGS BACK IN FOR PROBABILITY
            axes.hist(allspacings_list[frame][i], bins = spacingbinvalues, alpha=0.5, weights=weighting)


        #axes[col].plot(iterations[col], allframes_stdevs)  
        #ax[col].set_xlabel(xaxis_labels[col])
        #ax[col].set_ylabel('Spacings Standard Deviation')
    #plt.savefig('3DWater_{1}frame_hydrogenbonds_standarddeviation.png', dpi=95)
    plt.show()

else: 
    images=[]
    for name in filenames:
        images.append(imageio.imread(name))
    imageio.mimsave('3DNetwork_{3}frame_probdensity3dplot.gif', images, duration=0.5)

#axes.legend(names,loc=2)
print('end')
axes.set_ylabel('Probability')
axes.set_xlabel('Spacing')
#plt.savefig('3DWater_{1}frame_hydrogenbonds_spacingdistribution_sorted_1.55-2.2A.png', dpi=95)
plt.show()