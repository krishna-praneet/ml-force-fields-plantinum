import numpy as np
import math

# Initialization for periodic repitions
cell = [7.84000, 7.84000, 3.92000] # Platinum
matrix = [[0,0,1], [0,0,-1], [0,1,0], [0,-1,0],
         [1,0,0], [-1,0,0], [0,1,1], [0,1,-1],
         [1,0,1], [-1,0,1], [1,1,1], [-1,1,1],
         [1,-1,1], [-1,-1,1], [-1,1,0], [1,1,0],
         [-1,-1,0], [1,-1,0], [0,1,-1], [0,-1,-1],
         [1,0,-1], [-1,0,-1], [-1,1,-1], [1,1,-1],
         [-1,-1,-1], [1,-1,-1]]
rep_matrix = []
for pos in matrix:
    a = [cell[0]*pos[0], cell[1]*pos[1], cell[2]*pos[2]]
    rep_matrix.append(a)
rep_matrix = np.array(rep_matrix)

def fingerprint(data, Rc=7.0, eta=7.0):
    ''' 
    Function to transform bulk arrangement into fingerprints
    Input args: 
        data: atomic positions 
        Rc: cut-off radius
        eta:gaussian width
    Returns -> fingerprint
    '''
    ## Periodic translations,  only meant for bulk fingerprint 
    ##         Not required for cluter fingerprinting
    complete_data = data
    for i in rep_matrix:
        multi = np.array([i]*16, dtype='float64')
        translate = data + multi
        complete_data = np.concatenate((complete_data, translate), axis=1)
    complete_data = np.reshape(complete_data.flatten(), (16, 27, 3))
    ######################
    
    #Calculate fingerprints
    fingerprint = []
    for row in complete_data:
        curr_pos = np.array([[row[0]]*27]*16)
        diff = complete_data - curr_pos
        sqr = np.square(diff)
        summa = np.sum(sqr, axis=2)
        
        rij = np.sqrt(summa)
        
        a,b,c = [diff[:,:,i] for i in range(3)]
        cos1 = np.divide(a, rij)
        cos2 = np.divide(b, rij)
        cos3 = np.divide(c, rij)

        cos1[np.where(np.isnan(cos1))] = 0.
        cos2[np.where(np.isnan(cos2))] = 0.
        cos3[np.where(np.isnan(cos3))] = 0.
  
        exponent = np.exp(-1*np.square(rij/eta))

        filtered = np.where(rij > Rc, 0, rij)
        prod = (math.pi*filtered)/Rc
        cos = np.cos(np.where(prod==0, -math.pi, prod))
        damping = 0.5 * (cos+1)
        
        common = exponent * damping
        
        v1 = np.sum(cos1 * common)
        v2 = np.sum(cos2 * common)
        v3 = np.sum(cos3 * common)
        
        fingerprint.append([v1,v2,v3])
    return np.array(fingerprint)