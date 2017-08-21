import numpy as np
import numpy.linalg as linalg

'''
TO DO:
1.
The slicing copy for list is shallow
There are very likely to be wrong in some function.
2.
dynamic programming to map MPS --> Vect
3.
Consider merging Bond_dim / tol on truncation

'''



'''
## Variable Name
A_list = left canonical MPS (Normalized)
M_list = MPS ( not Normalized)
GL_list = GL_list????????
'''


'''
## Functions Available:
## Random_MPS:                      --> M_list
## Vec_decomp_A:        Vec         --> A_list,     Lambda_list
## Vec_decomp_A_tol:
    ## A_list is in right canonoical representation
    ## The return MPS is truncated if tol cond is met
    
## Mat_decomp_A:        Mat         --> A_list, L_list


## A_to_vector:      M_list ,index  --> c
## gen_index:           number      --> index(binary representation)
## dot_MPS( MPS1, MPS2)             --> scaler

## MPS_copy
## MPS_compression
## MPS_addition

## Vec_decomp_GL:       Vec         --> Gamma_list, Lambda_list
## A_decomp_GL:         A_list      --> Gamma_list, Lambda_list
## M_decomp_A:          M_list      --> A_list,     Lambda_list
## GL_print_vector:     Glist,Llist,index --> c
'''

def Norm_1_MPS(MPS):
    MPS_copy=MPS[:]
    L=len(MPS_copy)
    #a[0,1,1]=1 , a[1,1,1]=1 , a[1,0,0]=1 ,  a[0,0,0]=1

    Op = np.array([1,0,0,1,1,0,0,1])  ## marginizing(summing) left
    #    Op = np.array([1,1,0,0,0,0,1,1]) ## marginizing(summing) right
    Op = Op.reshape((2,2,2))
    for i in range(L-1):
        Sum=np.einsum('ijk,klm->ijlm',MPS_copy[i],MPS_copy[i+1])
        MPS_copy[i+1] = np.einsum('ijlm,jnl->inm',Sum,Op)
        print MPS_copy[i+1]

#    MPS_copy[0][0,0,0]=100

    return MPS_copy[L-1].sum()


def Random_MPS(L,d,Bond_dim):
    #This function generate random MPS without normalization
    MPS_list=[]
    for i in range(0,L):
        if i<L/2:
            l_dim = min(d**i,Bond_dim)
            r_dim = min(d**(i+1),Bond_dim)
            MPS_list.append( np.random.rand(l_dim,d,r_dim) )
            print "Generating random Matrix:",(l_dim,d,r_dim)


        elif (i+0.5)==L/2.:
            l_dim = min(d**i,Bond_dim)
            r_dim = min(d**i,Bond_dim)
            MPS_list.append( np.random.rand(l_dim,d,r_dim) )
            print "Generating random Matrix:",(l_dim,d,r_dim)


        else:
            l_dim = min(d**(L-i),Bond_dim)
            r_dim = min(d**(L-i-1),Bond_dim)
            MPS_list.append( np.random.rand(l_dim,d,r_dim) )
            print "Generating random Matrix:",(l_dim,d,r_dim)

    return MPS_list




def Vec_decomp_A( indices, Vec, Bond_dim=100, normalize=False):
    '''
    Decompose a d^L dim Vector, Ivec, into MPS in
    left-canonoical form with maximal dim, Bond_dim.
    '''
    A_list=[]
    Lambda_list=[]
    Psi=Vec.reshape((1,len(Vec)))
    L=len(indices)

    dim=1
    for i in range(L):
        dim=dim*indices[i]

    if (len(Vec) != dim):
        print "ERROR! Dim mismatch"
        raise Exception

    for i in range(L-1):
        l_dim,r_dim = Psi.shape
        Psi = np.reshape(Psi,(l_dim*indices[i], r_dim/indices[i]))

        U,s,Vh = linalg.svd(Psi,full_matrices=False)

        if len(s)>Bond_dim:
            U=U[:,:Bond_dim]
            print "Truncate !!!", sum(s[Bond_dim:]**2)
            s=s[:Bond_dim]
            if normalize == True:
                s_2norm = linalg.norm(s,2)
                s = s/ s_2norm

            Sig = np.diagflat(s)
            Psi=Sig.dot(Vh[:Bond_dim,:])

            A_list.append( np.reshape( U, (l_dim,indices[i],Bond_dim)))
            Lambda_list.append(s)

        else:
            _,rank=U.shape
            Sig = np.diagflat(s)

            Psi=Sig.dot(Vh)

            A_list.append( np.reshape( U, (l_dim,indices[i],rank)))
            Lambda_list.append( s)


    rank=len(s)
    A_list.append( np.reshape( Psi, (rank,indices[L-1],1)))

    return A_list,Lambda_list



## This is the old version of Vec_decomp_A
## should remove in the future
'''
def Vec_decomp_A( L, d, Ivec, Bond_dim=100):
#    Decompose a d^L dim Vector, Ivec, into MPS in
#    left-canonoical form with maximal dim, Bond_dim.

    A_list=[]
    Lambda_list=[]
    Psi=Ivec.reshape((1,len(Ivec)))

    for i in range(L-1):
        l_dim,r_dim = Psi.shape
        Psi = np.reshape(Psi,(l_dim*d, r_dim/d))

        U,s,Vh = linalg.svd(Psi,full_matrices=False)

        if len(s)>Bond_dim:
            U=U[:,:Bond_dim]
            print "Truncate !!!", sum(s[Bond_dim:]**2)
            s=s[:Bond_dim]
            #            s_2norm = linalg.norm(s,2)
            #            s = s/ s_2norm

            Sig = np.diagflat(s)
            Psi=Sig.dot(Vh[:Bond_dim,:])

            A_list.append( np.reshape( U, (l_dim,d,Bond_dim)))
            Lambda_list.append(s)

        else:
            _,rank=U.shape
            Sig = np.diagflat(s)

            Psi=Sig.dot(Vh)

            A_list.append( np.reshape( U, (l_dim,d,rank)))
            Lambda_list.append( s)


    rank=len(s)
    A_list.append( np.reshape( Psi, (rank,d,1)))

    return A_list,Lambda_list

'''




def Vec_decomp_A_tol(L, d, Ivec, tol, normalize=False):
    '''
    tol as relative tol, comparing to the first singular value

     Decompose a d^L dim Vector, Ivec, into MPS in left
     canonoical form with truncation at a tolerance, tol.
     '''

    A_list=[]
    Lambda_list=[]
    Psi=Ivec.reshape((1,len(Ivec)))

    for i in range(L-1):
        l_dim,r_dim = Psi.shape
        Psi = np.reshape(Psi,(l_dim*d, r_dim/d))

        U,s,Vh = linalg.svd(Psi,full_matrices=False)

        Bond_dim=len(s)
        for b_ind in range(len(s)):
            if s[b_ind]/s[0]<tol:
                Bond_dim=b_ind
                break

            else:
                pass

        if len(s)>Bond_dim:
            U=U[:,:Bond_dim]
            #            print "Truncate !!!", sum(s[Bond_dim:]**2)
            s=s[:Bond_dim]
            if normalize == True:
                s_2norm = linalg.norm(s,2)
                s = s/ s_2norm

            Sig = np.diagflat(s)
            Psi=Sig.dot(Vh[:Bond_dim,:])

            A_list.append( np.reshape( U, (l_dim,d,Bond_dim)))
            s=s[:Bond_dim]
            Lambda_list.append(s)

        else:
            _,rank=U.shape
            Sig = np.diagflat(s)

            Psi=Sig.dot(Vh)

            A_list.append( np.reshape( U, (l_dim,d,rank)))
            Lambda_list.append( s)

    rank=len(s)
    A_list.append( np.reshape( Psi, (rank,d,1)))

    return A_list,Lambda_list



def Mat_decomp_A( indices1, indices2, Mat, Bond_dim=100, normalize=False):
    '''
           2
           |
       1-- M --4
           |
           3
           
    '''
    M_list=[]
    L_list=[]
    dim1, dim2 = Mat.shape
    L=len(indices1)
    if L!=len(indices2):
        print "len indices1, indices2 mismatch!"
        raise Exception
    
    t_dim=1
    for i in range(L):
        t_dim=t_dim*indices1[i]
        
    if t_dim!=dim1:
        print "indices1, dim1 mismatch!"
        raise Exception
    
    t_dim=1
    for i in range(L):
        t_dim=t_dim*indices2[i]
        
    if t_dim!=dim2:
        print "indices2, dim2 mismatch!"
        raise Exception
    
    
    Psi=Mat.reshape((1,dim1,dim2))

    for i in range(L-1):
        l_dim, r_dim1, r_dim2 = Psi.shape
        
        Psi = np.reshape(Psi,(l_dim*indices1[i], r_dim1/indices1[i],
                              indices2[i], r_dim2/indices2[i]))

        Psi = np.einsum('ijkl->ikjl',Psi)
        Psi = np.reshape(Psi,(l_dim*indices1[i]*indices2[i],
                              r_dim1*r_dim2/indices1[i]/indices2[i]))
        
        U,s,Vh = linalg.svd(Psi,full_matrices=False)

        if len(s)>Bond_dim:
            U=U[:,:Bond_dim]
            print "Truncate !!!", sum(s[Bond_dim:]**2)
            s=s[:Bond_dim]
            if normalize == True:
                s_2norm = linalg.norm(s,2)
                s = s/ s_2norm

            Sig = np.diagflat(s)
            Psi=Sig.dot(Vh[:Bond_dim,:])

            Psi = np.reshape(Psi, (Bond_dim,
                                   r_dim1/indices1[i],r_dim2/indices2[i]))
            M_list.append( np.reshape( U, (l_dim,indices1[i],
                                           indices2[i],Bond_dim) ) )
            L_list.append(s)

        else:
            _,rank=U.shape
            Sig = np.diagflat(s)
            Psi=Sig.dot(Vh)
            
            Psi = np.reshape(Psi, (rank,
                                   r_dim1/indices1[i],r_dim2/indices2[i]))
            
            M_list.append( np.reshape( U, (l_dim,indices1[i],
                                           indices2[i],rank) ) )
            L_list.append( s)


    rank=len(s)
    M_list.append( np.reshape( Psi, (rank,indices1[L-1],
                                     indices2[L-1],1) ) )

    return M_list,L_list





def Mat_decomp_A_tol( indices1, indices2, Mat, tol=1e-5, normalize=False):
    '''
           2
           |
       1-- M --4
           |
           3
           
    '''
    M_list=[]
    L_list=[]
    dim1, dim2 = Mat.shape
    L=len(indices1)
    if L!=len(indices2):
        print "len indices1, indices2 mismatch!"
        raise Exception
    
    t_dim=1
    for i in range(L):
        t_dim=t_dim*indices1[i]
        
    if t_dim!=dim1:
        print "indices1, dim1 mismatch!"
        raise Exception
    
    t_dim=1
    for i in range(L):
        t_dim=t_dim*indices2[i]
        
    if t_dim!=dim2:
        print "indices2, dim2 mismatch!"
        raise Exception
    


    Psi=Mat.reshape((1,dim1,dim2))

    for i in range(L-1):
        l_dim, r_dim1, r_dim2 = Psi.shape
        
        Psi = np.reshape(Psi,(l_dim*indices1[i], r_dim1/indices1[i],
                              indices2[i], r_dim2/indices2[i]))

        Psi = np.einsum('ijkl->ikjl',Psi)
        Psi = np.reshape(Psi,(l_dim*indices1[i]*indices2[i],
                              r_dim1*r_dim2/indices1[i]/indices2[i]))
        
        U,s,Vh = linalg.svd(Psi,full_matrices=False)        

        Bond_dim=len(s)
        for b_ind in range(len(s)):
            if s[b_ind]/s[0]<tol:
                Bond_dim=b_ind
                break

            else:
                pass

        if len(s)>Bond_dim:
            U=U[:,:Bond_dim]
            print "Truncate !!!", sum(s[Bond_dim:]**2)
            s=s[:Bond_dim]
            if normalize == True:
                s_2norm = linalg.norm(s,2)
                s = s/ s_2norm

            Sig = np.diagflat(s)
            Psi=Sig.dot(Vh[:Bond_dim,:])

            Psi = np.reshape(Psi, (Bond_dim,
                                   r_dim1/indices1[i],r_dim2/indices2[i]))
            M_list.append( np.reshape( U, (l_dim,indices1[i],
                                           indices2[i],Bond_dim) ) )
            L_list.append(s)

        else:
            _,rank=U.shape
            Sig = np.diagflat(s)
            Psi=Sig.dot(Vh)
            
            Psi = np.reshape(Psi, (rank,
                                   r_dim1/indices1[i],r_dim2/indices2[i]))
            
            M_list.append( np.reshape( U, (l_dim,indices1[i],
                                           indices2[i],rank) ) )
            L_list.append( s)


    rank=len(s)
    M_list.append( np.reshape( Psi, (rank,indices1[L-1],
                                     indices2[L-1],1) ) )

    return M_list,L_list






def to_tensor_index_bi(Largeindex,L):
    '''
    Express Largeindex in L bit
    '''
    return [Largeindex >> i & 1 for i in range(L-1,-1,-1)]


def gen_tensor_index(dim,ind_dim):
    '''
 Generate all possible indices
    '''
    ind_list=[]
    ind_arr=np.zeros(len(ind_dim),dtype=int)
    ind_list.append(ind_arr.copy())
    for i in xrange(dim-1):
        ind_arr[-1]=ind_arr[-1]+1
        for j in range(1,len(ind_dim)):
            if ind_arr[-j]/ind_dim[-j]==1:
                ind_arr[-j]=0
                ind_arr[-(j+1)]=ind_arr[-(j+1)]+1
            else:
                pass

        ind_list.append(ind_arr.copy())
        
    return ind_list



'''
def A_to_Vec(L,d,A_list):
    Vec = np.zeros(d**L)
    for ind in range(d**L):

        binary_index=gen_index(ind,L)
        val=A_list[0][:,binary_index[0],:]
        for ind_j in range(1,L):
            val=np.einsum('ij,jl->il',val,A_list[ind_j][:,binary_index[ind_j],:])

        Vec[ind]=val

    return Vec
'''



def A_to_Vec(ind_dim,A_list):
    dim=1
    L=len(A_list)
    for i in range(L):
        dim=dim*ind_dim[i]
        
    ind_list=gen_tensor_index(dim,ind_dim)
    Vec = np.zeros(dim)
    
    for i in range(dim):
        indices=ind_list[i]
        val=A_list[0][:,indices[0],:]
        for j in range(1,L):
            val=np.einsum('ij,jl->il',val,A_list[j][:,indices[j],:])

        Vec[i]=val

    return Vec


def M_to_Mat(ind_dim1,ind_dim2,M_list):
    L=len(M_list)
    dim1=1
    for i in range(L):
        dim1=dim1*ind_dim1[i]

    dim2=1
    for i in range(L):
        dim2=dim2*ind_dim2[i]

    Mat=np.zeros((dim1,dim2))
    ind1_list=gen_tensor_index(dim1,ind_dim1)
    ind2_list=gen_tensor_index(dim2,ind_dim2)
    for i in range(dim1):
        for j in range(dim2):
            indices1=ind1_list[i]
            indices2=ind2_list[j]
            val=M_list[0][:,indices1[0],indices2[0],:]
            for k in range(1,L):
                val=np.einsum('ij,jl->il',val,
                              M_list[k][:,indices1[k],indices2[k],:])

            Mat[i,j]=val

    return Mat





def MPS_dot(MPS1,MPS2):
    ''' Return the inner product between two MPS
    '''
    L=len(MPS1)
    MPS_cont = np.einsum('ijk,ijl->kl',MPS1[0],MPS2[0])
    for i in range(1,L):
        MPS_cont = np.einsum('ij,ikl->jkl',MPS_cont,MPS1[i])
        MPS_cont = np.einsum('ijk,ijl->kl',MPS_cont,MPS2[i])

    return MPS_cont[0,0]





def MPS_dot_to_L(MPS1,MPS2,l):
    '''
    staring from the Left
    contracting to l-th varaible (not including l-th)
    index staring from 0,1, 2, ... ,l-1,l,...L-1
    returning a rank-2 tensor

    i MPS1 k
        j
        j
    i MPS2 l

    '''
    if l==0:
        return np.ones((1,1))

    MPS_cont = np.einsum('ijk,ijl->kl',MPS1[0],MPS2[0])
    for i in range(1,l):
        MPS_cont = np.einsum('ij,ikl->jkl',MPS_cont,MPS1[i])
        MPS_cont = np.einsum('ijk,ijl->kl',MPS_cont,MPS2[i])

    return MPS_cont





def MPS_dot_to_R(MPS1,MPS2,l):
    '''
    staring from the Right
    contracting to l-th varaible (not including l-th)
    index staring from 0,1, 2, ... ,l, ..., L-1
    returning a rank-2 tensor
    '''
    L=len(MPS1)
    MPS_cont = np.einsum('ijk,ljk->il',MPS1[L-1],MPS2[L-1])
    for i in range(L-2,l,-1):
        MPS_cont = np.einsum('ij,kli->klj',MPS_cont,MPS1[i])
        MPS_cont = np.einsum('klj,mlj->km',MPS_cont,MPS2[i])

    return MPS_cont




def MPS_copy(MPS):
    '''
    Return a copy by list slicing
    because the MPS is store in python list
    Caution! it is not the same for np.array
    '''
    return MPS[:]


def MPS_right_canonicalize(MPS):
    Lambda_list=[]
    L=len(MPS)
    for ind in range(L-1,0,-1):
        l_dim,d,r_dim = MPS[ind].shape
        U,s,Vh = linalg.svd(np.reshape(MPS[ind],(l_dim, d*r_dim))
                            ,full_matrices=False)
        rank=len(s)
        MPS[ind]= np.reshape(Vh, (rank,d,r_dim))
        Lambda_list.append( s)
        Sig = np.diagflat(s)
        MPS[ind-1]=np.einsum('ijk,kl->ijl',MPS[ind-1],U.dot(Sig))

    rank=len(s)
    MPS[0]=( np.reshape( MPS[0], (1,d,rank)))
    return


def MPS_left_canonicalize(MPS):
    Lambda_list=[]
    L=len(MPS)
    for ind in range(L-1):
        l_dim,d,r_dim = MPS[ind].shape
        U,s,Vh = linalg.svd(np.reshape(MPS[ind],(l_dim*d, r_dim))
                            ,full_matrices=False)
        rank=len(s)
        MPS[ind]= np.reshape(U, (l_dim,d,rank))
        Lambda_list.append( s)
        Sig = np.diagflat(s)
        MPS[ind+1]=np.einsum('ij,jkl->ikl',Sig.dot(Vh),MPS[ind+1])

    rank=len(s)
    MPS[L-1]=( np.reshape( MPS[L-1], (rank,d,1)))
    return


def MPS_compression_svd(MPS,Bond_dim,tol=1e-3):
    MPS_right_canonicalize(MPS)
    Lambda_list=[]
    L=len(MPS)
    for ind in range(L-1):
        l_dim,d,r_dim = MPS[ind].shape
        try:
            U,s,Vh = linalg.svd(np.reshape(MPS[ind],(l_dim*d, r_dim))
                                ,full_matrices=False)

        except:
            print "conv err, at compression_svd, ind=", ind
            return

        if len(s)>Bond_dim:
            bond_dim=Bond_dim
            if sum(s[bond_dim:]**2)>tol:
                #print "Trun_at_svd_comp:", sum(s[bond_dim:]**2)
                pass

#            print "Truncate !!!", sum(s[bond_dim:]**2)

        else:
            bond_dim=len(s)

        U=U[:,:bond_dim]
        s=s[:bond_dim]
        MPS[ind]=np.reshape(U, (l_dim,d,bond_dim))
        Lambda_list.append(s)
        Sig = np.diagflat(s)
        MPS[ind+1]=np.einsum('ij,jkl->ikl',Sig.dot(Vh[:bond_dim,:]),
                             MPS[ind+1])

    return

def MPS_compression_iter(MPS , MPS_trial ,max_iter=5):
    ## Iterative compression on MPS
    ## trail ansatz MPS_trial is required
    ## Modification on MPS_trial directly
    ## assume MPS_trial to be left canonical
    L=len(MPS)
    tol=1e-4
    conv=1
    err_old=1
    err_new=1
    count=0
    while(conv >tol ):
        count = count+1
        if count>max_iter:
            break

        ############################
        ### Sweeping to the left ###
        ############################

        ### Calculate Left and Right
        Left=MPS_dot_to_L(MPS_trial,MPS,L-1)
        ### update M
        MPS_trial[L-1]=np.einsum('ij,jkl->ikl',Left,MPS[L-1])
        ### AA...AM --> AA...AUSB
        l_dim, d,r_dim = MPS_trial[L-1].shape
        U,s,Vh = linalg.svd(MPS_trial[L-1].reshape((l_dim,d*r_dim)),
                            full_matrices=False)
        rank=len(s)
        MPS_trial[L-1]=np.reshape( Vh, (rank,d,r_dim))
        Sig = np.diagflat(s)
        ### AA...AUSB --> AA...AMB
        MPS_trial[L-2]=np.einsum('ijk,kl->ijl',MPS_trial[L-2],U.dot(Sig))

        ### Calculate Left and Right
        for i in range(L-2,0,-1): ## iter from L-2 to 1
            Left  = MPS_dot_to_L(MPS_trial,MPS,i)
            Right = MPS_dot_to_R(MPS_trial,MPS,i)
            ### update M            
            MPS_trial[i]=np.einsum('ij,jkl->ikl',Left,
                                   np.einsum('ijk,lk->ijl',MPS[i],Right))
            ### AA...AMB --> AA...AUSBB
            l_dim, d,r_dim = MPS_trial[i].shape
            U,s,Vh = linalg.svd(MPS_trial[i].reshape((l_dim,d*r_dim)),
                                full_matrices=False)
            rank=len(s)
            MPS_trial[i]=np.reshape( Vh, (rank,d,r_dim))
            Sig = np.diagflat(s)
            MPS_trial[i-1]=np.einsum('ijk,kl->ijl',MPS_trial[i-1],U.dot(Sig))

        ############################
        ### Sweeping to the Right ###
        ############################

        ### Calculate Left and Right
        Right=MPS_dot_to_R(MPS_trial,MPS,0)
        ### update M_trial
        MPS_trial[0]=np.einsum('ijk,lk->ijl',MPS[0],Right)
        ### MB...BB --> ASVB...BB
        l_dim, d,r_dim = MPS_trial[0].shape
        U,s,Vh = linalg.svd(MPS_trial[0].reshape((l_dim*d,r_dim)),
                            full_matrices=False)
        rank=len(s)
        MPS_trial[0]=np.reshape( U, (l_dim,d,rank))
        Sig = np.diagflat(s)
        ### ASVB...BB --> AM...BB
        MPS_trial[1]=np.einsum('ij,jkl->ikl',Sig.dot(Vh),MPS_trial[1])

        ### Calculate Left and Right
        for i in range(1,L-1): ## iter from to L-2
            Left  = MPS_dot_to_L(MPS_trial,MPS,i)
            Right = MPS_dot_to_R(MPS_trial,MPS,i)
            ### update M            
            MPS_trial[i]=np.einsum('ij,jkl->ikl',Left,
                                   np.einsum('ijk,lk->ijl',MPS[i],Right))
            ### AMB...BB --> AASVB...BB
            l_dim, d,r_dim = MPS_trial[i].shape
            U,s,Vh = linalg.svd(MPS_trial[i].reshape((l_dim*d,r_dim)),
                                full_matrices=False)
            rank=len(s)
            MPS_trial[i]=np.reshape( U, (l_dim,d,rank))
            Sig = np.diagflat(s)
            ### AASVB...BB --> AAM...BB
            MPS_trial[i+1]=np.einsum('ij,jkl->ikl',Sig.dot(Vh),MPS_trial[i+1])

        ##Update err
        err_old=err_new
        err_new=1-MPS_dot(MPS_trial,MPS_trial)
        conv=abs(err_new-err_old)

        print "conv: ",conv, "err: ",err_new
        ##        print MPS_dot(MPS_trial,MPS),MPS_dot(MPS_trial,MPS_trial), err_new


    return



def MPS_addition(MPS1,MPS2):
    L=len(MPS1)
    Final_MPS=[]
    ## M[0] = [M1[0] M2[0]]
    M1_l_dim,d,M1_r_dim = MPS1[0].shape
    M2_l_dim,d,M2_r_dim = MPS2[0].shape
    Final_MPS.append(  np.zeros( (M1_l_dim,d,M1_r_dim+M2_r_dim) ) )
    for d_ind in range(d):
        Final_MPS[0][:,d_ind,:M1_r_dim]     =MPS1[0][:,d_ind,:]
        Final_MPS[0][:,d_ind,M1_r_dim : ]   =MPS2[0][:,d_ind,:]

    for i in range(1,L-1):
        ## M[i] = [ M1[i], 0
        ##           0   , M2[i] ]
        M1_l_dim,d,M1_r_dim = MPS1[i].shape
        M2_l_dim,d,M2_r_dim = MPS2[i].shape
        Final_MPS.append(  np.zeros( (M1_l_dim+M2_l_dim,d,M1_r_dim+M2_r_dim) ) )
        for d_ind in range(d):
            Final_MPS[i][:M1_l_dim, d_ind , :M1_r_dim] = MPS1[i][:,d_ind,:]
            Final_MPS[i][M1_l_dim:, d_ind , M1_r_dim:] = MPS2[i][:,d_ind,:]

    ## M[L-1] = [   M1[L-1]
    ##              M2[L-1] ]
    M1_l_dim,d,M1_r_dim = MPS1[L-1].shape
    M2_l_dim,d,M2_r_dim = MPS2[L-1].shape
    Final_MPS.append(  np.zeros( (M1_l_dim+M2_l_dim,d,M1_r_dim) ) )
    for d_ind in range(d):
        Final_MPS[L-1][:M1_l_dim,d_ind,:] =   MPS1[L-1][:,d_ind,:]
        Final_MPS[L-1][M1_l_dim:,d_ind,:] =   MPS2[L-1][:,d_ind,:]


    return Final_MPS


def num_para(MPS):
    n=0
    for i in MPS:
        n=n+i.size

    return n



def MPO_to_MPS(MPO):
    ''' reshape MPO in MPS formulation'''
    MPS=[]
    for i in range(len(MPO)):
        l_dim,d1,d2,r_dim=MPO[i].shape
        MPS.append(MPO[i].reshape((l_dim,d1*d2,r_dim)))

    return MPS


def MPS_to_MPO(MPS,ind1,ind2):
    ''' reshape MPS to MPO formultation'''
    MPO=[]
    for i in range(len(MPS)):
        l_dim,d,r_dim=MPS[i].shape
        if ind1[i]*ind2[i] != d:
            print "Error, dim mismatch"
            raise Exception
        
        else:
            MPO.append(MPS[i].reshape((l_dim,ind1[i],ind2[i],r_dim)))

    return MPO




