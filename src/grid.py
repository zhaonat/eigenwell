'''
    we have seperate operators for omega to k
    vs k to omega
'''
import numpy as np
import scipy.sparse as sp

class FiniteDifferenceGrid():
    '''
        operations on a yee grid, such as averaging, derivatives, etc.
        works for 1d and 2d since we have TE and TM seperation
        fdfd sets up: 1 the matrix operator A:
        which may differ if it's a k to omega or omega to k solve
        A = dxf*dxb+dyf*dyb+omega^2 (mu_0, eps_r)
    '''
    def __init__(self, dL,N):
        '''
            for 1d, use N = [Nx,1]
            dL: [dx, dy] discretization
            eps_r: pixel-by-pixel Nx x Ny structure of relative permittivities
            omega: frequency
        '''
        assert (len(dL)==len(N) == 2, 'must specify 2 elem arr even for 1d sims')
        assert 0 not in dL and 0 not in N, 'do not make anything 0 in dL or N'
        self.dL = dL;
        self.N = N;
        self.make_derivatives();
        return;

    def createDws(self, s, f):
        '''
            s = 'x' or 'y': x derivative or y derivative
            f = 'b' or 'f'
            catches exceptions if s and f are misspecified
        '''
        M = np.prod(self.N);
        Nx = self.N[0];
        Ny = self.N[1];

        sign = 1 if f == 'f' else -1;
        dw = None; #just an initialization
        indices = np.reshape(np.arange(M), (Nx,Ny), order = 'F');
        if(s == 'x'):
            ind_adj = np.roll(indices, -sign, axis = 0)
            dw = self.dL[0]
        elif(s == 'y'):
            ind_adj = np.roll(indices, -sign, axis = 1)
            dw = self.dL[1];

        # we could use flatten here since the indices are already in 'F' order
        indices_flatten = np.reshape(indices, (M, ), order = 'F')
        indices_adj_flatten = np.reshape(ind_adj, (M, ), order = 'F')
        # on_inds = np.hstack((indices.flatten(), indices.flatten()))
        # off_inds = np.concatenate((indices.flatten(), ind_adj.flatten()), axis = 0);
        on_inds = np.hstack((indices_flatten, indices_flatten));
        off_inds = np.concatenate((indices_flatten, indices_adj_flatten), axis = 0);

        all_inds = np.concatenate((np.expand_dims(on_inds, axis =1 ), np.expand_dims(off_inds, axis = 1)), axis = 1)

        data = np.concatenate((-sign*np.ones((M)), sign*np.ones((M))), axis = 0)
        Dws = sp.csc_matrix((data, (all_inds[:,0], all_inds[:,1])), shape = (M,M));

        return (1/dw)*Dws;

    def createDws_bloch(self, s, f, k = 0, L = 0):
        M = np.prod(self.N);
        Nx = self.N[0];
        Ny = self.N[1];

        sign = 1 if f == 'f' else -1;
        bloch_term = sign*np.exp(-sign*1j*k*L);

        indices = np.reshape(np.arange(M), (Nx,Ny), order = 'F');
        if(s == 'x'):
            ind_adj = np.roll(indices, -sign, axis = 0)
            dw = self.dL[0]
            threshold = 1;
        elif(s == 'y'):
            ind_adj = np.roll(indices, -sign, axis = 1)
            dw = self.dL[1];
            threshold = Ny;

        off_diag = (sign/dw)*np.ones(self.N).astype('complex');
        on_diag = -(sign/dw)*np.ones(self.N).astype('complex');
        off_diag[np.abs(indices-ind_adj) > threshold] = (1/dw)*bloch_term;
        on_diag = np.reshape(on_diag, (M,),order = 'F')
        off_diag = np.reshape(off_diag, (M,),order = 'F')

        indices_flatten = np.reshape(indices, (M, ), order = 'F')
        indices_adj_flatten = np.reshape(ind_adj, (M, ), order = 'F')
        # on_inds = np.hstack((indices.flatten(), indices.flatten()))
        # off_inds = np.concatenate((indices.flatten(), ind_adj.flatten()), axis = 0);
        on_inds = np.hstack((indices_flatten, indices_flatten));
        off_inds = np.concatenate((indices_flatten, indices_adj_flatten), axis = 0);

        all_inds = np.concatenate((np.expand_dims(on_inds, axis =1 ), np.expand_dims(off_inds, axis = 1)), axis = 1)

        data = np.concatenate((on_diag, off_diag), axis = 0)
        Dws = sp.csc_matrix((data, (all_inds[:,0], all_inds[:,1])), shape = (M,M));
        return Dws;

    ## overloaded operators (can we have multiple functions with the same name?)
    # python does not natively support overloading
    def make_derivatives(self):
        self.Dxf = self.createDws('x', 'f');
        self.Dyf = self.createDws('y', 'f');
        self.Dxb = self.createDws('x', 'b');
        self.Dyb = self.createDws('y', 'b');


class NonUniformGrid(FiniteDifferenceGrid):
    def __init__(self, dL,N):
        super().__init__(dL,N); #base fd grid;
        self.N = N;

    def non_uniform_operator(self):
        dx_scale, dy_scale = generate_nonuniform_scaling();

        [Xs, Ys] = np.meshgrid(dx_scale, dy_scale);
        #meshgrid isn't right for y
        M = np.prod(Xs.shape)

        # we have to this kind of flip because the flattening
        # operation (:) doesn't retain row-major order
        Ys=Ys.T; Xs = Xs.T;
        Fsy = sp.spdiags(Ys.flatten(),0,M,M);
        Fsx = sp.spdiags(Xs.flatten(),0,M,M);

        # might as well construct the conjugate grid. What is the conjugate grid?
        xc = (dx_scale+np.roll(dx_scale,[0,1]))/2;
        yc = (dy_scale+np.roll(dy_scale,[0,1]))/2;

        [Xc, Yc] = np.meshgrid(xc, yc);
        Xc = Xc.T;
        Yc = Yc.T;
        Fsy_conj = sp.spdiags(Yc.flatten(),0,M,M);
        Fsx_conj = sp.spdiags(Xc.flatten(),0,M,M);
        return Fsx, Fsy, Fsx_conj, Fsy_conj;


    @staticmethod
    def generate_nonuniform_scaling(Nft, drt):
        '''
            this method should be used by the user?
            best way to parametrize this? use a dictionary or mask
            Nft: 1st column is x, 2nd column is y
            #sizes of all regions with the
            [coarse, transition, fine, transition, coarse]
            drt: list of discretizations...normalized by some reference
        '''
        Nx = np.sum(Nft[:,0]);
        Ny = np.sum(Nft[:,1]);
        dx_scale = np.ones(Nx)
        dy_scale = np.ones(Ny);

        num_regions = Nft.shape[0]; #iterate through 0,2,4
        x0 = y0 = 0;
        for i in range(0,num_regions,2):
            dx_scale[x0:x0+Nft[i,0]] = drt[i,0];
            dy_scale[y0:y0+Nft[i,1]] = drt[i,1];
            if(i==num_regions-1): #%no transition after last region
                x0 = x0+Nft[i,0];
                y0 = y0+Nft[i,1];
            else:
                x0 = x0+Nft[i,0]+Nft[i+1,0];
                y0 = y0+Nft[i,1]+Nft[i+1,1];


        x0 = Nft[1,0]; y0 = Nft[1,1];
        for i in range(1, num_regions,2): #2:2:num_regions
            dx1 = drt[i-1,0]; dx2 = drt[i+1,0];
            dy1 = drt[i-1,1]; dy2 = drt[i+1,1];
            nxt = Nft[i,0]; nyt = Nft[i,1];

            grading_x = np.logspace(np.log10(dx1), np.log10(dx2), nxt+1);
            grading_y = np.logspace(np.log10(dy1), np.log10(dy2), nyt+1);

            dx_scale[x0-1:x0+nxt] = grading_x;
            dy_scale[y0-1:y0+nyt] = grading_y;

            x0 = x0+Nft[i,0]+Nft[i+1,0];
            y0 = y0+Nft[i,1]+Nft[i+1,1];
