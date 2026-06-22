import torch
import torch.nn as nn
import numpy as np


def givens_3x3(i, j, theta, phi):
    """
    Genera matriz SU(2) embebida en niveles (i,j) dentro de SU(3).
    theta, phi pueden ser escalares o tensores [B].
    Retorna matriz [B,3,3].
    """
    batch = theta.shape[0]
    U = torch.eye(3, dtype=torch.complex64).unsqueeze(0).repeat(batch, 1, 1)
 
    ct = torch.cos(theta/2)
    st = torch.sin(theta/2)
 
    eiphi  = torch.exp(1j * phi)
    eimphi = torch.exp(-1j * phi)
 
    U[:, i, i] = ct
    U[:, j, j] = ct
    U[:, i, j] = -eiphi * st
    U[:, j, i] =  eimphi * st
 
    return U
 
def diag_phase(alpha1, alpha2):
    """
    Crea D = diag(e^{i a1}, e^{i a2}, e^{-i(a1+a2)}).
    Soporta batch.
    """
    batch = alpha1.shape[0]
    a3 = -(alpha1 + alpha2)
 
    d0 = torch.exp(1j * alpha1)
    d1 = torch.exp(1j * alpha2)
    d2 = torch.exp(1j * a3)
 
    D = torch.zeros(batch, 3, 3, dtype=torch.complex64)
    D[:,0,0] = d0
    D[:,1,1] = d1
    D[:,2,2] = d2
    return D
 
 
class QNN_0_Givens_Lz(nn.Module):
 
    def __init__(self, num_layers, num_features, num_params=None, init_weights=None):
        super().__init__()
 
        self.num_layers   = num_layers
        self.num_features = num_features
 
        # ------------------------------------------------
        # ESTADOS BASE |0>, |1>, |2>
        # ------------------------------------------------
        q0 = torch.tensor([[1],[0],[0]], dtype=torch.complex64)
        q1 = torch.tensor([[0],[1],[0]], dtype=torch.complex64)
        q2 = torch.tensor([[0],[0],[1]], dtype=torch.complex64)
 
        self.register_buffer("q0", q0)
        self.register_buffer("q1", q1)
        self.register_buffer("q2", q2)
        self.register_buffer("psi0", q0)
 
        # ------------------------------------------------
        # Lz Observable
        # ------------------------------------------------
        Lz = torch.tensor([[ 1., 0., 0.],
                           [0., 0., 0.],
                           [ 0., 0., -1.]
                           ], dtype=torch.complex64)
        self.register_buffer("Lz", Lz)
 
        # ------------------------------------------------
        # VARIATIONAL LAYERS
        # ------------------------------------------------
        self.num_var_params = 8
        self.weights = nn.ParameterList()
 
        for l in range(num_layers):
            if init_weights is not None:
                base = init_weights[l][:self.num_var_params]
            else:
                base = (torch.rand(self.num_var_params) * 2 - 1)
            self.weights.append(nn.Parameter(base))
 
 
        # ------------------------------------------------
        # ENCODING MAP
        # ------------------------------------------------
        enc_order = ["theta01", "phi01", "theta12", "phi12",
             "theta02", "phi02", "alpha1", "alpha2"]
        enc_full  = enc_order * ((num_features // 8) + 1)
        self.enc_sequence = enc_full[:num_features]
 
    # =====================================================
    #                    FORWARD PASS
    # =====================================================
    def forward(self, batch):
        batch_size = batch.shape[0]
        batch_c    = batch.to(torch.complex64)
 
        # Estado inicial para todo el batch
        state = self.psi0.expand(batch_size, 3, 1).clone()
 
        # ------------------------------------
        # LOOP DE LAYERS
        # ------------------------------------
        for layer in range(self.num_layers):
 
            # ============================================
            # 1) ENCODING SU(3) COMPLETO POR SAMPLE
            # ============================================
            
            params = batch_c
            
            # Convertimos features a 8 parámetros
            if params.shape[1] < 8:
                pad = torch.zeros(batch_size, 8 - params.shape[1],
                                  dtype=torch.complex64, device=batch.device)
                params = torch.cat([params, pad], dim=1)
            else:
                params = params[:, :8]  # usamos solo 8
            
 
            # Model 2 ----------------------------------
            x1, x2 = params[:,0], params[:,1]
            x3, x4 = params[:,2], params[:,3]
            x5, x6 = params[:,4], params[:,5]
            x7, x8 = params[:,6], params[:,7]
            
            D    = diag_phase(x1, x2)
            U01a = givens_3x3(0,1, x3, x4)
            U12  = givens_3x3(1,2, x5, x6)
            U01b = givens_3x3(0,1, x7, x8)
            
            U_enc = U01b @ U12 @ U01a @ D
            state = torch.bmm(U_enc, state)
 
            # ============================================
            # 2) VARIATIONAL BLOCK (SU(3) UNIVERSAL)
            # ============================================
            t01, p01, t12, p12, t02, p02, a1, a2 = self.weights[layer]
 
            # Expand to batch
            t01 = t01.expand(batch_size)
            p01 = p01.expand(batch_size)
            t12 = t12.expand(batch_size)
            p12 = p12.expand(batch_size)
            t02 = t02.expand(batch_size)
            p02 = p02.expand(batch_size)
            a1  = a1.expand(batch_size)
            a2  = a2.expand(batch_size)
 
 
            # Model 2 ----------------------------------
            U01a = givens_3x3(0,1, t01, p01)
            U12  = givens_3x3(1,2, t12, p12)
            U01b = givens_3x3(0,1, t02, p02)
            D    = diag_phase(a1, a2)
 
            Uvar = U01b @ U12 @ U01a @ D
            state = torch.bmm(Uvar, state)
 
        # ====================================================
        # DENSITY MATRIX
        # ====================================================
        rho = state @ state.conj().transpose(-2, -1)
        rho = (rho + rho.conj().transpose(-2, -1)) / 2
 
        # ====================================================
        # EXPECTATION VALUE OF Lz
        # ====================================================
        Lz_batch = self.Lz.unsqueeze(0).expand(batch_size, -1, -1)
        exp_Lz = torch.real(torch.einsum("bij,bij->b", rho, Lz_batch))
 
        return exp_Lz  # shape [B]