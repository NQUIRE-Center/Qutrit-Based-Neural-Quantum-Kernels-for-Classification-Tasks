import torch
import torch.nn as nn
import numpy as np


class QNN_0_Euler_Lz(nn.Module):
    
    def __init__(self, num_layers, num_features, num_params=None, init_weights=None):
        super(QNN_0_Euler_Lz, self).__init__()
        self.num_layers   = num_layers
        self.num_features = num_features
        
        # -----------------------------
        # Qutrit basis
        # -----------------------------
        q0 = torch.tensor([[1], [0], [0]], dtype=torch.complex64)
        q1 = torch.tensor([[0], [1], [0]], dtype=torch.complex64)
        q2 = torch.tensor([[0], [0], [1]], dtype=torch.complex64)
        self.register_buffer("q0", q0)
        self.register_buffer("q1", q1)
        self.register_buffer("q2", q2)

        # Outer product helper
        gm = lambda A, B: torch.kron(A, B.T)
        
        # Label projectors
        self.register_buffer("label_ops", torch.stack([gm(q, q) for q in [q0, q1, q2]]))
        
        # Gell-Mann generators
        gm1 = (gm(q0, q1) + gm(q1, q0)).to(torch.complex64)
        gm2 = (-1j * (gm(q0, q1) - gm(q1, q0))).to(torch.complex64)
        gm3 = (gm(q0, q0) - gm(q1, q1)).to(torch.complex64)
        gm4 = (gm(q0, q2) + gm(q2, q0)).to(torch.complex64)
        gm5 = (-1j * (gm(q0, q2) - gm(q2, q0))).to(torch.complex64)
        gm6 = (gm(q1, q2) + gm(q2, q1)).to(torch.complex64)
        gm7 = (-1j * (gm(q1, q2) - gm(q2, q1))).to(torch.complex64)
        gm8 = (1 / torch.sqrt(torch.tensor(3., dtype=torch.float32)) * (gm(q0, q0) + gm(q1, q1) - 2 * gm(q2, q2))).to(torch.complex64)


        # Observable Lz = diag(2, 0, -2)
        Lz = torch.tensor([[ 1., 0., 0.],
                           [ 0., 0., 0.],
                           [ 0., 0., -1.]], dtype=torch.complex64)
        self.register_buffer("Lz", Lz)

        # -----------------------------
        # ENCODING generators (data reuploading)
        # -----------------------------
        
        enc_generators = [gm8, gm3, gm2, gm3, gm5, gm3, gm2, gm3]
        enc_list = [enc_generators[j % len(enc_generators)] for j in range(num_features)]
        self.register_buffer("gens_enc", torch.stack(enc_list))

        # -----------------------------
        # VARIATIONAL block: 
        # -----------------------------
        
        var_generators = [gm8, gm3, gm2, gm3, gm5, gm3, gm2, gm3]
        self.register_buffer("gens_var", torch.stack(var_generators)) 
        self.num_var_params = 8
        
        # -----------------------------
        # Parameters θ per layer (4 por capa)
        # -----------------------------
        self.weights = nn.ParameterList()
        for l in range(num_layers):
            if init_weights is not None:
                # si init_weights[l] tiene más de 4, nos quedamos con los primeros 4
                base = init_weights[l][:self.num_var_params]
            else:
                base = torch.rand(self.num_var_params) * 2 - 1
            self.weights.append(nn.Parameter(base))

        # initial state |0>
        self.register_buffer("psi0", q0)

    # ---------------------------------------------------------------
    def forward(self, batch):
        batch_size = batch.shape[0]
        batch_c = batch.to(torch.cfloat)

        # |0> para todo el batch
        state = self.psi0.expand(batch_size, -1, -1).clone()  # [B, 3, 1]

        for i in range(self.num_layers):
            # -------------------------
            # Encoding: U_enc(x)
            # -------------------------
            for j in range(self.num_features):
                G = self.gens_enc[j]
                x = batch_c[:, j].view(-1, 1, 1)
                U = torch.matrix_exp(1j * x * G)
                state = torch.bmm(U, state)

            # -------------------------
            # Variational block
            # -------------------------
            thetas = self.weights[i]                       
            for k, G in enumerate(self.gens_var):          
                U = torch.matrix_exp(1j * thetas[k] * G)   # [3,3] (broadcast)
                state = torch.matmul(U, state)             # [B,3,1]

        # ====================================================
        # DENSITY MATRIX
        # ====================================================
        rho = state @ state.conj().transpose(-2, -1)
        rho = (rho + rho.conj().transpose(-2, -1)) / 2  # ensure Hermitian

        # ====================================================
        # EXPECTATION VALUE OF Lz
        # ====================================================
        Lz_batch = self.Lz.unsqueeze(0).expand(batch_size, -1, -1)
        exp_Lz = torch.real(torch.einsum("bij,bij->b", rho, Lz_batch))
 
        return exp_Lz  # shape [B]