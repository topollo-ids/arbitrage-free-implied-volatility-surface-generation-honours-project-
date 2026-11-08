"""
Verbatim copies of functions from the ORIGINAL VolGAN.py, used only as a reference in tests.
VolGAN.py itself cannot be imported on a current install (yfinance / SciPy imports fail, B11),
so the needed functions are extracted here unchanged. Source line numbers are noted per block.
"""

import numpy as np
from scipy.stats import norm


# ---- VolGAN.py lines 66-87 ----
def penalty_mutau(mu,T):
    """
    Same as penalty K, T, but with moneyness instead
    """
    P_T = np.zeros((len(T),len(T)))
    P_K = np.zeros((len(mu),len(mu)))
    PB_K = np.zeros((len(mu),len(mu)))
    #P_T first, the last one is zero
    for j in np.arange(0,len(T)-1,1):
        P_T[j,j] = T[j]/(T[j+1]-T[j])
        P_T[j+1,j] = -T[j]/(T[j+1]-T[j])
    #now P_K and then PB_K
    for i in np.arange(0,len(mu)-1,1):
        P_K[i,i] = -1/(mu[i+1]-mu[i])
        P_K[i,i+1] = 1/(mu[i+1]-mu[i])
    #PB_K: note that it is a scaled finite difference, but let's compute it on its own just in case
    #once we fix the grid we have to run this function only once so it doesn't matter much
    for i in np.arange(1,len(mu)-1,1):
        PB_K[i,i-1] = -(mu[i+1]-mu[i]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
        PB_K[i,i] = (mu[i+1] - mu[i-1]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
        PB_K[i,i+1] = -(mu[i]-mu[i-1]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
    return P_T,P_K,PB_K

# ---- VolGAN.py lines 116-130 ----
def arbitrage_penalty(C,P_T,P_K,PB_K):
    """
    Given the prices of calls C for a fixed grid (K,T)
    P_T, P_K, PB_K: the matrices calculated by penalty_matrices
    Returns matrix penalties (for each point) in order
    1) penalty for violating C being increasing in T
    2) penalty for violating C being decreasing in K
    3) penalty for violating C being convex in K
    plus
    4) the sum of all penalties together (scalar)
    """
    P1 = np.maximum(0,np.matmul(C,P_T))
    P2 = np.maximum(0,np.matmul(P_K,C))
    P3 = np.maximum(0,np.matmul(PB_K,C))
    return P1,P2,P3,np.sum(P1+P2+P3)

# ---- VolGAN.py lines 201-208 ----
def smallBS(m,tau,sigma,r):
    """
    Relative call
    """
    d1 = (-np.log(m)+tau*(r+0.5*sigma*sigma))/(sigma*np.sqrt(tau))
    d2 = d1-sigma*np.sqrt(tau)
    price = norm.cdf(d1)-m*norm.cdf(d2)*np.exp(-r*tau)
    return price


def original_preprocessing(surfaces_transform, log_rtn):
    """
    The array-building part of the original DataPreprocesssing, with its body copied verbatim from
    VolGAN.py lines 544-547, 550-558 and 585-589. Left out: the dates line (needs the loader) and the
    dormant normalisation statistics (lines 560-583, B9). Takes the vols as decimals and the log-returns.
    """
    #realised volatility at time t-1
    realised_vol_tm1 = np.zeros(len(log_rtn)-22)
    for i in range(len(realised_vol_tm1)):
        realised_vol_tm1[i] = np.sqrt(252 / 21) * np.sqrt(np.sum(log_rtn[i:(i+21)]**2))
    #log-return at t, t-1, t-2
    log_rtn_t = log_rtn[22:]
    log_rtn_tm1 = np.sqrt(252) * log_rtn[21:-1]
    log_rtn_tm2 = np.sqrt(252) * log_rtn[20:-2]
    #log implied vol at t and t-1
    log_iv_t = np.log(surfaces_transform[22:])
    log_iv_tm1 = np.log(surfaces_transform[21:-1])
    #we want to simulate the increment at time t (t - t-1)
    log_iv_inc_t = log_iv_t - log_iv_tm1
    #condition for generator and discriminator
    condition = np.concatenate((np.expand_dims(log_rtn_tm1,axis=1),np.expand_dims(log_rtn_tm2,axis=1),np.expand_dims(realised_vol_tm1,axis=1),log_iv_tm1),axis=1)
    #true: what we are trying to predict, increments at time t
    log_rtn_t_ann = np.sqrt(252) * log_rtn_t
    true = np.concatenate((np.expand_dims(log_rtn_t_ann,axis=1),log_iv_inc_t),axis=1)
    return true, condition


# ---- VolGAN.py lines 451-536 (Generator and Discriminator) ----
import torch
import torch.nn as nn

class Generator(nn.Module):
    '''
    VolGAN generator
    Generator Class
    Values:
        noise_dim: the dimension of the noise, a scalar
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
        output_dim: output dimension, a scalar
    '''
    def __init__(self, noise_dim,cond_dim, hidden_dim,output_dim, mean_in = False, std_in = False, mean_out = False, std_out = False):
        super(Generator, self).__init__()
        self.input_dim = noise_dim+cond_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        self.mu_i = mean_in
        self.std_i = std_in
        self.mu_o = mean_out
        self.std_o = std_out

        #Add the modules
   
        self.linear1 = nn.Linear(in_features = self.input_dim, out_features = self.hidden_dim)
        self.linear2 = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim * 2)
        self.linear3 = nn.Linear(in_features = self.hidden_dim * 2, out_features = self.output_dim)
        self.activation1 = nn.Softplus()
        self.activation2 = nn.Softplus()
        self.activation3 = nn.Sigmoid()
       

    def forward(self, noise,condition):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''
        #x = combine_vectors(noise.to(torch.float),condition.to(torch.float),2)
        #condition: S_t-1, sigma_t-1, r_t-1, implied vol_t-1
        #out: increment in r_t, increment in implied vol _t
        
        # condition = (condition - self.mu_i) / self.std_i
        out = torch.cat([noise,condition],dim=-1).to(torch.float)
        out = self.linear1(out)
        out = self.activation1(out)
        out = self.linear2(out)
        out = self.activation2(out)
        out = self.linear3(out)
        #uncomment to normalise
        # out = self.mu_o + self.std_o * out
        #out = torch.max(out,torch.tensor(10**(-5)))
        
        return out

class Discriminator(nn.Module):
    '''
    VolGAN discriminator
      in_dim: the input dimension (concatenated with the condition), a scalar
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, in_dim, hidden_dim, mean = False, std = False):
        super(Discriminator, self).__init__()
        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_features=self.input_dim, out_features= self.hidden_dim)
        self.linear2 = nn.Linear(in_features = self.hidden_dim, out_features = 1)
        self.sigmoid = nn.Sigmoid()
        self.Softplus = nn.Softplus()
        self.mu_i = mean
        self.std_i = std

       


    def forward(self, in_chan):
        '''
        in_chan: concatenated condition with real or fake
        h_0 and c_0: for the LSTM
        '''
        x = in_chan
        #uncomment to normalise
        # x = (x - self.mu_i) / self.std_i
        out = self.linear1(x)
        out = self.Softplus(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


def original_smoothness(fake_surface, m, tau, lk, lt, device='cpu'):
    """
    The released smoothness penalty. Setup copied verbatim from VolGAN.py lines 655 and 658-677,
    the per-sample loop from lines 732-738 (indentation shifted from 12 to 4 spaces, nothing else).
    fake_surface: (B, lk*lt) tensor, whatever object is being penalised. Returns the batch means.
    """
    curr_batch_size = fake_surface.shape[0]
    moneyness_t = torch.tensor(m,dtype=torch.float,device=device)
    #smoothness penalties
    Ngrid = lk * lt
    tau_t = torch.tensor(tau,dtype=torch.float,device=device)
    t_seq = torch.zeros((tau_t.shape[0]),dtype=torch.float,device=device)
    for i in range(tau_t.shape[0]-1):
        t_seq[i] = 1/((tau_t[i+1]-tau_t[i])**2)
    matrix_t = torch.zeros((Ngrid,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-1):
        matrix_t[i,i] = -1
        matrix_t[i,i+1] = 1
    tsq = t_seq.repeat(lk).unsqueeze(0)
    matrix_m = torch.zeros((Ngrid-lk,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-lk):
        matrix_m[i,i] = -1
        matrix_m[i,i+lk] = 1
        
    m_seq = torch.zeros((lk*(lt-1)),dtype=torch.float,device=device)
    for i in range(moneyness_t.shape[0]-1):
        m_seq[i*lk:(i+1)*lk] = 1/((moneyness_t[i+1]-moneyness_t[i])**2)
    
    penalties_m = [None] * curr_batch_size
    penalties_t = [None] * curr_batch_size
    for iii in range(curr_batch_size):
        penalties_m[iii] = torch.matmul(m_seq,(torch.matmul(matrix_m,fake_surface[iii])**2))
        penalties_t[iii] = torch.matmul(tsq,(torch.matmul(matrix_t,fake_surface[iii])**2))
    m_penalty = sum(penalties_m) / curr_batch_size
    t_penalty = sum(penalties_t) / curr_batch_size
    return m_penalty, t_penalty


# ---- VolGAN.py lines 90-111 and 113-114 (penalty_tensor, penalty_mutau_tensor: built but unused in training) ----
from tqdm import tqdm

def penalty_tensor(K,T,device):
    """
    matrices for calculating the arbitrage penalty (tensors)
    """
    P_T = torch.zeros(size=(len(T),len(T)),dtype=torch.float,device = device)
    P_K = torch.zeros(size=(len(K),len(K)),dtype=torch.float,device = device)
    PB_K = torch.zeros(size=(len(K),len(K)),dtype=torch.float,device = device)
    #P_T first, the last one is zero
    for j in tqdm(np.arange(0,len(T)-1,1)):
        P_T[j,j] = T[j]/(T[j+1]-T[j])
        P_T[j+1,j] = -T[j]/(T[j+1]-T[j])
    #now P_K and then PB_K
    for i in tqdm(np.arange(0,len(K)-1,1)):
        P_K[i,i] = -1/(K[i+1]-K[i])
        P_K[i,i+1] = 1/(K[i+1]-K[i])
    #PB_K: note that it is a scaled finite difference, but let's compute it on its own just in case
    #once we fix the grid we have to run this function only once so it doesn't matter much
    for i in tqdm(np.arange(1,len(K)-1,1)):
        PB_K[i,i-1] =  -(K[i+1]-K[i]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))
        PB_K[i,i] = (K[i+1] - K[i-1]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))
        PB_K[i,i+1] = -(K[i]-K[i-1]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))
    return P_T,P_K,PB_K

def penalty_mutau_tensor(mu,T,device):
    return penalty_tensor(mu,T,device)


# ---- VolGAN.py lines 647-779 (GradientMatching) ----
def GradientMatching(gen,gen_opt,disc,disc_opt,criterion,condition_train,true_train,m,tau,ms,taus,n_grad,lrg,lrd,batch_size,noise_dim,device, lk = 10, lt = 8):
    """
    perform gradient matching
    """
    n_train = condition_train.shape[0]
    n_batches =  n_train // batch_size + 1
    dtm = tau * 365
    mP_t,mP_k,mPb_K = penalty_mutau_tensor(m,dtm,device)
    moneyness_t = torch.tensor(m,dtype=torch.float,device=device)
    
    
    #smoothness penalties
    Ngrid = lk * lt
    tau_t = torch.tensor(tau,dtype=torch.float,device=device)
    t_seq = torch.zeros((tau_t.shape[0]),dtype=torch.float,device=device)
    for i in range(tau_t.shape[0]-1):
        t_seq[i] = 1/((tau_t[i+1]-tau_t[i])**2)
    matrix_t = torch.zeros((Ngrid,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-1):
        matrix_t[i,i] = -1
        matrix_t[i,i+1] = 1
    tsq = t_seq.repeat(lk).unsqueeze(0)
    matrix_m = torch.zeros((Ngrid-lk,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-lk):
        matrix_m[i,i] = -1
        matrix_m[i,i+lk] = 1
        
    m_seq = torch.zeros((lk*(lt-1)),dtype=torch.float,device=device)
    for i in range(moneyness_t.shape[0]-1):
        m_seq[i*lk:(i+1)*lk] = 1/((moneyness_t[i+1]-moneyness_t[i])**2)
    
    n_epochs = n_grad
    discloss = [False] * (n_batches*n_epochs)
    genloss = [False] * (n_batches*n_epochs)
    dscpred_real = [False] * (n_batches*n_epochs)
    dscpred_fake = [False] * (n_batches*n_epochs)
    gen_fake = [False] * (n_batches*n_epochs)
    genprices_fk = [False] * (n_batches*n_epochs)
    BCE_grad = []
    m_smooth_grad = []
    t_smooth_grad = []
    gen.train()
    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(n_train)
        condition_train = condition_train[perm,:]
        true_train = true_train[perm,:]
        for i in range(n_batches):
            curr_batch_size = batch_size
            if i==(n_batches-1):
                curr_batch_size = n_train-i*batch_size
            condition = condition_train[(i*batch_size):(i*batch_size+curr_batch_size),:]
            surface_past = condition_train[(i*batch_size):(i*batch_size+curr_batch_size),3:]
            real = true_train[(i*batch_size):(i*batch_size+curr_batch_size),:]

            real_and_cond = torch.cat((condition,real),dim=-1)
            #update the discriminator
            disc_opt.zero_grad()
            noise = torch.randn((curr_batch_size,noise_dim), device=device,dtype=torch.float)
            fake = gen(noise,condition)
            fake_and_cond = torch.cat((condition,fake),dim=-1)

            disc_fake_pred = disc(fake_and_cond.detach())
            disc_real_pred = disc(real_and_cond)
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_opt.step()
            
            dscpred_real[epoch*n_batches+i] = disc_real_pred[0].detach().item()
            dscpred_fake[epoch*n_batches+i] = disc_fake_pred[0].detach().item()
            
            discloss[epoch*n_batches+i] = disc_loss.detach().item()
            
            #update the generator
            gen_opt.zero_grad()
            noise = torch.randn((curr_batch_size,noise_dim), device=device,dtype=torch.float)
            fake = gen(noise,condition)

            fake_and_cond = torch.cat((condition,fake),dim=-1)
            
            disc_fake_pred = disc(fake_and_cond)
            
            fake_surface = torch.exp(fake[:,1:]+ surface_past)

            penalties_m = [None] * curr_batch_size
            penalties_t = [None] * curr_batch_size
            for iii in range(curr_batch_size):
                penalties_m[iii] = torch.matmul(m_seq,(torch.matmul(matrix_m,fake_surface[iii])**2))
                penalties_t[iii] = torch.matmul(tsq,(torch.matmul(matrix_t,fake_surface[iii])**2))
            m_penalty = sum(penalties_m) / curr_batch_size
            t_penalty = sum(penalties_t) / curr_batch_size
            
            m_penalty.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            m_smooth_grad.append(total_norm)
            gen_opt.zero_grad()
            
            t_penalty.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            t_smooth_grad.append(total_norm)
            
            gen_opt.zero_grad()
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            BCE_grad.append(total_norm)
            gen_opt.step()
            genloss[epoch*n_batches+i] = gen_loss.detach().item()
            gen_fake[epoch*n_batches+i] = fake[0].detach()
            genprices_fk[epoch*n_batches+i]= condition[0].detach()
        
            

    alpha = np.mean(np.array(BCE_grad) / np.array(m_smooth_grad))
    beta = np.mean(np.array(BCE_grad) / np.array(t_smooth_grad))
    print("alpha :", alpha, "beta :", beta)
    return gen,gen_opt,disc,disc_opt,criterion, alpha, beta


# ---- VolGAN.py lines 938-1036 (TrainLoopNoVal) ----
def TrainLoopNoVal(alpha,beta,gen,gen_opt,disc,disc_opt,criterion,condition_train,true_train,m,tau,ms,taus,n_epochs,lrg,lrd,batch_size,noise_dim,device, lk = 10, lt = 8):
    """
    train loop for VolGAN
    """
    n_train = condition_train.shape[0]
    n_batches =  n_train // batch_size + 1
    dtm = tau * 365
    mP_t,mP_k,mPb_K = penalty_mutau_tensor(m,dtm,device)
    moneyness_t = torch.tensor(m,dtype=torch.float,device=device)
    #smoothness penalties
    Ngrid = lk * lt
    tau_t = torch.tensor(tau,dtype=torch.float,device=device)
    t_seq = torch.zeros((tau_t.shape[0]),dtype=torch.float,device=device)
    for i in range(tau_t.shape[0]-1):
        t_seq[i] = 1/((tau_t[i+1]-tau_t[i])**2)
    matrix_t = torch.zeros((Ngrid,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-1):
        matrix_t[i,i] = -1
        matrix_t[i,i+1] = 1
    tsq = t_seq.repeat(lk).unsqueeze(0)
    matrix_m = torch.zeros((Ngrid-lk,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-lk):
        matrix_m[i,i] = -1
        matrix_m[i,i+lk] = 1
        
    m_seq = torch.zeros((lk*(lt-1)),dtype=torch.float,device=device)
    for i in range(moneyness_t.shape[0]-1):
        m_seq[i*lk:(i+1)*lk] = 1/((moneyness_t[i+1]-moneyness_t[i])**2)
    
    discloss = [False] * (n_batches*n_epochs)
    genloss = [False] * (n_batches*n_epochs)
    dscpred_real = [False] * (n_batches*n_epochs)
    dscpred_fake = [False] * (n_batches*n_epochs)
    gen_fake = [False] * (n_batches*n_epochs)
    genprices_fk = [False] * (n_batches*n_epochs)

    gen.train()
    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(n_train)
        condition_train = condition_train[perm,:]
        true_train = true_train[perm,:]
        for i in range(n_batches):
            curr_batch_size = batch_size
            if i==(n_batches-1):
                curr_batch_size = n_train-i*batch_size
            condition = condition_train[(i*batch_size):(i*batch_size+curr_batch_size),:]
            surface_past = condition_train[(i*batch_size):(i*batch_size+curr_batch_size),3:]
            real = true_train[(i*batch_size):(i*batch_size+curr_batch_size),:]

            real_and_cond = torch.cat((condition,real),dim=-1)
            #update the discriminator
            disc_opt.zero_grad()
            noise = torch.randn((curr_batch_size,noise_dim), device=device,dtype=torch.float)
            fake = gen(noise,condition)
            fake_and_cond = torch.cat((condition,fake),dim=-1)

            disc_fake_pred = disc(fake_and_cond.detach())
            disc_real_pred = disc(real_and_cond)
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_opt.step()
            
            dscpred_real[epoch*n_batches+i] = disc_real_pred[0].detach().item()
            dscpred_fake[epoch*n_batches+i] = disc_fake_pred[0].detach().item()
            
            discloss[epoch*n_batches+i] = disc_loss.detach().item()
            
            #update the generator
            gen_opt.zero_grad()
            noise = torch.randn((curr_batch_size,noise_dim), device=device,dtype=torch.float)
            fake = gen(noise,condition)

            fake_and_cond = torch.cat((condition,fake),dim=-1)
            
            disc_fake_pred = disc(fake_and_cond)
            
            # fake_surface = torch.exp(fake[:,1:]+ surface_past)
            fake_surface = fake[:,1:]+ surface_past

            penalties_m = [None] * curr_batch_size
            penalties_t = [None] * curr_batch_size
            for iii in range(curr_batch_size):
                penalties_m[iii] = torch.matmul(m_seq,(torch.matmul(matrix_m,fake_surface[iii])**2))
                penalties_t[iii] = torch.matmul(tsq,(torch.matmul(matrix_t,fake_surface[iii])**2))
            m_penalty = sum(penalties_m) / curr_batch_size
            t_penalty = sum(penalties_t) / curr_batch_size
            
            gen_opt.zero_grad()
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) + alpha * m_penalty + beta * t_penalty
            gen_loss.backward()
            gen_opt.step()
            genloss[epoch*n_batches+i] = gen_loss.detach().item()
            gen_fake[epoch*n_batches+i] = fake[0].detach()
            genprices_fk[epoch*n_batches+i]= condition[0].detach()
            
        
    return gen,gen_opt,disc,disc_opt,criterion


# ---- VolGAN.py lines 1106-1203 and 1274-1286 (weighted statistics; used only to show bugs B4, B5, B6, B17) ----
def reweighting_stats_mean(penalties,beta):
    """
    post re-weighting stats (mean)
    penalties: arbitrage penalties #days x #samples
    """
    data_m = penalties
    transform = np.exp(- beta * data_m)
    for i in range(transform.shape[0]):
        transform[i,:] = transform[i,:] / np.sum(transform[i,:])

    mean_after = np.zeros(transform.shape[0])
    for i in tqdm(range(transform.shape[0])):
        mean_after[i] = np.sum(transform[i,:] * data_m[i,:])
    median_after = np.zeros(transform.shape[0])
    for i in tqdm(range(transform.shape[0])):
        mean_after[i] = np.sum(transform[i,:] * data_m[i,:])
        args = np.argsort(data_m[i])
        data_m[i,:] = data_m[i,args]
        transform[i,:] = transform[i,:]
        med = data_m[i,0]
        sm = 0
        j = 0
        while sm < 0.5:
            sm = sm + transform[i,j]
            med = data_m[i,j]
            j = j +1
        median_after[i] = med
    
    return mean_after, median_after
def VolGAN_sample(sample_surfaces, sample_returns,weights):
    """
    sample_surfaces, sample_returns: raw generated
    weights: associated with the raw samples
    """
    n = weights.shape[0]
    N = weights.shape[1]
    Nm = sample_surfaces.shape[2]
    sample_back_s = np.zeros((n, Nm))
    sample_back_r = np.zeros(n)
    for i in range(n):
        index = np.random.choice(range(N), size=1, replace=True, p=weights[i,:])
        sample_back_s[i,:] = sample_surfaces[index,i,:]
        sample_back_r[i] = sample_returns[index,i]
    return sample_back_s, sample_back_r


def VolGAN_mean_surface_day(variable,weights):
    """
    mean surface for a ficed day (VolGAN)
    """
    mn = np.zeros(variable.shape[1])
    for i in range(len(weights)):
        mn = mn + weights[i] * variable[i,:]
    return mn

def VolGAN_quantile_surface_day(variable,weights,quantile):
    """
    VolGAN quantile for a fixed day
    """
    args = np.argsort(weights)
    weights = weights[args]
    variable = variable[args,:]
    tot = 0
    i = 0
    while tot < quantile:
        tot = tot + weights[i]
        i = i + 1
    if i == len(weights):
        qnt = variable[-1,:]
    else:
        qnt = variable[i, :]
    return qnt

def VolGAN_mean_ret_day(variable,weights):
    """
    mean VolGAN return on a fixed day
    """
    mn = 0
    for i in range(len(weights)):
        mn = mn + weights[i] * variable[i]
    return mn
def VolGAN_quantile_ret_day(variable,weights,quantile):
    """
    returns VolGAN quantile on a fixed day
    """
    args = np.argsort(variable)
    weights = weights[args]
    variable = variable[args]
    tot = 0
    i = 0
    while tot < quantile:
        tot = tot + weights[i]
        i = i + 1
    if i == len(weights):
        qnt = variable[-1]
    else:
        qnt = variable[i]
    return qnt

def quantile(quantity, weights, quantile):
    """
    custom quantile function
    """
    indxs = np.argsort(quantity)
    quantity = quantity[indxs]
    weights = weights[indxs]
    i = -1
    w = 0
    while w < quantile:
        i = i + 1
        w = w + weights[i]
    return quantity[i]

# ---- VolGAN.py lines 167-181, 217-235, 1222-1272 (pricing, interpolation, VIX; used only to show F11, B2, B3) ----
def BS_OptionPrice(St,tau,r,K,sigma):
    """
    St: current asset price (at time t)
    tau: time to expiry of the option
    r: risk-free interest rate
    K: strike
    sigma: volatility
    ***all of the above can be both vectors and scalars***
    returns CALL PRICE(s) using the Black-Scholes formula
    """
    #delta
    d1 = (np.log(St/K)+tau*(r+0.5*sigma*sigma))/(sigma*np.sqrt(tau))
    d2 = d1-sigma*np.sqrt(tau)
    price = St*norm.cdf(d1)-K*norm.cdf(d2)*np.exp(-r*tau)
    return price

def get_points(I_known,K_known,tau_known,K_want,tau_want):
    """
    given I(K,T) returns points in I(K*,tau*) using interpolation in K and T
    K_known and T_known need to be sequences
    """
    #interpolate Ks first (fix tau known)
    xvals = K_want
    S = np.zeros(shape=(len(K_want),len(tau_known)))
    for i in range(len(tau_known)):
        yinterp = np.interp(xvals, K_known,I_known[:,i])
        S[:,i] = yinterp
    #interpolate taus now
    xvals = tau_want
    I = np.zeros(shape=(len(K_want),len(tau_want)))
    xvals = tau_want
    for i in range(len(K_want)):
        yinterp = np.interp(xvals,tau_known,S[i,:])
        I[i,:] = yinterp
    return I

def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    y = np.interp(x, xp, yp)
    y = np.where(x<xp[0], yp[0]+(x-xp[0])*(yp[0]-yp[1])/(xp[0]-xp[1]), y)
    y = np.where(x>xp[-1], yp[-1]+(x-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2]), y)
    return y

def get_points3(I_known,m_known,tau_known,m_want,tau_want):
    """
    given I(K,T) returns points in I(K*,tau*) using interpolation in K and T
    K_known and T_known need to be sequences
    """
    #interpolate Ks first (fix tau known)
    xvals = m_want
    S = np.zeros(shape=(len(m_want),len(tau_known)))
    for i in range(len(tau_known)):
        yinterp = extrap(xvals, m_known,I_known[:,i])
        S[:,i] = yinterp
    #interpolate taus now
    xvals = tau_want
    I = np.zeros(shape=(len(m_want),len(tau_want)))
    xvals = tau_want
    for i in range(len(m_want)):
        yinterp = extrap(xvals,tau_known,S[i,:])
        I[i,:] = yinterp
    return I
def VIX(St, IV,m,tau):
    """
    VIX simulation
    St: underlying
    IV: implied vol surface
    m, tau: moneyness and implied vol grid
    """
    m_min = 0.5
    m_max = 1.5
    n_s = 50
    m1 = np.linspace(m_min,m_max,2*n_s)
    strikes = m1 * St
    i0 = np.argmin(np.abs(strikes-St))
    sigma1 = get_points3(IV,m,tau,m1,[1/12])[0][0]
    r = 0
    Call = BS_OptionPrice(St,1/12,r,strikes,sigma1)
    Put = Call - St + strikes
    DeltaK = np.zeros(len(strikes))
    DeltaK[1:] = np.diff(strikes)
    DeltaK[0] = DeltaK[1]
    VIX = 0
    for i in range(i0 + 1):
        VIX = VIX + Put[i] * DeltaK[i] / (strikes[i]**2)
    VIX = 100 * np.sqrt(2 / (1 / 12) * VIX - 1 / (1 / 12) * (St / strikes[i0] - 1)**2)
    return VIX
