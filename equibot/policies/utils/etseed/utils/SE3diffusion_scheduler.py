import torch, numpy as np
torch.manual_seed(3407)
from .se_math import se3,so3
from .data_utils import bezier_curve
from diffusers import DDPMScheduler
from pdb import set_trace as bp
import math
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

class DiffusionScheduler(torch.nn.Module):
    def __init__(self,num_steps=100,beta_T:float = 0.05,sigma_r: float = 0.05,sigma_t: float = 0.03,mode='squaredcos_cap_v2',device= torch.device("cuda")):
        super().__init__()
        self.num_steps: int = num_steps # 100
        self.beta_1: float = 1e-4
        self.beta_T:float=beta_T
        self.sigma_r: float = sigma_r # 0.2 0.05 0.001 0.0005
        self.sigma_t: float = sigma_t # 0.1 0.03 0.001 0.0003
        self.mode = mode # ['linear','cosine','squaredcos_cap_v2']
        self.S = 0.008 # 0.008
       
        device = device
        # TODO DO NOT HARDCODE
        if self.mode == 'linear':
            self.betas = torch.linspace(self.beta_1, self.beta_T, steps=self.num_steps) 
            self._train_betas=torch.linspace(self.beta_1, self.beta_T, steps=100)
        elif self.mode == 'cosine':
            def betas_fn(s,T = self.num_steps):
                T = self.num_steps
                def f(t, T, s):
                    return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
                alphas = [f(0, T, s)]
                for t in range(0, T):
                    alphas.append(f(t, T, s) / alphas[-1])
                betas = [1 - alpha / alphas[0] for alpha in alphas]

                return [min(beta, 0.999) for beta in betas]
            betas = betas_fn(s=self.S)
            self.betas = torch.FloatTensor(betas)
            og_betas = betas_fn(s=self.S,T=100)
            self.og_betas = torch.FloatTensor(og_betas)

        elif self.mode == 'squaredcos_cap_v2':
            self.betas = betas_for_alpha_bar(self.num_steps)
            self.og_betas = betas_for_alpha_bar(100)
            
        else:
            raise RuntimeError(f"f{self.mode} is not yet implemented")
        
        self.alphas = 1.0 - self.betas
        self.log_alphas = torch.log(self.alphas)
        self.alphas_cumsum = torch.cumsum(self.log_alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.alpha_bars = self.alphas_cumsum.exp().to(device)

        self.og_alphas = 1.0 - self.og_betas
        self.og_log_alphas = torch.log(self.og_alphas)
        self.og_alphas_cumsum = torch.cumsum(self.og_log_alphas, dim=0)
        self.og_alpha_bars = self.og_alphas_cumsum.exp().to(device)
        
        self.gamma0 = torch.zeros_like(self.betas).to(device)
        self.gamma1 = torch.zeros_like(self.betas).to(device)
        self.gamma2 = torch.zeros_like(self.betas).to(device)
        self.lambda0=torch.zeros_like(self.betas).to(device)
        self.lambda1=torch.zeros_like(self.betas).to(device)
        self.v_coeff1=torch.zeros_like(self.betas).to(device)
        self.v_coeff2=torch.zeros_like(self.betas).to(device)
        # self.v_coeff1=torch.ones_like(self.betas).to(device)
        self._alpha=torch.zeros_like(self.betas).to(device)
        self._beta=torch.zeros_like(self.betas).to(device)

        for t in range(1, self.num_steps):  # 2 to T
            alpha_prod_t = self.alpha_bars[t]
            alpha_prod_t_prev = self.alpha_bars[t - 1] if t > 0 else self.one
            self.gamma0[t] = self.betas[t] * torch.sqrt(alpha_prod_t_prev) / (1. - alpha_prod_t)
            self.gamma1[t] = (1. - alpha_prod_t_prev) * torch.sqrt(alpha_prod_t) / (1. - alpha_prod_t)
            self.gamma2[t] = torch.sqrt((1. - alpha_prod_t_prev) * self.betas[t] / (1. - alpha_prod_t))
            self.lambda0[t] = 1./ torch.sqrt(alpha_prod_t)
            self.lambda1[t] = self.betas[t] / torch.sqrt(1. - alpha_prod_t)
            self.v_coeff1[t] = torch.sqrt(alpha_prod_t)
            self.v_coeff2[t] = torch.sqrt(1.-alpha_prod_t)
            prev_a= self.alphas[t-1] if t > 0 else self.one
            self._beta[t]=(1.0-prev_a)/(1-self.alphas[t])
            self._alpha[t]=prev_a-self.alphas[t]*(1.0-prev_a)/(1.0-self.alphas[t])

            
        self.gamma2[-1]=0.0

    def set_timesteps(self,num_steps):
        self.num_steps = num_steps
    
    def add_noise(self,
        original_samples: torch.FloatTensor, # [B, Ho, 4, 4]
        timesteps: torch.IntTensor, # [B]
        device,
        no_noise=False):
        B = original_samples.shape[0] # batch
        Ho = original_samples.size(1)  # horizon
        
        # H_T the identity transformation
        H_T = torch.eye(4)[None].expand(B,Ho, -1, -1).to(device) # H_T: [B,Ho,4,4]
        alpha_bars = self.alpha_bars[timesteps].to(device) # [B]
      
        # H_t [B,Ho,4,4] the interpolation part, see eq 35
        H_t = se3.exp((1. - torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(H_T @ (torch.inverse(original_samples).to(torch.float32)))) @ original_samples.to(torch.float32)

        # add noise
        # the gamma in perturbation
        # BUG SHOULD BE SIGMA_T, SIGMA_R in kornia
        # scale = torch.cat([torch.ones(3) * self.sigma_r, torch.ones(3) * self.sigma_t])[None].to(device)  # [1, 6] 
        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        noise = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]
            
        # perturbation part in eq 34
        H_pure_noise = se3.exp(noise) #  [B,Ho,4,4]
        if no_noise:
            return H_t,H_pure_noise
        
        # perturbation + interpolation, see eq 34
        noisy_interpolated_H_t = H_pure_noise @ H_t #  [B,Ho,4,4]
        

        return noisy_interpolated_H_t, H_pure_noise

    
    def add_noise9(self,
        original_samples: torch.FloatTensor, # [B, Ho, 4, 4]
        timesteps: torch.IntTensor, # [B]
        device,
        no_noise=False):
        B = original_samples.shape[0] # batch
        Ho = original_samples.size(1)  # horizon
        
        # H_T the identity transformation
        H_T = torch.eye(4)[None].expand(B,Ho, -1, -1).to(device) # H_T: [B,Ho,4,4]
        alpha_bars = self.alpha_bars[timesteps].to(device) # [B]
      
        # H_t [B,Ho,4,4] the interpolation part, see eq 35
        # H_t = se3.exp((1. - torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(H_T @ (torch.inverse(original_samples).to(torch.float32)))) @ original_samples.to(torch.float32)
        H_t = se3.exp((torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(((original_samples).to(torch.float32)))) 
        
        # add noise
        # the gamma in perturbation
        # BUG SHOULD BE SIGMA_T, SIGMA_R in kornia
        # scale = torch.cat([torch.ones(3) * self.sigma_r, torch.ones(3) * self.sigma_t])[None].to(device)  # [1, 6] 
        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        _noise =  scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]
        noise=torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1)*_noise   
        # perturbation part in eq 34
        H_pure_noise = se3.exp(noise) #  [B,Ho,4,4]
        if no_noise:
            return H_t,H_pure_noise
        
        # perturbation + interpolation, see eq 34
        noisy_interpolated_H_t = H_pure_noise @ H_t #  [B,Ho,4,4]
        
        snr=alpha_bars/(1-alpha_bars)
        return noisy_interpolated_H_t, se3.exp(_noise),H_t, snr

    def add_noise9_decoupled(
        self,
        original_samples: torch.FloatTensor,  # [B, Ho, 4, 4]
        timesteps: torch.IntTensor,           # [B]
        device,
        no_noise: bool = False,
    ):
        """
        - Interpolate rotation on SO(3): R_t = exp( sqrt(alpha_rot) * log(R) )
        - Interpolate translation in R^3: t_t = sqrt(alpha_trn) * t
        - Rotation noise: left-compose (so(3) Gaussian via exp)
        - Translation noise: add in WORLD frame (simple + in R^3), NOT via SE(3) multiplication
        """
        B = original_samples.shape[0]
        Ho = original_samples.size(1)

        # schedules (allow separate rot/trans; fall back if not present)
        alpha_rot = getattr(self, "alpha_bars_rot", self.alpha_bars)[timesteps].to(device)   # [B]
        alpha_trn = getattr(self, "alpha_bars_trans", self.alpha_bars)[timesteps].to(device) # [B]

        sqrt_alpha_rot = torch.sqrt(alpha_rot).view(B, 1, 1)
        sqrt_alpha_trn = torch.sqrt(alpha_trn).view(B, 1, 1)
        sqrt_one_m_rot = torch.sqrt(1.0 - alpha_rot).view(B, 1, 1)
        sqrt_one_m_trn = torch.sqrt(1.0 - alpha_trn).view(B, 1, 1)

        # decompose
        H = original_samples.to(torch.float32)
        R = H[..., :3, :3]           # [B,Ho,3,3]
        t = H[..., :3,  3]           # [B,Ho,3]

        # interpolation
        rot_log = so3.log(R)                               # [B,Ho,3]
        R_t = so3.exp(sqrt_alpha_rot * rot_log)            # [B,Ho,3,3]
        t_t = sqrt_alpha_trn * t                           # [B,Ho,3]

        # assemble H_t
        H_t = torch.eye(4, dtype=torch.float32, device=device).view(1,1,4,4).expand(B, Ho, -1, -1).clone()
        H_t[..., :3, :3] = R_t
        H_t[..., :3,  3] = t_t
        assert R_t.shape == (B, Ho, 3, 3)
        assert t_t.shape == (B, Ho, 3)
        assert H_t.shape == (B, Ho, 4, 4)
        if no_noise:
            H_rot_noise = torch.eye(4, dtype=torch.float32, device=device).view(1,1,4,4).expand(B, Ho, -1, -1)
            out = H_rot_noise @ H_t
            snr = {
                "rot": alpha_rot / (1.0 - alpha_rot + 1e-12),
                "trans": alpha_trn / (1.0 - alpha_trn + 1e-12),
            }
            t_noise_world = torch.zeros(B, Ho, 3, dtype=torch.float32, device=device)
            return out, H_rot_noise, H_t, snr, t_noise_world

        # ---------- NOISE ----------
        # (1) rotation noise via left composition
        eps_r = torch.randn(B, Ho, 3, device=device, dtype=torch.float32)
        R_noise = so3.exp(sqrt_one_m_rot * (self.sigma_r * eps_r))         # [B,Ho,3,3]

        H_rot_noise = torch.eye(4, dtype=torch.float32, device=device).view(1,1,4,4).expand(B, Ho, -1, -1).clone()
        H_rot_noise[..., :3, :3] = R_noise

        # compose rotation noise into interpolation
        out = H_rot_noise @ H_t                                            # [B,Ho,4,4]

        # (2) translation noise added in WORLD frame (independent of orientation)
        eps_t = torch.randn(B, Ho, 3, device=device, dtype=torch.float32)
        t_noise_world = sqrt_one_m_trn * (self.sigma_t * eps_t)            # [B,Ho,3]
        out[..., :3, 3] = out[..., :3, 3] + t_noise_world                  # add directly

        snr = {
            "rot": alpha_rot / (1.0 - alpha_rot + 1e-12),
            "trans": alpha_trn / (1.0 - alpha_trn + 1e-12),
        }
        return out, H_rot_noise, H_t, snr

    def add_noise2(self,
        original_samples: torch.FloatTensor, # [B, Ho, 4, 4]
        timesteps: torch.IntTensor, # [B]
        device,
        no_noise=False):
        B = original_samples.shape[0] # batch
        Ho = original_samples.size(1)  # horizon
        
        # H_T the identity transformation
        H_T = torch.eye(4)[None].expand(B,Ho, -1, -1).to(device) # H_T: [B,Ho,4,4]
        alpha_bars = self.alpha_bars[timesteps].to(device) # [B]
      
        # H_t [B,Ho,4,4] the interpolation part, see eq 35
        # H_t = se3.exp((1. - torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(H_T @ (torch.inverse(original_samples).to(torch.float32)))) @ original_samples.to(torch.float32)
        # H_t = se3.exp((torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(original_samples))
        H_t = se3.exp((torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(original_samples))

        # add noise
        # the gamma in perturbation
        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        lie_noise= scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)
        # lie_noise= torch.randn(B,Ho, 6).to(device)
        noise = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * lie_noise  # [B,Ho, 6]
        # noise = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * lie_noise  # [B,Ho, 6]
            
        # perturbation part in eq 34
        H_pure_noise = se3.exp(noise) #  [B,Ho,4,4]
        if no_noise:
            return H_t,H_pure_noise
        
        # perturbation + interpolation, see eq 34
        noisy_interpolated_H_t = H_pure_noise @ H_t #  [B,Ho,4,4]
        noisy_interpolated_lie_H_t=se3.log(noisy_interpolated_H_t)

        return noisy_interpolated_H_t, H_pure_noise, noisy_interpolated_lie_H_t, lie_noise, se3.log(original_samples)
    

    def add_noisetest(self,
        original_samples: torch.FloatTensor, # [B, Ho, 4, 4]
        timesteps: torch.IntTensor, # [B]
        device,
        no_noise=False):
        B = original_samples.shape[0] # batch
        Ho = original_samples.size(1)  # horizon
        
        # H_T the identity transformation
        H_T = torch.eye(4)[None].expand(B,Ho, -1, -1).to(device) # H_T: [B,Ho,4,4]
        a = self.alphas.to(device)[timesteps.to(device)].unsqueeze(-1).unsqueeze(-1) # [B]
        lie_h_0=se3.log(original_samples)

        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        lie_noise= scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)
        lie_h_t = a*lie_h_0+(1.0-a)*se3.log(H_T)+(1.0-a)*lie_noise
        return lie_h_t,se3.exp(lie_h_t),lie_h_0
    

    def denoisetest(self,
        lie_h_0, # [B,Ho,4,4]
        timestep, # [B]
        lie_sample, # [B,Ho,4,4]
        device,
        abs_to_rel=False):

        timestep = timestep[0].cpu() # scalar
        B = lie_sample.shape[0]
        Ho = lie_sample.shape[1]
        # see algorithm 2, but no longer use A^{k->0}A^k
        alpha = self._alpha[timestep].to(device)
        beta = self._beta[timestep].to(device)
        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        lie_noise= scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)
        lie_sample=alpha*lie_h_0+beta*lie_sample+beta*scale*lie_noise        
        return lie_sample,se3.exp(lie_sample)    
    

    def add_noise3(self,
        original_samples: torch.FloatTensor, # [B, Ho, 4, 4]
        timesteps: torch.IntTensor, # [B]
        device,
        no_noise=False):
        B = original_samples.shape[0] # batch
        Ho = original_samples.size(1)  # horizon
        v_coeff1=self.v_coeff1[timesteps].to(device).unsqueeze(-1).unsqueeze(-1)
        v_coeff2=self.v_coeff2[timesteps].to(device).unsqueeze(-1).unsqueeze(-1)

        # H_T the identity transformation
        H_T = torch.eye(4)[None].expand(B,Ho, -1, -1).to(device) # H_T: [B,Ho,4,4]
        alpha_bars = self.alpha_bars[timesteps].to(device) # [B]
      
        # H_t [B,Ho,4,4] the interpolation part, see eq 35
        # H_t = se3.exp((1. - torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(H_T @ (torch.inverse(original_samples).to(torch.float32)))) @ original_samples.to(torch.float32)
        H_t = se3.exp((torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(original_samples))

        # add noise
        # the gamma in perturbation
        scale = torch.cat([torch.ones(3) * self.sigma_r, torch.ones(3) * self.sigma_t])[None].to(device)  # [1, 6] 
        lie_noise= scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)
        noise = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * lie_noise  # [B,Ho, 6]
            
        # perturbation part in eq 34
        H_pure_noise = se3.exp(noise) #  [B,Ho,4,4]
        if no_noise:
            return H_t,H_pure_noise
        
        # perturbation + interpolation, see eq 34
        noisy_interpolated_H_t = H_pure_noise @ H_t #  [B,Ho,4,4]
        noisy_interpolated_lie_H_t=se3.log(noisy_interpolated_H_t)
        lie_tgt_v=v_coeff1*lie_noise-v_coeff2*se3.log(original_samples)
        tgt_v=se3.exp(lie_tgt_v)
        return noisy_interpolated_H_t, H_pure_noise, noisy_interpolated_lie_H_t, lie_noise, tgt_v,lie_tgt_v
    
    
    def denoise(self,
                reconstructed_H_0, # [B,Ho,4,4]
                timestep, # [B]
                sample, # [B,Ho,4,4]
                device,
                abs_to_rel=False):
        
        timestep = timestep[0].cpu() # scalar
        B = sample.shape[0]
        Ho = sample.shape[1]
        # see algorithm 2, but no longer use A^{k->0}A^k
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        self.gamma2[-1]=0.0
        gamma2 = self.gamma2[timestep].to(device)
        alpha_bars = self.alpha_bars[timestep].to(device) # [B]

        scale = torch.cat([torch.ones(3) * self.sigma_r, torch.ones(3) * self.sigma_t])[None].to(device)
        # print(reconstructed_H_0)
        # print(se3.log(reconstructed_H_0))
        # print("^^^^^")
        if abs_to_rel: 
            reconstructed_H_0=reconstructed_H_0@sample
        
        sample = se3.exp(gamma0 * se3.log(reconstructed_H_0) + gamma1 * se3.log(sample))# + scale*gamma2*torch.randn(B,Ho,6).to(device))#torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1)*
        return sample # sample = A^{k-1}, reconstructed_H_0 = A^{k->0}A^k, see algorithm 2
    

    def denoise9(self,
                reconstructed_H_0, # [B,Ho,4,4]
                timestep, # [B]
                sample, # [B,Ho,4,4]
                device,
                abs_to_rel=False):
        
        timestep = timestep[0].cpu() # scalar
        B = sample.shape[0]
        Ho = sample.shape[1]
        # see algorithm 2, but no longer use A^{k->0}A^k
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        self.gamma2[-1]=0.0
        gamma2 = self.gamma2[timestep].to(device)
        if timestep>0: 
            timestep=timestep-1
        alpha_bars = self.alpha_bars[timestep].to(device) # [B]
        # scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        # print(reconstructed_H_0)
        # print(se3.log(reconstructed_H_0))
        # print("^^^^^")
        if abs_to_rel: 
            reconstructed_H_0=reconstructed_H_0@sample
            
        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        
        # # noise = scale*gamma2*torch.randn(B,Ho,6).to(device)  # [B,Ho, 6]
        # # noise = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]
        # noise=self.betas[timestep]* scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device) 
        # H_pure_noise=se3.exp(noise)
        # # a=gamma0 * se3.log(reconstructed_H_0)
        # # A=se3.exp(a)
        # # b=gamma1 * se3.log(sample)
        # # B=se3.exp(b)
        # # commuter=A@B-B@A
        # # sample = se3.exp(a + b+se3.log(commuter))# + scale*gamma2*torch.randn(B,Ho,6).to(device))#scale*torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1)*
        # sample = se3.exp(gamma0 * se3.log(reconstructed_H_0) + gamma1 * se3.log(sample))# + scale*gamma2*torch.randn(B,Ho,6).to(device))#scale*torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1)*
        # # sample = se3.exp(scale*gamma1 * se3.log(sample))@se3.exp(scale*gamma0 * se3.log(reconstructed_H_0))


        # alpha_bars = self.alpha_bars[timestep].to(device) # [B]
        # tau_t=self.betas[timestep]
        # if timestep>0: 
        #     timestep=timestep-1
        # alpha_bars_prev = self.alpha_bars[timestep].to(device) # [B]
        # x0=reconstructed_H_0
        # xt=sample
        # log_x0=se3.log(x0)
        # log_xt=se3.log(xt)
        # noise_t = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]
        # sample=torch.sqrt(alpha_bars_prev)*log_x0+noise_t*torch.sqrt(1-alpha_bars_prev-tau_t**2)
        # sample=se3.exp(sample)
        # H_pure_noise=se3.exp(tau_t*noise)



        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim_inverse.py
        alpha_bars = self.alpha_bars[timestep].to(device) # [B]
        eta_t=self.betas[timestep]
        if timestep>0: 
            timestep=timestep-1
        # else:
        #     return reconstructed_H_0,None
        alpha_bars_prev = self.alpha_bars[timestep].to(device) # [B]
        x0=reconstructed_H_0
        xt=sample
        log_x0=se3.log(x0)
        log_xt=se3.log(xt)
        noise_t = (log_xt-torch.sqrt(alpha_bars)*log_x0)/torch.sqrt(1-alpha_bars)#torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]
        # sqrt_recip_alphas_cumprod=torch.sqrt(1./alpha_bars)
        # sqrt_recipm1_alphas_cumprod=torch.sqrt(1./alpha_bars -1)
        # noise_t=sqrt_recip_alphas_cumprod*(log_xt-log_x0)/sqrt_recipm1_alphas_cumprod # lucidrians

        # xt_nf = se3.exp(torch.sqrt(alpha_bars).unsqueeze(-1).unsqueeze(-1) * se3.log(x0.to(torch.float32)))
        # _noise = se3.log(xt@torch.inverse(xt_nf))
        # noise_t= _noise #se3.log(se3.exp(scale*-torch.sqrt(alpha_bars/torch.sqrt(1-alpha_bars))*se3.log(torch.inverse(x0)))@se3.exp(scale*se3.log(xt)/torch.sqrt(1-alpha_bars)))
        
        # noise_t2 = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]

        eta=1 # 0 ,1 or what 
        sigma = eta * ((1 - alpha_bars / alpha_bars_prev) * (1 - alpha_bars_prev) / (1 - alpha_bars)).sqrt() # lucidrains/ddim d3
        # dir_t=torch.sqrt(1-alpha_bars_prev-sigma**2)*noise_t # for lie algebra
        dir_t=torch.sqrt(1-alpha_bars_prev)*noise_t
        sample=torch.sqrt(alpha_bars_prev)*log_x0+dir_t
        # nx0=se3.exp(torch.sqrt(alpha_bars_prev)*log_x0)
        # ndt=se3.exp(dir_t)
        # sample=torch.sqrt(alpha_bars_prev)*log_x0+noise_t*torch.sqrt(1-alpha_bars_prev-tau_t**2)
        
        noise = sigma*scale*torch.randn(B,Ho,6).to(device)  # [B,Ho, 6]
        # noise = scale*gamma2*torch.randn(B,Ho,6).to(device)  # [B,Ho, 6]
        # noise = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]
        # noise=self.betas[timestep]* scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device) 
        sample=se3.exp(sample)
        # sample=se3.exp(sample+sigma*noise)
        # sample=ndt@nx0
        H_pure_noise=se3.exp(noise)


        # noise = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]
        # H_pure_noise = se3.exp(noise)
        # original_samples=H_pure_noise@reconstructed_H_0
        # H_T = torch.eye(4)[None].expand(B,Ho, -1, -1).to(device) # H_T: [B,Ho,4,4]
        # sample = se3.exp((1. - torch.sqrt(alpha_bars)).unsqueeze(-1).unsqueeze(-1) * se3.log(H_T @ (torch.inverse(original_samples).to(torch.float32)))) @ original_samples.to(torch.float32)

        # perturbation part in eq 34
        return sample,H_pure_noise # sample = A^{k-1}, reconstructed_H_0 = A^{k->0}A^k, see algorithm 2
        
        # noise = torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]
        # noise = se3.exp(noise)
        # return H_pure_noise@sample,noise # sample = A^{k-1}, reconstructed_H_0 = A^{k->0}A^k, see algorithm 2


    def ddim_denoise_decoupled(
        self,
        reconstructed_H_0,   # [B,Ho,4,4] model's estimate of clean H_0
        timestep,            # [B]
        sample,              # [B,Ho,4,4] = H_t
        device,
        abs_to_rel: bool = False,
        predict_h0: bool = True,
        eta: float = 1.0,    # DDIM stochasticity (0 = deterministic)
    ):
        """
        DDIM reverse step consistent with add_noise9_decoupled:

        Forward (decoupled):
        R_t = exp( sqrt(alpha_rot_t) * log(R) )  left-composed with exp( sqrt(1-a_rot_t) * σ_r * ε_r )
        t_t = sqrt(alpha_trn_t) * t              +            sqrt(1-a_trn_t) * σ_t * ε_t

        Reverse (DDIM, per-part):
        eps_r,t are *predicted* from (R_t, t_t) and x0-hat, then
        R_{t-1} = exp( sqrt(a_rot_prev) * log(R0) + sqrt(1-a_rot_prev) * eps_r )
        t_{t-1} = sqrt(a_trn_prev) * t0           + sqrt(1-a_trn_prev) * eps_t
        + optional stochasticity controlled by eta:
            rot: left-compose exp( σ_r * sigma_rot_step * ξ_r )
            trn: add σ_t * sigma_trn_step * ξ_t  (world frame)
        """
        # ----- indices / sizes -----
        t_idx = timestep[0].cpu().item()  # assume uniform t across batch
        prev_idx = max(t_idx - 1, 0)
        B, Ho = sample.shape[0], sample.shape[1]

        if abs_to_rel:
            raise NotImplementedError("abs_to_rel not supported in decoupled variant.")

        if not predict_h0:
            raise NotImplementedError("predict_h0=False not implemented for decoupled DDIM.")

        # ----- schedules (decoupled; fall back to shared if not present) -----
        alpha_rot_t   = getattr(self, "alpha_bars_rot",   self.alpha_bars)[t_idx].to(device)  # scalar
        alpha_trn_t   = getattr(self, "alpha_bars_trans", self.alpha_bars)[t_idx].to(device)  # scalar
        alpha_rot_tm1 = getattr(self, "alpha_bars_rot",   self.alpha_bars)[prev_idx].to(device)
        alpha_trn_tm1 = getattr(self, "alpha_bars_trans", self.alpha_bars)[prev_idx].to(device)

        # broadcast helpers
        def bcast3(x):  # -> [1,1,1] for rot log vectors
            return x.view(1, 1, 1)
        def bcastv(x):  # -> [1,1,1] for trans vectors
            return x.view(1, 1, 1)

        sqrt_aR_t   = torch.sqrt(alpha_rot_t)
        sqrt_aT_t   = torch.sqrt(alpha_trn_t)
        sqrt_1m_aR_t = torch.sqrt(torch.clamp(1.0 - alpha_rot_t, min=1e-12))
        sqrt_1m_aT_t = torch.sqrt(torch.clamp(1.0 - alpha_trn_t, min=1e-12))

        sqrt_aR_tm1   = torch.sqrt(alpha_rot_tm1)
        sqrt_aT_tm1   = torch.sqrt(alpha_trn_tm1)
        sqrt_1m_aR_tm1 = torch.sqrt(torch.clamp(1.0 - alpha_rot_tm1, min=1e-12))
        sqrt_1m_aT_tm1 = torch.sqrt(torch.clamp(1.0 - alpha_trn_tm1, min=1e-12))

        # ----- decompose inputs -----
        H_t   = sample.to(torch.float32)
        R_t   = H_t[..., :3, :3]      # [B,Ho,3,3]
        tt    = H_t[..., :3,  3]      # [B,Ho,3]

        H0hat = reconstructed_H_0.to(torch.float32)
        R0hat = H0hat[..., :3, :3]    # [B,Ho,3,3]
        t0hat = H0hat[..., :3,  3]    # [B,Ho,3]

        # ----- predict per-part noise eps at step t -----
        # Rotation: eps_r_t ~ (log(R_t) - sqrt(aR_t)*log(R0)) / sqrt(1 - aR_t)
        log_Rt  = so3.log(R_t)        # [B,Ho,3]
        log_R0  = so3.log(R0hat)      # [B,Ho,3]
        eps_r_t = (log_Rt - bcast3(sqrt_aR_t) * log_R0) / bcast3(sqrt_1m_aR_t)  # [B,Ho,3]

        # Translation: eps_t_t ~ (t_t - sqrt(aT_t)*t0) / sqrt(1 - aT_t)
        eps_t_t = (tt - bcastv(sqrt_aT_t) * t0hat) / bcastv(sqrt_1m_aT_t)        # [B,Ho,3]

        # ----- deterministic DDIM update to t-1 (eta=0 path) -----
        # R_{t-1} (in log space):
        log_R_tm1 = bcast3(sqrt_aR_tm1) * log_R0 + bcast3(sqrt_1m_aR_tm1) * eps_r_t
        R_tm1     = so3.exp(log_R_tm1)                                              # [B,Ho,3,3]

        # t_{t-1} (linear):
        t_tm1     = bcastv(sqrt_aT_tm1) * t0hat + bcastv(sqrt_1m_aT_tm1) * eps_t_t  # [B,Ho,3]

        # ----- stochasticity (eta) per DDIM: step-specific sigmas -----
        # sigma_step = eta * sqrt( (1 - a_t / a_tm1) * (1 - a_tm1) / (1 - a_t) )
        # handle t==0 by forcing sigma_step=0
        if t_idx == 0:
            sigma_rot_step = torch.tensor(0.0, device=device)
            sigma_trn_step = torch.tensor(0.0, device=device)
        else:
            sigma_rot_step = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_rot_t / alpha_rot_tm1) * (1.0 - alpha_rot_tm1) / (1.0 - alpha_rot_t + 1e-12), min=0.0)
            )
            sigma_trn_step = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_trn_t / alpha_trn_tm1) * (1.0 - alpha_trn_tm1) / (1.0 - alpha_trn_t + 1e-12), min=0.0)
            )

        # sample fresh noise for this step
        eps_r = torch.randn(B, Ho, 3, device=device, dtype=torch.float32)
        eps_t = torch.randn(B, Ho, 3, device=device, dtype=torch.float32)

        # rotation stochastic term: left-compose
        if sigma_rot_step.item() != 0.0:
            R_noise = so3.exp(bcast3(sigma_rot_step * self.sigma_r) * eps_r)    # [B,Ho,3,3]
        else:
            R_noise = torch.eye(3, dtype=torch.float32, device=device).view(1,1,3,3).expand(B, Ho, -1, -1).clone()

        # translation stochastic term: additive in world frame
        t_noise_world = bcastv(sigma_trn_step * self.sigma_t) * eps_t           # [B,Ho,3]

        # compose rotation noise on the left; add translation noise
        R_out = torch.einsum("bhij,bhjk->bhik", R_noise, R_tm1)
        t_out = t_tm1 + t_noise_world

        H_out = torch.eye(4, dtype=torch.float32, device=device).view(1,1,4,4).expand(B, Ho, -1, -1).clone()
        H_out[..., :3, :3] = R_out
        H_out[..., :3,  3] = t_out

        # --- mimic original returns ---
        H_pure_noise = torch.eye(4, dtype=torch.float32, device=device).view(1,1,4,4).expand(B, Ho, -1, -1).clone()
        H_pure_noise[..., :3, :3] = R_noise
        H_pure_noise[..., :3,  3] = t_noise_world

        return H_out, H_pure_noise

    def ddim_denoise(self,
                reconstructed_H_0, # [B,Ho,4,4]
                timestep, # [B]
                sample, # [B,Ho,4,4]
                device,
                abs_to_rel=False,
                predict_h0=True):
        
        timestep = timestep[0].cpu() # scalar
        B = sample.shape[0]
        Ho = sample.shape[1]
        # see algorithm 2, but no longer use A^{k->0}A^k
        # if timestep>0: 
        #     timestep=timestep-1
        alpha_bars = self.alpha_bars[timestep].to(device) # [B]
        
        if abs_to_rel: 
            # reconstructed_H_0=reconstructed_H_0@sample
            raise NotImplementedError
            
        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 


        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim_inverse.py
        alpha_bars = self.alpha_bars[timestep].to(device) # [B]
        eta_t=self.betas[timestep]
        # if timestep>0: 
        #     timestep=timestep-1

        alpha_bars_prev = self.alpha_bars[timestep].to(device) # [B]
        x0=reconstructed_H_0
        xt=sample
        if not predict_h0:
            og_scale=torch.sqrt(1-self.og_alpha_bars[int(timestep*100/self.num_steps)])
            _noise=x0
            print(_noise.shape,int(timestep*100/self.num_steps),timestep,"<<<<FDFSFD")
            _log_noise=torch.sqrt(1-alpha_bars)*se3.log(_noise)/og_scale
            _noise=se3.exp(_log_noise)
            log_x0=se3.log(torch.inverse(_noise)@xt)/torch.sqrt(alpha_bars)
        else:
            log_x0=se3.log(x0)
        log_xt=se3.log(xt)
        noise_t = (log_xt-torch.sqrt(alpha_bars)*log_x0)/torch.sqrt(1-alpha_bars)#torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1) * scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device)  # [B,Ho, 6]

        eta=1 # 0 ,1 or what 
        sigma = eta * ((1 - alpha_bars / alpha_bars_prev) * (1 - alpha_bars_prev) / (1 - alpha_bars)).sqrt() # lucidrains/ddim d3
        dir_t=torch.sqrt(1-alpha_bars_prev)*noise_t
        sample=torch.sqrt(alpha_bars_prev)*log_x0+dir_t
        
        noise = sigma*scale*torch.randn(B,Ho,6).to(device)  # [B,Ho, 6]
        sample=se3.exp(sample)
        H_pure_noise=se3.exp(noise)

        # perturbation part in eq 34
        return sample,H_pure_noise # sample = A^{k-1}, reconstructed_H_0 = A^{k->0}A^k, see algorithm 2


    def ddpm_denoise_decoupled(
        self,
        reconstructed_H_0,   # [B,Ho,4,4] model's estimate of clean H_0 (or reconstruction)
        timestep,            # [B]
        sample,              # [B,Ho,4,4] = H_t
        device,
        abs_to_rel: bool = False,
        predict_h0: bool = True,
    ):
        """
        Reverse (denoise) step consistent with add_noise9_decoupled:

        - Rotation:    R_{t-1} = exp( g0 * log(R0) + g1 * log(R_t) )  then optional left-compose noise exp(sigma_r * g2 * eps_r)
        - Translation: t_{t-1} = g0 * t0 + g1 * t_t                 then optional additive world noise (sigma_t * g2 * eps_t)

        The g0,g1,g2 are your gamma schedules at 'timestep'. We keep the same stochasticity
        pattern as your original ddpm_denoise: an extra gamma2-scaled noise term.
        """
        # --- setup / schedules ---
        t_idx = timestep[0].cpu().item()  # scalar index
        B, Ho = sample.shape[0], sample.shape[1]

        if abs_to_rel:
            raise NotImplementedError("abs_to_rel not supported for decoupled variant yet.")

        # gamma schedules (match your original ddpm_denoise usage)
        gamma0 = self.gamma0[t_idx].to(device)
        gamma1 = self.gamma1[t_idx].to(device)
        # make sure gamma2[-1] = 0.0 like your original code path
        self.gamma2[-1] = 0.0
        gamma2 = self.gamma2[t_idx].to(device)

        # decoupled alpha-bars (used only if you later add an eps-pred branch)
        alpha_rot = getattr(self, "alpha_bars_rot", self.alpha_bars)[timestep].to(device)   # [B]
        alpha_trn = getattr(self, "alpha_bars_trans", self.alpha_bars)[timestep].to(device) # [B]

        # expand scalars to broadcast nicely
        g0 = gamma0.view(1, 1, 1)  # [1,1,1] -> broadcasts over [B,Ho,3]
        g1 = gamma1.view(1, 1, 1)
        g2 = gamma2.item()  # keep as python float for simple mult

        # --- decompose inputs ---
        H_t = sample.to(torch.float32)                 # [B,Ho,4,4]
        R_t = H_t[..., :3, :3]                         # [B,Ho,3,3]
        t_t = H_t[..., :3,  3]                         # [B,Ho,3]

        H0_hat = reconstructed_H_0.to(torch.float32)   # [B,Ho,4,4]
        R0_hat = H0_hat[..., :3, :3]                   # [B,Ho,3,3]
        t0_hat = H0_hat[..., :3,  3]                   # [B,Ho,3]

        # For now we only support predict_h0=True clean-pose prediction (common setting).
        # If you later want eps-prediction, mirror the forward:
        #   R_t = exp(sqrt(alpha_rot)*log(R)), so log(R) = log(R_t)/sqrt(alpha_rot)
        #   t_t = sqrt(alpha_trn)*t,          so t   = t_t/sqrt(alpha_trn)
        if not predict_h0:
            raise NotImplementedError("predict_h0=False branch not implemented for decoupled variant.")

        # --- deterministic part (rotation: log/exp blend; translation: linear blend) ---
        log_R0 = so3.log(R0_hat)        # [B,Ho,3]
        log_Rt = so3.log(R_t)           # [B,Ho,3]
        log_R_tm1 = g0 * log_R0 + g1 * log_Rt
        R_tm1 = so3.exp(log_R_tm1)      # [B,Ho,3,3]

        t_tm1 = g0.squeeze(-1) * t0_hat + g1.squeeze(-1) * t_t   # [B,Ho,3]

        # --- optional stochasticity (mirror your ddpm_denoise gamma2 noise term) ---
        # rotation noise: left-compose in SO(3)
        if g2 != 0.0:
            eps_r = torch.randn(B, Ho, 3, device=device, dtype=torch.float32)  # [B,Ho,3]
            R_noise = so3.exp(self.sigma_r * g2 * eps_r)                       # [B,Ho,3,3]
        else:
            R_noise = torch.eye(3, dtype=torch.float32, device=device).view(1,1,3,3).expand(B, Ho, -1, -1).clone()

        # translation noise: additive in world coords
        if g2 != 0.0:
            eps_t = torch.randn(B, Ho, 3, device=device, dtype=torch.float32)  # [B,Ho,3]
            t_noise_world = self.sigma_t * g2 * eps_t                          # [B,Ho,3]
        else:
            t_noise_world = torch.zeros(B, Ho, 3, dtype=torch.float32, device=device)

        # compose rotation noise on the left (consistent with forward)
        R_out = torch.einsum("bhij,bhjk->bhik", R_noise, R_tm1)  # R_out = R_noise @ R_tm1
        t_out = t_tm1 + t_noise_world

        H_out = torch.eye(4, dtype=torch.float32, device=device).view(1,1,4,4).expand(B, Ho, -1, -1).clone()
        H_out[..., :3, :3] = R_out
        H_out[..., :3,  3] = t_out

        # --- mimic original returns ---
        H_pure_noise = torch.eye(4, dtype=torch.float32, device=device).view(1,1,4,4).expand(B, Ho, -1, -1).clone()
        H_pure_noise[..., :3, :3] = R_noise
        H_pure_noise[..., :3,  3] = t_noise_world

        return H_out, H_pure_noise


    def ddpm_denoise(self,
                reconstructed_H_0, # [B,Ho,4,4]
                timestep, # [B]
                sample, # [B,Ho,4,4]
                device,
                abs_to_rel=False,
                predict_h0=True):
        
        timestep = timestep[0].cpu() # scalar
        B = sample.shape[0]
        Ho = sample.shape[1]
        # see algorithm 2, but no longer use A^{k->0}A^k
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        self.gamma2[-1]=0.0
        gamma2 = self.gamma2[timestep].to(device)
        if abs_to_rel: 
            # reconstructed_H_0=reconstructed_H_0@sample
            raise NotImplementedError
        
        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        
        noise=self.betas[timestep]* scale.unsqueeze(0) * torch.randn(B,Ho, 6).to(device) 
        H_pure_noise=se3.exp(noise)

        x0=reconstructed_H_0
        xt=sample
        if not predict_h0:
            alpha_bars = self.alpha_bars[timestep].to(device) # [B]
            _noise=x0
            log_x0=se3.log(torch.inverse(_noise)@xt)/torch.sqrt(alpha_bars)
        else:
            log_x0=se3.log(x0)
        log_xt=se3.log(xt)
        sample = se3.exp(gamma0 * log_x0 + gamma1 * log_xt)# + scale*gamma2*torch.randn(B,Ho,6).to(device))#scale*torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1)*

        return sample,H_pure_noise # sample = A^{k-1}, reconstructed_H_0 = A^{k->0}A^k, see algorithm 2


    def denoise2(self,
                lie_H_0, # [B,Ho,6]
                timestep, # [B]
                noisy_lie_actions,
                device,
                ):
        
        timestep = timestep[0].cpu() # scalar
        B = lie_H_0.shape[0]
        Ho = lie_H_0.shape[1]
        # see algorithm 2, but no longer use A^{k->0}A^k
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        gamma2 = self.gamma2[timestep].to(device)
        lambda0 = self.lambda0[timestep].to(device)
        lambda1 = self.lambda1[timestep].to(device)
        alpha_bars = self.alpha_bars[timestep].to(device) # [B]

        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        scale=scale.unsqueeze(0)
        # lie_noise=torch.randn(B,Ho,6).to(device)
        # prev_lie_h_t=lambda0*(lie_H_t-lambda1*lie_pred) + gamma2*scale*lie_noise

        noise=torch.randn(B,Ho,6).to(device)
        noisy_lie_actions = gamma0 *lie_H_0 + gamma1 * noisy_lie_actions + scale*gamma2*noise*torch.sqrt(1. - alpha_bars).unsqueeze(-1).unsqueeze(-1)

        return noisy_lie_actions,se3.exp(noisy_lie_actions)
    
    def denoise22(self,
                lie_noise, # [B,Ho,6]
                timestep, # [B]
                noisy_actions,
                noisy_lie_actions,
                device,
                ):
        
        timestep = timestep[0].cpu() # scalar
        B = lie_noise.shape[0]
        Ho = lie_noise.shape[1]
        # see algorithm 2, but no longer use A^{k->0}A^k
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        gamma2 = self.gamma2[timestep].to(device)
        lambda0 = self.lambda0[timestep].to(device)
        lambda1 = self.lambda1[timestep].to(device)
        v_coeff1=self.v_coeff1[timestep].to(device).unsqueeze(-1).unsqueeze(-1)
        v_coeff2=self.v_coeff2[timestep].to(device).unsqueeze(-1).unsqueeze(-1)

        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        scale=scale.unsqueeze(0)
        # lie_noise=torch.randn(B,Ho,6).to(device)
        # prev_lie_h_t=lambda0*(lie_H_t-lambda1*lie_pred) + gamma2*scale*lie_noise

        lie_h0=se3.log(torch.inverse(se3.exp(v_coeff2*lie_noise))@noisy_actions)/v_coeff1
        h0=se3.exp(lie_h0)

        noise=torch.randn(B,Ho,6).to(device)
        noisy_lie_actions = gamma0 *lie_h0 + gamma1 * noisy_lie_actions + scale*gamma2*noise
        
        return noisy_lie_actions,se3.exp(noisy_lie_actions)
    
    # see eq 10 in DiffusionReg paper, (exp are applied to both sides)
    def pre_compute_loss(self,
                H_0, # [B,Ho,4,4]
                timestep, # [B]
                H_t, # [B,Ho,4,4]
                predicted, # [B,Ho,4,4]
                device):
        
        interpolated, _=self.denoise(H_0,timestep,H_t,device)
        
        return interpolated,predicted
    

    def denoise3(self,
                lie_H_t, # [B,Ho,6]
                timestep, # [B]
                lie_v_pred,
                device,
                ):
        
        timestep = timestep[0].cpu() # scalar
        B = lie_H_t.shape[0]
        Ho = lie_H_t.shape[1]
        # see algorithm 2, but no longer use A^{k->0}A^k
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        gamma2 = self.gamma2[timestep].to(device)
        lambda0 = self.lambda0[timestep].to(device)
        lambda1 = self.lambda1[timestep].to(device)
        v_coeff1=self.v_coeff1[timestep].to(device).unsqueeze(-1).unsqueeze(-1)
        v_coeff2=self.v_coeff2[timestep].to(device).unsqueeze(-1).unsqueeze(-1)

        lie_noise_pred = v_coeff1 * lie_v_pred + v_coeff2 * lie_H_t
        lie_x0_pred = v_coeff1 * lie_H_t - v_coeff2 * lie_v_pred

        scale = torch.cat([torch.ones(3) * self.sigma_t, torch.ones(3) * self.sigma_r])[None].to(device)  # [1, 6] 
        # prev_h_t = se3.exp(gamma0 * lie_x0_pred + gamma1 * lie_H_t + scale*gamma2*torch.randn(B,Ho,6).to(device))
        prev_lie_h_t=gamma0 * lie_x0_pred + gamma1 * lie_H_t + gamma2*lie_noise_pred
        # lie_noise=torch.randn(B,Ho,6).to(device)
        # prev_lie_h_t=gamma0 * lie_x0_pred + gamma1 * lie_H_t + gamma2*scale*lie_noise
        prev_h_t = se3.exp(prev_lie_h_t)

        # lie_noise=torch.randn(B,Ho,6).to(device)
        # prev_lie_h_t=lambda0*(lie_H_t-lambda1*lie_noise_pred) #+ gamma2*scale*lie_noise
        # prev_h_t = se3.exp(prev_lie_h_t)

        return prev_lie_h_t,prev_h_t,se3.exp(lie_x0_pred)
    
    # see eq 10 in DiffusionReg paper, (exp are applied to both sides)
    def pre_compute_loss(self,
                H_0, # [B,Ho,4,4]
                timestep, # [B]
                H_t, # [B,Ho,4,4]
                predicted, # [B,Ho,4,4]
                device):
        
        interpolated, _=self.denoise(H_0,timestep,H_t,device)
        
        return interpolated,predicted
    

class DiffusionScheduler_vanilla(torch.nn.Module):

    def __init__(self, num_steps=100, beta_1=1e-4, beta_T=0.05, sigma_r_inv=0.05, sigma_t_inv=0.03,
    sigma_r_equiv=0.2, sigma_t_equiv=0.001,  mode='cosine'):
        super().__init__()
        self.num_steps: int = num_steps
        self.beta_1: float = beta_1
        self.beta_T: float = beta_T
        self.sigma_r_inv: float = sigma_r_inv
        self.sigma_t_inv:float = sigma_t_inv
        self.sigma_r_equiv: float = sigma_r_equiv
        self.sigma_t_equiv:float = sigma_t_equiv
        # self.mode = ["linear", "cosine"]
        self.mode = mode
        self.S = 0.008
        self.betas = torch.zeros([self.num_steps + 1])  # 初始化 betas
        device = torch.device("cuda")
        
        if self.mode == 'linear' or self.mode == 'custom':
            betas = torch.linspace(self.beta_1, self.beta_T, steps=self.num_steps)
            self.betas[1:] = betas     # Padding
        elif self.mode == 'cosine':
            def betas_fn(s):
                T = self.num_steps
                def f(t, T, s):
                    return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
                alphas = [f(0, T, s)]
                for t in range(1, T + 1):
                    alphas.append(f(t, T, s) / alphas[-1])
                betas = [1 - alpha / alphas[0] for alpha in alphas]
                return [min(beta, 0.999) for beta in betas]
            betas = betas_fn(s=self.S)
            self.betas = torch.FloatTensor(betas)
        
        self.alphas = 1 - self.betas
        log_alphas = torch.log(self.alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        self.alpha_bars = log_alphas.exp().to(device)

        self.gamma0 = torch.zeros_like(self.betas)
        self.gamma1 = torch.zeros_like(self.betas)
        self.gamma2 = torch.zeros_like(self.betas)
        
        for t in range(2, self.num_steps + 1):  # 2 to T
            self.gamma0[t] = self.betas[t] * torch.sqrt(self.alpha_bars[t - 1]) / (1. - self.alpha_bars[t])
            self.gamma1[t] = (1. - self.alpha_bars[t - 1]) * torch.sqrt(self.alphas[t]) / (1. - self.alpha_bars[t])
            self.gamma2[t] = (1. - self.alpha_bars[t - 1]) * self.betas[t] / (1. - self.alpha_bars[t])
        if self.mode == 'custom':
            gamma1_control = np.array([[0,0.2],[0.1,0.05],[0.2,0.05],[1,0]])
            gamma2_control = np.array([[0,0.8],[0.1,0.95],[0.2,0.95],[1,1]])
            self.gamma0[2:] = torch.tensor(bezier_curve(gamma1_control, np.linspace(0, 1, self.num_steps-1)))
            self.gamma1[2:] = torch.tensor(bezier_curve(gamma2_control, np.linspace(0, 1, self.num_steps-1)))

    def set_timesteps(self,num_steps):
        self.num_steps = num_steps
    
    def add_noise_inv(self,
        original_samples: torch.FloatTensor, # [1, 1, 4, 4]
        timesteps: torch.IntTensor,
        device):
        batch_size = original_samples.shape[0]
        T_a = original_samples.shape[1]
        original_samples = original_samples.reshape(-1, 4, 4)  # [B * horizon, 4, 4]
        scale = torch.cat([torch.ones(3) * self.sigma_r_inv, torch.ones(3) * self.sigma_t_inv])[None].to(device)  # [1, 6]
        alpha_bars = self.alpha_bars[timesteps].to(device)[:, None]  # [B, 1]
 
        H_T = torch.eye(4)[None].expand(batch_size*T_a, -1, -1).to(device)
        # interpolation function F 
        
        F = se3.exp((1. - torch.sqrt(alpha_bars)) * se3.log(H_T @ torch.inverse(original_samples)))

        # Perturbation 
        noise_origin = torch.randn(batch_size, 6).repeat(T_a, 1, 1).transpose(0, 1).reshape(-1, 6)  # [B * horizon, 6]
        noise = torch.sqrt(1. - alpha_bars) * scale * noise_origin.to(device)  # [B, 6]

        # Perturbation * interpolation
        H_noise = se3.exp(noise) @ F  # [B * horizon, 4, 4]
        H_t_noise = H_noise @ original_samples # [B * horizon, 4, 4]

        #! chenrui: 这里的加噪跟点云配准那篇一样
        return H_t_noise, H_noise

    def add_noise_equiv(self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
        device,):
        batch_size = original_samples.shape[0]
        T_a = original_samples.shape[1]
        original_samples = original_samples.reshape(-1, 4, 4)  # [B * horizon, 4, 4]
        alpha_bars = self.alpha_bars[timesteps].to(device)[:, None]  # [B, 1]

        # Perturbation 
        scale = torch.cat([torch.ones(3) * self.sigma_r_equiv, torch.ones(3) * self.sigma_t_equiv])[None].to(device)  # [1, 6]
        noise_origin = torch.randn(batch_size, 6).repeat(T_a, 1, 1).transpose(0, 1).reshape(-1, 6)  # [B * horizon, 6]

        noise = scale * noise_origin.to(device)  # [B, 6]

        # Perturbation
        # print(noise)
        H_noise = se3.exp(noise) # [B * horizon, 4, 4]
        # print(H_noise)
        H_t_noise = H_noise @ original_samples # [B * horizon, 4, 4]

        #! chenrui: 这里的加噪跟点云配准那篇一样
        return H_t_noise, H_noise

    def denoise(self,
                model_output, 
                timestep,  
                sample, 
                device):
        # print("model_output",model_output.shape, # [1, 4, 4]
        #       "timestep.shape", timestep.shape, # [4, 100, 1]
        #       "sample.shape",sample.shape) # [4, 100, 9]
        # exit(0)
        timestep = timestep[0].cpu()
        B = model_output.shape[0]
        H_0 = (torch.inverse(model_output) @ sample)
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        sample = se3.exp(gamma0 * se3.log(H_0) + gamma1 * se3.log(sample))
        return sample

    def denoise_equiv(self,
                model_output, # [B, 4, 4]
                timestep,  
                sample, # (B, pred_horizon, 4, 4)
                device):

        timestep = timestep[0].to(device)
        B = model_output.shape[0]

        H_0 = torch.inverse(model_output) @ sample.to(device)

        return H_0
