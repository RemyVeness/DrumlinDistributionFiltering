#impport
import random
import numpy as np
import matplotlib.pyplot as plt
#from scipy.interpolate import make_interp_spline 
from scipy.interpolate import make_smoothing_spline
from scipy.stats import norm
from scipy.stats import ks_2samp
from scipy.stats import chisquare
from statsmodels.distributions.empirical_distribution import ECDF


def resample (in_list,scale): 
    """
    Resample
    Args:
        in_list (list): List to be resampled to be resamples; 
        scale (int): number of resamples to take
        Returns:
            list: Resampled output. 
    
    """    
    output = []
    for i in range(0,scale): 
        j = random.randrange(len(in_list))
        output.append(in_list[j])
    return output

sig_hist_ar = np.genfromtxt(
    'D:/SSD_INT_leaving/SSD_INT/BIIS/HPC_dump/Summary_Figures/Sig_Hist_Vel_ar.csv', 
    delimiter=","
)

sig_anti_hist_ar = np.genfromtxt(
    'D:/SSD_INT_leaving/SSD_INT/BIIS/HPC_dump/Summary_Figures/Sig_Hist_Antivel_ar.csv', 
    delimiter=","
)

#bootstrapping variable
null_filtered_sums = []

for k in range (0, 1000):
    # Combine all values into one list and resample 
    vel_list = sig_hist_ar.flatten().tolist()
    vel_len = len(vel_list)
    anti_len = (sig_anti_hist_ar.shape)[0]
    
    # Resample random list to same size as velocity list
    randlist = []
    
    for _ in range(0,vel_len):        
        j = random.randrange(anti_len)
        randlist.append((sig_anti_hist_ar[j]))

    
    if (len(randlist)) != vel_len:
        print(
            'Resampling error: Sample length: ' + str(len(randlist)) + 
            ' not same size as datset length: ' + str (vel_len)
        )
    
    # Define bin edges
    bin_edges=np.arange(min(vel_list), max(vel_list) + 10, 10)
    
    # Compute histograms
    fs_hist, fs_bin_edges = np.histogram(vel_list, bins=bin_edges)
    noise_hist, noise_bin_edges = np.histogram(randlist, bins=bin_edges)
    
    # Bin mids
    bin_mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    noise_bin_mids = 0.5 * (noise_bin_edges[1:] + noise_bin_edges[:-1])
    
    noise_hist = np.append(650, noise_hist)
    noise_hist = np.delete(noise_hist,5)
    noise_hist = noise_hist.astype(float)
    noise_hist[1:7] *= 1.1                   # Changed from [1:8]
    noise_hist[4:6] *= 1.3
#    noise_hist [30:]*= 1.5                  # Swapped for [35:]
    noise_hist [30:34]*= 1.4
    noise_hist [35:]*= 1.5
    
    
    #a = 1.08 # Scale 
    #k = 1.1 # Power/shapex = -15 # Offset
    # Adjust shape of distribution and filter
    #filtered_hist = fs_hist - (a*(noise_hist ** k)+x)
    # Compute observed filtered histogram
    filtered_hist = fs_hist - noise_hist
    filtered_hist_clipped = np.clip(filtered_hist, 0, None)

    # The summary metric for observed dataset (same as null case)
    null_filtered_sums.append(filtered_hist_clipped.sum())
    
    # Summation for plotting later
    if k == 0:
        sum_hist = filtered_hist
    else:
        sum_hist = sum_hist + filtered_hist    

filtered_hist = sum_hist  
filtered_hist_clipped = np.clip(filtered_hist, 0, None)    

# compute the observed filtered histogram
vel_list = sig_hist_ar.flatten().tolist()
vel_len = len(vel_list)
anti_len = (sig_anti_hist_ar.shape)[0]

# Keep only possitive values
mask = filtered_hist >= 0
bin_mids_pos = bin_mids[mask]
filtered_hist_pos = filtered_hist[mask]
bin_edges_pos = fs_bin_edges[:-1][mask]

# Normalize
if filtered_hist_pos.max() > 0:
    filtered_hist_pos = filtered_hist_pos / filtered_hist_pos.max()
    
# # Interp  spline 
# spl = make_interp_spline(bin_mids_pos, filtered_hist_pos, k=3)
# xnew = np.linspace(bin_mids_pos.min(), bin_mids_pos.max(), 300)
# filtered_hist_smooth = spl(xnew)

# Smooth spline (not follow)
spline_smooth = make_smoothing_spline(bin_mids_pos, filtered_hist_pos, lam=1e4)  # adjust lam for smoothness
y_smooth = spline_smooth(bin_mids_pos)  # smooth trend, not exact fit

# Clip to x ≤ 420
clip_mask = bin_mids <= 420

bin_mids_clip      = bin_mids[clip_mask]
filtered_hist_clip = np.clip(filtered_hist, 0, None)[clip_mask]  # keep only ≥0
noise_hist_clip    = noise_hist[clip_mask]

# Ensure weights sum to > 0
weights = filtered_hist_clip
if np.sum(weights) == 0:
    raise ValueError("All clipped histogram values are zero — nothing to fit.")

# Weighted mean/std for normal fit
mean = np.average(bin_mids_clip, weights=weights)
variance = np.average((bin_mids_clip - mean) ** 2, weights=weights)
std = np.sqrt(variance)

# Create fitted normal PDF across clipped x range
x = np.linspace(bin_mids_clip.min(), bin_mids_clip.max(), 300)
p = norm.pdf(x, mean, std)
# Normalise to overlay with histogram
p = p / np.max(p)

print(f"Weighted Normal fit: μ = {mean:.2f}, σ = {std:.2f}")

# Calculate mode
modal_bin_index = np.argmax(filtered_hist)
modal_bin_start = bin_edges[modal_bin_index]
modal_bin_end = bin_edges[modal_bin_index + 1]
modal_bin_mid = 0.5 * (modal_bin_start + modal_bin_end)
smooth_mode = bin_mids_pos[np.argmax(y_smooth)]

print('smooth mode = ' + str(smooth_mode))
print('bin mode = ' + str(modal_bin_mid))

# Perform bootstrap test

observed_sum = filtered_hist_clipped.sum()
boot_p_value = np.mean(np.array(null_filtered_sums) >= observed_sum)  # bootstrap p-value
print(f"Bootstrap p-value for filtered histogram significance = {boot_p_value:.2e}")

# Perform K-S Test

# Expand histograms into pseudo-samples
scale_factor = 50

signal_samples = np.repeat(bin_mids_clip,
                           (filtered_hist_clip * scale_factor).astype(int))
noise_samples  = np.repeat(bin_mids_clip,
                           (noise_hist_clip * scale_factor).astype(int))

# KS test
ks_stat, p_value = ks_2samp(signal_samples, noise_samples)

print(f"KS statistic = {ks_stat:.4f}, p = {p_value:.2e}")
if p_value < 0.05:
    print("Distributions differ significantly (signal ≠ noise).")
else:
    print("No significant difference (signal may be explained by noise).")
    
    
# ChiSquare test
noise_hist_scaled = noise_hist * (fs_hist.sum() / noise_hist.sum())

# Mask out zero bins in expected for c2
mask_nonzero = noise_hist_scaled > 0
fs_hist_nonzero = fs_hist[mask_nonzero]
noise_hist_nonzero = noise_hist_scaled[mask_nonzero]

# Chi2 test
if len(fs_hist_nonzero) < 2:
    print("Not enough nonzero bins for chi-square test.")
    chi2, p_chi = np.nan, np.nan
else:
    # Renormalize expected to have same total as observed after masking
    noise_hist_nonzero = noise_hist_nonzero * (fs_hist_nonzero.sum() / noise_hist_nonzero.sum())

    chi2, p_chi = chisquare(f_obs=fs_hist_nonzero, f_exp=noise_hist_nonzero)
    print(f"Chi-square = {chi2:.2f}, p = {p_chi:.3e}")

# Test plot
fig, axs = plt.subplots(1,2,figsize = (14,7))
axs[0].set_xlim(0,1000)  
axs[0].set_ylim(0,1000)
axs[0].plot(bin_mids, fs_hist, color = 'firebrick', label='Unfiltered match')
#axs[0].plot(bin_mids, (a*(noise_hist**k)+x), color = 'black')
axs[0].plot(bin_mids, noise_hist, color = 'black', label='Filtered match')
axs[0].tick_params(axis='x', direction='in')
axs[0].tick_params(axis='y', direction='in')
axs[0].tick_params(right=True, top=True)
#axs[0].text(.7, .9, str(a)+r'*n$^{%f}$' % (k)+' '+ str(x),transform=axs[0].transAxes)
axs[0].legend(loc='upper right')
axs[0].text(.95, .95, 'a', size='x-large',transform=axs[0].transAxes)
axs[0].set_xlabel('Velocity during match (ma$^{-1}$)')
axs[0].set_ylabel('Frequency')

# Plot filtered histogram
# fig, ax = plt.subplots(figsize = (7,7))
# plt.bar(
#     bin_edges[:-1],filtered_hist,width = 25,
#     color ='gold', edgecolor ='grey', linewidth = 0.5
# )
# plt.xlim(0,500)  
# plt.ylim(0,100)
# ax.tick_params(axis='x', direction='in')
# ax.tick_params(axis='y', direction='in')
# ax.tick_params(right=True, top=True)
# plt.xlabel('Velocity during match (ma$^{-1}$)')
# plt.ylabel('Frequency')
# plt.savefig(
#     "C:/Users/rv3530/OneDrive - Sheffield Hallam University/Research/"
#     "Drumlin_paper/Figures/All_Thk_filtered_hist.png"
# )

# Make Extra stats and figures

# Plot smoothed frequency polygon
#fig, ax = plt.subplots(figsize = (7,7))
axs[1].set_xlim(0,420)  
axs[1].set_ylim(0,1.1)
axs[1].tick_params(axis='x',direction ='in')
axs[1].tick_params(axis='y',direction ='in')
axs[1].tick_params(right=True, top=True)
#axs[1].fill_between(bin_mids_pos,y_smooth,color='firebrick',alpha=0.5) # plot a shaded polygon 
axs[1].plot(x,p,'k' , linewidth=0.7)
#axs[1].plot(bin_mids_pos,filtered_hist_pos, marker='+', color='black', linestyle = 'None')
axs[1].bar(bin_edges_pos,filtered_hist_pos,width = 10, color ='firebrick', edgecolor ='grey', linewidth = 0.5)
axs[1].vlines(x=smooth_mode, ymin=0, ymax=y_smooth[np.argmax(y_smooth)], color='grey', linewidth = 0.7) # plot vertical line
axs[1].text(.95, .95, 'b', size='x-large', transform=axs[1].transAxes)
axs[1].set_xlabel('Velocity during match (ma$^{-1}$)')
axs[1].set_ylabel('Normalised frequency')
plt.show()
plt.savefig(
    "C:/Users/rv3530\OneDrive - Sheffield Hallam University/"
    "Research/Drumlin_paper/Fig_2_filtered_hist.png"
)

# --- Expand histograms into pseudo-samples ---
signal_samples = np.repeat(bin_mids_clip,
                           (filtered_hist_clip * scale_factor).astype(int))
noise_samples = np.repeat(bin_mids_clip,
                          (noise_hist_clip * scale_factor).astype(int))
raw_samples = np.repeat(bin_mids,
                        (fs_hist * scale_factor).astype(int))  # unfiltered

# --- Compute ECDFs ---
ecdf_signal = ECDF(signal_samples)
ecdf_noise  = ECDF(noise_samples)
ecdf_raw    = ECDF(raw_samples)

# --- Plot ECDFs ---
plt.figure(figsize=(7,7))
plt.step(ecdf_noise.x, ecdf_noise.y, where='post',
         label="Baseline", color="black")
plt.step(ecdf_raw.x, ecdf_raw.y, where='post',
         label="Unfiltered match", color="firebrick")
plt.step(ecdf_signal.x, ecdf_signal.y, where='post',
         label="Filtered match", color="goldenrod")
plt.xlabel("Velocity (ma$^{-1}$)")
plt.ylabel("Cumulative probability")
plt.xlim(0,420)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
plt.savefig(
    "C:/Users/rv3530\OneDrive - Sheffield Hallam University/"
    "Research/Drumlin_paper/Fig_0_Cumulative_probablity.png"
)

# KS test: Unfiltered vs Noise (≤420) 

# Clip both datasets to ≤420
clip_mask_unf = bin_mids <= 420
bin_mids_unf_clip = bin_mids[clip_mask_unf]
unfiltered_hist_clip = fs_hist[clip_mask_unf]

noise_clip_mask = noise_bin_mids <= 420
bin_mids_noise_clip = noise_bin_mids[noise_clip_mask]
noise_hist_clip = noise_hist[noise_clip_mask]

# Expand into pseudo-samples
scale_factor = 50
unfiltered_samples = np.repeat(bin_mids_unf_clip,
                               (unfiltered_hist_clip * scale_factor).astype(int))
noise_samples = np.repeat(bin_mids_noise_clip,
                          (noise_hist_clip * scale_factor).astype(int))

# KS test
ks_stat_unf_noise, p_value_unf_noise = ks_2samp(unfiltered_samples, noise_samples)
print(f"[Unfiltered vs Noise] KS statistic = {ks_stat_unf_noise:.4f}, p = {p_value_unf_noise:.2e}")

#  Chi-square test: Unfiltered vs Noise (≤420) 

# Scale noise histogram to have same total as unfiltered
noise_hist_scaled = noise_hist_clip * (unfiltered_hist_clip.sum() / noise_hist_clip.sum())

# Mask zeros
mask_nonzero = noise_hist_scaled > 0
unfiltered_nonzero = unfiltered_hist_clip[mask_nonzero]
noise_nonzero = noise_hist_scaled[mask_nonzero]

if len(unfiltered_nonzero) < 2:
    chi2_unf_noise, p_chi_unf_noise = np.nan, np.nan
    print("Not enough nonzero bins for unfiltered vs noise chi-square.")
else:
    chi2_unf_noise, p_chi_unf_noise = chisquare(f_obs=unfiltered_nonzero, f_exp=noise_nonzero)
    print(f"[Unfiltered vs Noise] Chi-square = {chi2_unf_noise:.2f}, p = {p_chi_unf_noise:.3e}")
