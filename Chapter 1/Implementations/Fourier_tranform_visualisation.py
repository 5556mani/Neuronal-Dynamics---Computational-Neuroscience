import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Intuition Behind Convolution

# u(t) = ∫ k(s) * I(t-s) ds
#
# At the current time (t), look back through all past inputs.
# Each past input contributes according to how much the system
# still "remembers" it (given by k(s)).
# Add all these surviving contributions to get the output u(t).

# k(s) is the system's memory.
# k(0) = 1 because the present is remembered completely.
# As we go further into the past, k(s) decreases because older
# inputs gradually fade away.

# The kernel is NOT fixed.
# As time moves forward, the peak of the kernel also moves forward,
# always centered on the current time and looking backward.
# Convolution is simply this kernel sliding across the input.

# At every position:
#   1. Overlay the kernel on the input.
#   2. Multiply overlapping values.
#   3. Sum them.
#   4. That sum is one point of the output.
# Repeating this for every time instant draws the red output curve.

# Frequency Domain:
# Instead of sliding through time, decompose the input into sine waves.
# The filter simply scales each frequency (Output = Input × Filter).
# Combining these scaled sine waves reconstructs exactly the same
# output obtained from convolution in the time domain.

# --- 1. PHYSICAL PARAMETERS ---
pulse_width = 20.0
tau_m = 10.0      # System time constant (memory sluggishness)
t_max = 120.0     # Total simulation time
dt = 0.5          # Time step for numerical integration
pulse_start = 40.0

# --- 2. TIME DOMAIN COMPUTATION (Numerical Convolution) ---
tau = np.arange(0, t_max, dt)

# Input I(tau): Square pulse
I = np.where((tau >= pulse_start) & (tau < pulse_start + pulse_width), 1.0, 0.0)

# Pre-compute the output u(t) via numerical convolution to keep the animation fast
# u(t) = sum [ I(s) * k(t-s) * dt ]
u = np.zeros_like(tau)
for i, current_t in enumerate(tau):
    # s is the integration variable up to current_t
    s = tau[:i+1]
    # k(t-s) is the kernel looking backward
    k_s = np.exp(-(current_t - s) / tau_m)
    # Numerical integral
    u[i] = np.sum(I[:i+1] * k_s) * dt 
    
# Normalize output roughly to 1.0 for visual scaling against the input
u = u / tau_m 

# --- 3. FREQUENCY DOMAIN COMPUTATION (Analytical) ---
w = np.linspace(0.1, 50, 400) # Angular frequency (rad/s)

# Input Spectrum |I(w)|: Fourier transform of a square pulse (Sinc function)
# (Scaled down slightly for visualization)
I_w = np.abs(np.sin(w * pulse_width / 2) / (w / 2)) * 0.2

# Filter Curve |k(w)|: Fourier transform of exponential decay e^(-t/tau_m)
K_w = 1.0 / np.sqrt(1.0 + (w * tau_m)**2)

# Output Spectrum |u(w)|: Multiplication in frequency domain
U_w = I_w * K_w

# --- 4. PLOTTING AND ANIMATION SETUP ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title('Linear System: Time vs Frequency')

# Set up Time Domain axis (Top)
ax1.set_xlim(0, t_max)
ax1.set_ylim(0, 1.2)
ax1.set_title(r'Time Domain: Sliding Local Convolution  $u(t) = \int \kappa(t-\tau) I(\tau) d\tau$')
ax1.set_xlabel(r'Time History ($\tau$) [ms]')
ax1.set_ylabel('Amplitude')
ax1.grid(True, linestyle='--', alpha=0.6)

# Time domain static lines
ax1.plot(tau, I, color='#3b82f6', lw=2, label=r'Input $I(\tau)$')

# Time domain animated lines
line_kernel, = ax1.plot([], [], color='#94a3b8', lw=2, linestyle='--', label=r'Sliding Kernel $\kappa(t-\tau)$')
fill_kernel = ax1.fill_between([], [], color='#cbd5e1', alpha=0.5)
line_out, = ax1.plot([], [], color='#ef4444', lw=3, label=r'Output $u(t)$')
ax1.legend(loc='upper right')

# Set up Frequency Domain axis (Bottom)
ax2.set_xlim(0, 50)
ax2.set_ylim(0, np.max(K_w) * 1.1)
ax2.set_title(r'Frequency Domain: Global Multiplication  $\hat{u}(\omega) = \hat{\kappa}(\omega) \times \hat{I}(\omega)$')
ax2.set_xlabel(r'Frequency ($\omega$) [rad/s]')
ax2.set_ylabel('Magnitude')
ax2.grid(True, linestyle='--', alpha=0.6)

# Frequency domain lines (all static)
ax2.plot(w, I_w, color='#3b82f6', lw=2, label=r'Input Spectrum $|\hat{I}(\omega)|$')
ax2.plot(w, K_w, color='#94a3b8', lw=2, linestyle='--', label=r'Filter Curve $|\hat{\kappa}(\omega)|$')
ax2.plot(w, U_w, color='#ef4444', lw=3, label=r'Output Spectrum $|\hat{u}(\omega)|$')
ax2.legend(loc='upper right')

plt.tight_layout()

# --- 5. ANIMATION LOOP ---
def init():
    line_kernel.set_data([], [])
    line_out.set_data([], [])
    return line_kernel, line_out

def update(frame_idx):
    current_time = tau[frame_idx]
    
    # 1. Update the sliding kernel
    # It only exists for time <= current_time
    valid = tau <= current_time
    k_slide = np.zeros_like(tau)
    k_slide[valid] = np.exp(-(current_time - tau[valid]) / tau_m)
    line_kernel.set_data(tau, k_slide)
    
    # Update the shaded area under the kernel
    global fill_kernel
    fill_kernel.remove()
    fill_kernel = ax1.fill_between(tau, k_slide, where=valid, color='#cbd5e1', alpha=0.5)
    
    # 2. Update the output (draw up to current_time)
    line_out.set_data(tau[:frame_idx], u[:frame_idx])
    
    return line_kernel, line_out

# Create the animation
# frames = len(tau) generates an index for every time step
# interval = 20 ms between frames
ani = animation.FuncAnimation(fig, update, frames=len(tau),
                              init_func=init, blit=False, interval=20, repeat=False)

plt.show()