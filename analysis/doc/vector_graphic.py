import numpy as np
import matplotlib.pyplot as plt
import imageio

def create_complex_vector_and_freq_animation(
    filename,
    num_frames=100,
    initial_phase_step=0.1,
    amplitude_base=1.0,
    amplitude_mod=0.5,
    mod_rate=2,
    fps=10
):
    """
    Create an animated GIF showing a rotating, amplitude-modulated complex vector.
    
    The top plot shows the vector in the complex plane, now with red and blue dots on the real and imaginary axes.
    The middle plot shows the real value over time with a red dot marking the latest value.
    The bottom plot shows the cumulative phase and instantaneous frequency over time.
    
    Parameters:
      filename (str): Name of the output GIF file.
      num_frames (int): Total number of frames.
      initial_phase_step (float): Base phase step for rotation; this step is linearly increased.
      amplitude_base (float): The base amplitude of the vector.
      amplitude_mod (float): The modulation depth (added sinusoidally to amplitude).
      mod_rate (float): Number of modulation cycles over the full animation.
      fps (int): Frames per second in the GIF.
    """
    # --- Precompute Data for All Frames ---
    phases = np.zeros(num_frames)
    amplitudes = np.zeros(num_frames)
    real_vals = np.zeros(num_frames)
    inst_freq = np.zeros(num_frames)
    
    current_phase = 0.0
    for i in range(num_frames):
        phase_step = initial_phase_step * (1 + i/num_frames)
        current_phase += phase_step
        phases[i] = current_phase
        
        # Amplitude modulated by a sinusoid.
        amplitude = amplitude_base + amplitude_mod * np.sin(2 * np.pi * mod_rate * i / num_frames)
        amplitudes[i] = amplitude
        
        # Compute the complex vector.
        vec = amplitude * np.exp(1j * current_phase)
        real_vals[i] = vec.real
        
        # Instantaneous frequency approximated by the phase difference.
        if i == 0:
            inst_freq[i] = phase_step
        else:
            inst_freq[i] = phases[i] - phases[i-1]
    
    # --- Precompute Fixed Axis Limits ---
    # For the middle subplot (Real Value over Time).
    real_min = real_vals.min()
    real_max = real_vals.max()
    real_margin = 0.1 * (real_max - real_min)
    real_ylim = (real_min - real_margin, real_max + real_margin)
    
    # For the bottom subplot (Phase and Instantaneous Frequency).
    phase_min = phases.min()
    phase_max = phases.max()
    phase_margin = 0.05 * (phase_max - phase_min)
    phase_ylim = (phase_min - phase_margin, phase_max + phase_margin)
    
    freq_min = inst_freq.min()
    freq_max = inst_freq.max()
    freq_margin = 0.1 * (freq_max - freq_min)
    freq_ylim = (freq_min - freq_margin, freq_max + freq_margin)
    
    # x-axis limit (frame index) for the bottom two plots.
    xlim = (0, num_frames - 1)
    
    frames_list = []  # list to collect frames

    # --- Create Frames for the Animation ---
    # Reset current_phase to ensure consistency with precomputed phases.
    current_phase = 0.0
    for i in range(num_frames):
        phase_step = initial_phase_step * (1 + i/num_frames)
        current_phase += phase_step
        # Use precomputed phase and amplitude
        vec = amplitudes[i] * np.exp(1j * phases[i])
        
        # Create figure with three vertically stacked subplots.
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        plt.subplots_adjust(hspace=0.4)

        # --- Top subplot: Complex Plane ---
        ax = axs[0]
        max_val = amplitude_base + abs(amplitude_mod) + 0.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Complex Plane")
        ax.set_aspect('equal')
        
        # Draw a reference circle at the base amplitude.
        circle = plt.Circle((0, 0), amplitude_base, color='gray', fill=False, linestyle='--')
        ax.add_artist(circle)
        
        # --- Draw solid thin gray axis lines ---
        ax.axhline(y=0, color='gray', linewidth=1)  # Horizontal axis (Real axis)
        ax.axvline(x=0, color='gray', linewidth=1)  # Vertical axis (Imaginary axis)

        # --- Draw the vector as an arrow using quiver for better visibility ---
        ax.quiver(0, 0, vec.real, vec.imag, angles='xy', scale_units='xy', scale=1, color='black', width=0.005, headwidth=6, headlength=8)

        # --- Added dots on the axes: ---
        ax.plot(vec.real, 0, 'ro')  # Red dot on the real axis.
        ax.plot(0, vec.imag, 'bo')  # Blue dot on the imaginary axis.

        # --- Add thin gray dotted lines for decomposition visualization ---
        ax.plot([vec.real, vec.real], [0, vec.imag], 'gray', linestyle='dotted', linewidth=1)  # Vertical line
        ax.plot([0, vec.real], [vec.imag, vec.imag], 'gray', linestyle='dotted', linewidth=1)  # Horizontal line

        # --- Middle subplot: Real Value over Time ---
        ax = axs[1]
        ax.plot(range(i+1), real_vals[:i+1], 'r-')
        # Add a red dot at the most recent real value.
        ax.plot(i, real_vals[i], 'ro')
        ax.set_xlim(xlim)
        ax.set_ylim(real_ylim)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Real Value")
        ax.set_title("Real Value of the Vector Over Time")
        ax.grid(True)

        # --- Bottom subplot: Phase and Instantaneous Frequency Over Time ---
        ax1 = axs[2]
        ax1.plot(range(i+1), phases[:i+1], 'g-', label="Phase")
        ax1.set_xlim(xlim)
        ax1.set_ylim(phase_ylim)
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Phase", color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.set_title("Phase and Instantaneous Frequency Over Time")
        
        # Twin y-axis for the instantaneous frequency.
        ax2 = ax1.twinx()
        ax2.plot(range(i+1), inst_freq[:i+1], 'm--', label="Inst. Frequency")
        ax2.set_ylim(freq_ylim)
        ax2.set_ylabel("Instantaneous Frequency", color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        
        # Render the figure and extract the image.
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())
        image = image[..., :3]  # remove alpha channel
        frames_list.append(image)
        plt.close(fig)

    # --- Save as Animated GIF ---
    imageio.mimsave(filename, frames_list, fps=fps)
    print(f"Animation saved as {filename}")


def create_complex_vector_animation(
    filename,
    num_frames=100,
    initial_phase_step=0.1,
    amplitude_base=1.0,
    amplitude_mod=0.5,
    mod_rate=2,
    fps=10
):
    """
    Create an animated GIF showing a rotating, amplitude-modulated complex vector.
    
    The top subplot shows the complex vector in the complex plane with added dots on the real and imaginary axes.
    The middle subplot shows the real value over time with a red dot marking the latest point.
    The bottom subplot shows the imaginary value over time with a blue dot marking the latest point.
    
    Parameters:
      filename (str): Name of the output GIF file.
      num_frames (int): Total number of frames.
      initial_phase_step (float): Base phase step for rotation; this step is linearly increased.
      amplitude_base (float): The base amplitude of the vector.
      amplitude_mod (float): The modulation depth (added sinusoidally to amplitude).
      mod_rate (float): Number of modulation cycles over the full animation.
      fps (int): Frames per second in the GIF.
    """
    # --- Precompute Data for All Frames ---
    phases = np.zeros(num_frames)
    amplitudes = np.zeros(num_frames)
    real_vals = np.zeros(num_frames)
    imag_vals = np.zeros(num_frames)
    
    current_phase = 0.0
    for i in range(num_frames):
        phase_step = initial_phase_step * (1 + i/num_frames)
        current_phase += phase_step
        phases[i] = current_phase
        
        # Amplitude modulated by a sinusoid.
        amplitude = amplitude_base + amplitude_mod * np.sin(2 * np.pi * mod_rate * i / num_frames)
        amplitudes[i] = amplitude
        
        # Compute the complex vector.
        vec = amplitude * np.exp(1j * current_phase)
        real_vals[i] = vec.real
        imag_vals[i] = vec.imag
    
    # --- Precompute Fixed Axis Limits ---
    # For the middle subplot (Real Value over Time).
    real_min = real_vals.min()
    real_max = real_vals.max()
    real_margin = 0.1 * (real_max - real_min)
    real_ylim = (real_min - real_margin, real_max + real_margin)
    
    # For the bottom subplot (Imaginary Value Over Time).
    imag_min = imag_vals.min()
    imag_max = imag_vals.max()
    imag_margin = 0.1 * (imag_max - imag_min)
    imag_ylim = (imag_min - imag_margin, imag_max + imag_margin)
    
    # x-axis limit (frame index) for the bottom two plots.
    xlim = (0, num_frames - 1)
    
    frames_list = []  # list to collect frames

    # --- Create Frames for the Animation ---
    current_phase = 0.0
    for i in range(num_frames):
        phase_step = initial_phase_step * (1 + i/num_frames)
        current_phase += phase_step
        vec = amplitudes[i] * np.exp(1j * phases[i])
        
        # Create figure with three vertically stacked subplots.
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        plt.subplots_adjust(hspace=0.4)

        # --- Top subplot: Complex Plane ---
        ax = axs[0]
        max_val = amplitude_base + abs(amplitude_mod) + 0.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Complex Plane")
        ax.set_aspect('equal')
        
        # Draw a reference circle at the base amplitude.
        circle = plt.Circle((0, 0), amplitude_base, color='gray', fill=False, linestyle='--')
        ax.add_artist(circle)
        # --- Draw solid thin gray axis lines ---
        ax.axhline(y=0, color='gray', linewidth=1)  # Horizontal axis (Real axis)
        ax.axvline(x=0, color='gray', linewidth=1)  # Vertical axis (Imaginary axis)

        # --- Draw the vector as an arrow using quiver for better visibility ---
        ax.quiver(0, 0, vec.real, vec.imag, angles='xy', scale_units='xy', scale=1, color='black', width=0.005, headwidth=6, headlength=8)

        # --- Added dots on the axes: ---
        ax.plot(vec.real, 0, 'ro')  # Red dot on the real axis.
        ax.plot(0, vec.imag, 'bo')  # Blue dot on the imaginary axis.

        # --- Add thin gray dotted lines for decomposition visualization ---
        ax.plot([vec.real, vec.real], [0, vec.imag], 'gray', linestyle='dotted', linewidth=1)  # Vertical line
        ax.plot([0, vec.real], [vec.imag, vec.imag], 'gray', linestyle='dotted', linewidth=1)  # Horizontal line

        # --- Middle subplot: Real Value over Time ---
        ax = axs[1]
        ax.plot(range(i+1), real_vals[:i+1], 'r-')
        # Add a red dot at the most recent real value.
        ax.plot(i, real_vals[i], 'ro')
        ax.set_xlim(xlim)
        ax.set_ylim(real_ylim)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Real Value")
        ax.set_title("Real Value of the Vector Over Time")
        ax.grid(True)

        # --- Bottom subplot: Imaginary Value over Time ---
        ax = axs[2]
        ax.plot(range(i+1), imag_vals[:i+1], 'b-')
        # Add a blue dot at the most recent imaginary value.
        ax.plot(i, imag_vals[i], 'bo')
        ax.set_xlim(xlim)
        ax.set_ylim(imag_ylim)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Imaginary Value")
        ax.set_title("Imaginary Value of the Vector Over Time")
        ax.grid(True)
        
        # Render the figure and extract the image.
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())
        image = image[..., :3]  # remove alpha channel
        frames_list.append(image)
        plt.close(fig)

    # --- Save as Animated GIF ---
    imageio.mimsave(filename, frames_list, fps=fps)
    print(f"Animation saved as {filename}")


# Example usage:
if __name__ == '__main__':
    create_complex_vector_animation("complex_vector_animation.gif",
                                    num_frames=120, #120
                                    initial_phase_step=0.1,
                                    amplitude_base=1.0,
                                    amplitude_mod=0.0,
                                    mod_rate=3,
                                    fps=6)

    create_complex_vector_and_freq_animation("complex_vector_animation_and_freq.gif",
                                    num_frames=120, # 120
                                    initial_phase_step=0.1,
                                    amplitude_base=1.0,
                                    amplitude_mod=0.5,
                                    mod_rate=3,
                                    fps=6)
