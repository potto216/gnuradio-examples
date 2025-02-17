import numpy as np
import matplotlib.pyplot as plt
import imageio
from typing import Optional

def create_complex_vector_animation(
    filename: str,
    num_frames: int = 100,
    initial_phase_step: float = 0.1,
    frequency_ramp: float = 0,
    amplitude_base: float = 1.0,
    amplitude_mod: float = 0.5,
    f0: Optional[float] = None,
    mod_rate: float = 2,
    fps: int = 10,
    time_plot_position: str = 'below',
    plot_imag_time: bool = True,
    figure_number: int = 2,
    angle_symbol: Optional[str] = "\u03B8"  # New parameter: Unicode angle symbol (default: Greek theta)
) -> None:
    """
    Create an animated GIF showing a rotating, amplitude-modulated complex vector.
    
    The figure consists of:
      - A complex plane plot of the vector.
      - Time-domain plots of the vector's real and (optionally) imaginary components.
    
    Each plot's title is prefixed with the figure number and a letter indicating its order 
    (e.g. "2A:" for the top plot, "2B:" for the next, etc.). For the complex plane plot,
    if an angle symbol is provided (default is the Greek theta), the plot is annotated in the
    first quadrant with the angle (in degrees) between the vector and the x-axis. The plot title
    is also appended with this information.
    
    Parameters:
      filename (str): Name of the output GIF file.
      num_frames (int): Total number of frames in the animation.
      initial_phase_step (float): Base phase increment for rotation.
      frequency_ramp (float): Rate of change of the frequency over the full animation.
      amplitude_base (float): Base amplitude of the vector.
      amplitude_mod (float): Modulation depth added sinusoidally to the amplitude.
      f0 (Optional[float]): Frequency of the rotating vector in Hertz (must be positive if provided).
      mod_rate (float): Number of amplitude modulation cycles over the full animation.
      fps (int): Frames per second in the output GIF.
      time_plot_position (str): 'above' or 'below' determines where the time-domain plots are placed.
      plot_imag_time (bool): If False, omit the imaginary component time-domain plot.
      figure_number (int): Figure number to display in each plot title.
      angle_symbol (Optional[str]): Unicode character to use as the angle symbol (e.g. Greek theta). 
                                    If None, no angle annotation is shown.
    
    Raises:
      ValueError: If f0 is provided and is not positive, or if time_plot_position is invalid.
    """
    if f0 is not None and f0 <= 0:
        raise ValueError("f0 must be positive if provided.")
    if time_plot_position.lower() not in ('above', 'below'):
        raise ValueError("time_plot_position must be either 'above' or 'below'.")

    sample_rate = f0 if f0 is not None else 1
    if f0 is not None:
        print(f"Simulation sample rate: {sample_rate} Hz")
        time_label = "Time (s)"
    else:
        print(f"Simulation sample rate: {sample_rate} sample per unit time")
        time_label = "Samples"
    
    # Precompute data for all frames.
    phases = np.zeros(num_frames)
    amplitudes = np.zeros(num_frames)
    real_vals = np.zeros(num_frames)
    imag_vals = np.zeros(num_frames)
    
    current_phase = 0.0
    for i in range(num_frames):
        phase_step = initial_phase_step + frequency_ramp * i / num_frames
        current_phase += phase_step
        phases[i] = current_phase
        
        amplitude = amplitude_base + amplitude_mod * np.sin(2 * np.pi * mod_rate * i / num_frames)
        amplitudes[i] = amplitude
        
        vec = amplitude * np.exp(1j * current_phase)
        real_vals[i] = vec.real
        imag_vals[i] = vec.imag

    # Precompute fixed axis limits.
    real_min, real_max = real_vals.min(), real_vals.max()
    real_margin = 0.1 * (real_max - real_min) if real_max != real_min else 0.1
    real_ylim = (real_min - real_margin, real_max + real_margin)

    imag_min, imag_max = imag_vals.min(), imag_vals.max()
    imag_margin = 0.1 * (imag_max - imag_min) if imag_max != imag_min else 0.1
    imag_ylim = (imag_min - imag_margin, imag_max + imag_margin)

    if f0 is not None:
        xlim = (0, (num_frames - 1) / f0)
    else:
        xlim = (0, num_frames - 1)
    
    frames_list = []

    # Create frames.
    current_phase = 0.0
    for i in range(num_frames):
        phase_step = initial_phase_step + frequency_ramp * i / num_frames
        current_phase += phase_step
        vec = amplitudes[i] * np.exp(1j * phases[i])
        
        # Determine subplot layout.
        if time_plot_position.lower() == 'above':
            if plot_imag_time:
                fig, axs = plt.subplots(3, 1, figsize=(8, 10))
                ax_time_real, ax_time_imag, ax_complex = axs
                # Order (top-to-bottom): A: Real Time, B: Imag Time, C: Complex Plane.
                ax_time_real.set_title(f"$\\bf{{{figure_number}A:}}$ Real Component Over Time")
                ax_time_imag.set_title(f"$\\bf{{{figure_number}B:}}$ Imaginary Component Over Time")
                ax_complex.set_title(f"$\\bf{{{figure_number}C:}}$ Complex Plane: Rotating Vector")
            else:
                fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                ax_time_real, ax_complex = axs
                # Order: A: Real Time, B: Complex Plane.
                ax_time_real.set_title(f"$\\bf{{{figure_number}A:}}$ Real Component Over Time")
                ax_complex.set_title(f"$\\bf{{{figure_number}B:}}$ Complex Plane: Rotating Vector")
        else:  # 'below'
            if plot_imag_time:
                fig, axs = plt.subplots(3, 1, figsize=(8, 10))
                ax_complex, ax_time_real, ax_time_imag = axs
                # Order: A: Complex Plane, B: Real Time, C: Imag Time.
                ax_complex.set_title(f"$\\bf{{{figure_number}A:}}$ Complex Plane: Rotating Vector")
                ax_time_real.set_title(f"$\\bf{{{figure_number}B:}}$ Real Component Over Time")
                ax_time_imag.set_title(f"$\\bf{{{figure_number}C:}}$ Imaginary Component Over Time")
            else:
                fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                ax_complex, ax_time_real = axs
                # Order: A: Complex Plane, B: Real Time.
                ax_complex.set_title(f"$\\bf{{{figure_number}A:}}$ Complex Plane: Rotating Vector")
                ax_time_real.set_title(f"$\\bf{{{figure_number}B:}}$ Real Component Over Time")
        
        plt.subplots_adjust(hspace=0.4)

        # Complex Plane Plot.
        ax = ax_complex
        max_val = amplitude_base + abs(amplitude_mod) + 0.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_xlabel("Real (X coordinate)")
        ax.set_ylabel("Imaginary (Y coordinate)")
        ax.set_aspect('equal')
        
        circle = plt.Circle((0, 0), amplitude_base, color='gray', fill=False, linestyle='--')
        ax.add_artist(circle)
        ax.axhline(y=0, color='gray', linewidth=1)
        ax.axvline(x=0, color='gray', linewidth=1)
        ax.quiver(0, 0, vec.real, vec.imag, angles='xy', scale_units='xy', scale=1,
                  color='black', width=0.005, headwidth=6, headlength=8)
        ax.plot(vec.real, 0, 'ro')
        ax.plot(0, vec.imag, 'bo')
        ax.plot([vec.real, vec.real], [0, vec.imag], color='gray', linestyle='dotted', linewidth=1)
        ax.plot([0, vec.real], [vec.imag, vec.imag], color='gray', linestyle='dotted', linewidth=1)

        # Add angle annotation if requested.
        if angle_symbol is not None:
            # Compute the vector's polar angle in degrees (normalized to [0, 360)).
            angle = np.degrees(np.angle(vec))
            if angle < 0:
                angle += 360
            # Place the annotation in the first quadrant.
            ax.text(max_val * 0.5, max_val * 0.5,
                    f"{angle_symbol} = {angle:.1f}°",
                    fontsize=12, color='purple',
                    bbox=dict(facecolor='white', edgecolor='none'),
                    zorder=5)  # Add zorder parameter to place text on top of other elements.
        
        # Real Component Time-Domain Plot.
        t = np.arange(i + 1) / sample_rate if f0 is not None else np.arange(i + 1)
        ax = ax_time_real
        ax.plot(t, real_vals[:i+1], 'r-')
        ax.plot(t[-1], real_vals[i], 'ro')
        ax.set_xlim(xlim)
        ax.set_ylim(real_ylim)
        ax.set_xlabel(time_label)
        ax.set_ylabel("Real (X coordinate) Value")
        ax.grid(True)

        # Imaginary Component Time-Domain Plot (if enabled).
        if plot_imag_time:
            ax = ax_time_imag
            ax.plot(t, imag_vals[:i+1], 'b-')
            ax.plot(t[-1], imag_vals[i], 'bo')
            ax.set_xlim(xlim)
            ax.set_ylim(imag_ylim)
            ax.set_xlabel(time_label)
            ax.set_ylabel("Imaginary (Y coordinate) Value")
            ax.grid(True)
        
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())[...,:3]
        frames_list.append(image)
        plt.close(fig)
    
    imageio.mimsave(filename, frames_list, fps=fps)
    print(f"Animation saved as {filename}")


def create_real_signal_and_positive_negative_complex_vectors_animation(
    filename: str,
    num_frames: int = 100,
    initial_phase_step: float = 0.1,
    frequency_ramp: float = 0,
    amplitude_base: float = 1.0,
    amplitude_mod: float = 0.5,
    mod_rate: float = 2,
    f0: Optional[float] = None,
    fps: int = 10,
    time_plot_position: str = 'below',
    figure_number: int = 2  # New parameter for figure number
) -> None:
    """
    Create an animated GIF showing a rotating, amplitude-modulated complex vector along with its real signal.
    
    The figure includes:
      - A time-domain plot of the real component.
      - A complex plane plot of the vector.
    
    The plot titles are prefixed with the figure number and a letter (e.g. "2A:" for the top plot).
    
    Parameters:
      filename (str): Name of the output GIF file.
      num_frames (int): Total number of frames.
      initial_phase_step (float): Base phase increment for rotation.
      frequency_ramp (float): Rate of change of the frequency over the full animation.
      amplitude_base (float): Base amplitude of the vector.
      amplitude_mod (float): Modulation depth added sinusoidally.
      mod_rate (float): Number of amplitude modulation cycles.
      f0 (Optional[float]): Frequency of the rotating vector in Hertz.
      fps (int): Frames per second in the GIF.
      time_plot_position (str): 'above' or 'below' to determine the subplot order.
      figure_number (int): Figure number to display in plot titles.
    
    Raises:
      ValueError: If f0 is provided and is not positive, or if time_plot_position is invalid.
    """
    if f0 is not None and f0 <= 0:
        raise ValueError("f0 must be positive if provided.")
    if time_plot_position.lower() not in ('above', 'below'):
        raise ValueError("time_plot_position must be either 'above' or 'below'.")
    
    sample_rate = f0 if f0 is not None else 1
    if f0 is not None:
        print(f"Simulation sample rate: {sample_rate} Hz")
        time_label = "Time (s)"
    else:
        print(f"Simulation sample rate: {sample_rate} sample per unit time")
        time_label = "Samples"
    
    phases = np.zeros(num_frames)
    amplitudes = np.zeros(num_frames)
    real_vals = np.zeros(num_frames)
    inst_freq = np.zeros(num_frames)
    
    current_phase = 0.0
    for i in range(num_frames):
        phase_step = initial_phase_step + frequency_ramp * i / num_frames
        current_phase += phase_step
        phases[i] = current_phase
        
        amplitude = amplitude_base + amplitude_mod * np.sin(2 * np.pi * mod_rate * i / num_frames)
        amplitudes[i] = amplitude
        
        vec = amplitude * np.exp(1j * current_phase)
        real_vals[i] = vec.real
        
        inst_freq[i] = phase_step if i == 0 else phases[i] - phases[i-1]
    
    real_min, real_max = real_vals.min(), real_vals.max()
    real_margin = 0.1 * (real_max - real_min) if real_max != real_min else 0.1
    real_ylim = (real_min - real_margin, real_max + real_margin)
    
    if f0 is not None:
        xlim = (0, (num_frames - 1) / f0)
    else:
        xlim = (0, num_frames - 1)
    
    frames_list = []

    for i in range(num_frames):
        phase_step = initial_phase_step * (1 + i / num_frames)
        vec = amplitudes[i] * np.exp(1j * phases[i])
        
        if time_plot_position.lower() == 'above':
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            ax_time, ax_complex = axs
            # Top plot (A): Time-domain; Bottom plot (B): Complex Plane.
            ax_time.set_title(f"$\\bf{{{figure_number}A:}}$ Real Signal Over Time")
            ax_complex.set_title(f"$\\bf{{{figure_number}B:}}$ Complex Plane: Rotating Vector")
        else:
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            ax_complex, ax_time = axs
            # Top plot (A): Complex Plane; Bottom plot (B): Time-domain.
            ax_complex.set_title(f"$\\bf{{{figure_number}A:}}$ Complex Plane: Rotating Vector")
            ax_time.set_title(f"$\\bf{{{figure_number}B:}}$ Real Signal Over Time")
        
        plt.subplots_adjust(hspace=0.4)
        
        t = np.arange(i+1) / sample_rate if f0 is not None else np.arange(i+1)
        
        ax_time.plot(t, real_vals[:i+1], 'r-')
        ax_time.plot(t[-1], real_vals[i], 'ro')
        ax_time.set_xlim(xlim)
        ax_time.set_ylim(real_ylim)
        ax_time.set_xlabel(time_label)
        ax_time.set_ylabel("Real (X coordinate) Value")
        ax_time.grid(True)
        
        ax_complex.set_xlim(- (amplitude_base + abs(amplitude_mod) + 0.5),
                            amplitude_base + abs(amplitude_mod) + 0.5)
        ax_complex.set_ylim(- (amplitude_base + abs(amplitude_mod) + 0.5),
                            amplitude_base + abs(amplitude_mod) + 0.5)
        ax_complex.set_xlabel("Real (X coordinate)")
        ax_complex.set_ylabel("Imaginary (Y coordinate)")
        ax_complex.set_aspect('equal')
        circle = plt.Circle((0, 0), amplitude_base, color='gray', fill=False, linestyle='--')
        ax_complex.add_artist(circle)
        ax_complex.axhline(y=0, color='gray', linewidth=1)
        ax_complex.axvline(x=0, color='gray', linewidth=1)
        ax_complex.quiver(0, 0, vec.real*0.5, vec.imag*0.5, angles='xy', scale_units='xy', scale=1,
                          color='black', width=0.01, headwidth=4, headlength=5)
        ax_complex.quiver(vec.real*0.5, vec.imag*0.5, vec.real*0.5, -vec.imag*0.5,
                          angles='xy', scale_units='xy', scale=1, color='gray', linestyle='dotted',
                          width=0.005, headwidth=4, headlength=5)
        ax_complex.quiver(0, 0, vec.real*0.5, -vec.imag*0.5, angles='xy', scale_units='xy', scale=1,
                          color='gray', width=0.01, headwidth=4, headlength=5)
        ax_complex.quiver(vec.real*0.5, -vec.imag*0.5, vec.real*0.5, vec.imag*0.5,
                          angles='xy', scale_units='xy', scale=1, color='gray', linestyle='dotted',
                          width=0.005, headwidth=4, headlength=5)
        ax_complex.plot(vec.real, 0, 'ro')
        ax_complex.plot(0, vec.imag*0.5, 'bo')
        ax_complex.plot(0, -vec.imag*0.5, 'bd')
        ax_complex.plot([0, vec.real*0.5], [vec.imag*0.5, vec.imag*0.5], color='gray', linestyle='dotted', linewidth=1)
        ax_complex.plot([0, vec.real*0.5], [-vec.imag*0.5, -vec.imag*0.5], color='gray', linestyle='dotted', linewidth=1)
        
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())[...,:3]
        frames_list.append(image)
        plt.close(fig)
    
    imageio.mimsave(filename, frames_list, fps=fps)
    print(f"Animation saved as {filename}")


# Example usage:
if __name__ == '__main__':
    create_complex_vector_animation("images/fig1_complex_vector_animation.gif",
                                    num_frames=120,
                                    initial_phase_step=0.1,
                                    frequency_ramp=0,
                                    amplitude_base=1.0,
                                    amplitude_mod=0.0,
                                    mod_rate=0,
                                    f0=12000,
                                    fps=6,
                                    time_plot_position='below',
                                    plot_imag_time=True,
                                    figure_number=1)

    create_complex_vector_animation("images/fig3_complex_vector_freq_ramp_animation.gif",
                                    num_frames=120,
                                    initial_phase_step=0.1,
                                    frequency_ramp=0.2,
                                    amplitude_base=1.0,
                                    amplitude_mod=0.4,
                                    mod_rate=2,
                                    f0=12000,
                                    fps=6,
                                    time_plot_position='below',
                                    plot_imag_time=False,
                                    figure_number=3)    

    create_real_signal_and_positive_negative_complex_vectors_animation("images/fig4_real_signal_and_positive_negative_complex_vectors.gif", 
                                    num_frames=120,
                                    initial_phase_step=0.1,
                                    frequency_ramp=0.2,
                                    amplitude_base=1.0,
                                    amplitude_mod=0.4,
                                    mod_rate=2,
                                    f0=12000,
                                    fps=6,
                                    time_plot_position='below',
                                    figure_number=4)
