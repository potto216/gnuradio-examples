import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Add legend with custom handles
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def animate_complex_vectors(num_vectors=5, frames=200, interval=50, save_path='animation.gif', amplitudes=None, frequencies=None, t_max=2*np.pi):
    """
    Create an animated GIF demonstrating that a single rotating complex vector is equivalent to
    the sum of several fixed-frequency rotating vectors.

    Parameters:
      num_vectors: int
          The number of red rotating vectors (max 5 recommended).
      frames: int
          Number of frames in the animation.
      interval: int
          Delay between frames in milliseconds.
      save_path: str
          Path to save the output GIF.
      amplitudes: list or None
          A list of amplitudes (magnitudes) for the red vectors. If None, defaults to decaying values.
      frequencies: list or None
          A list of fixed angular velocities (in radians per unit time) for each red vector. 
          If None, defaults to increasing values starting at 1.
      t_max: float
          The maximum time value for the animation (default is 2π).
    """
    # Set default amplitudes and frequencies if not provided
    if amplitudes is None:
        amplitudes = [1/(i+1) for i in range(num_vectors)]
    if frequencies is None:
        frequencies = [i+1 for i in range(num_vectors)]
    
    # Create time values
    t_vals = np.linspace(0, t_max, frames)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    # Calculate an approximate limit for plotting based on the sum of amplitudes
    lim = sum(amplitudes) + 0.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Rotating Vectors Demonstration")
    
    # Create line objects for each red vector and one for the black total vector
    #red_lines = [ax.plot([], [], 'r-', lw=2)[0] for _ in range(num_vectors)]
    red_lines = [ax.plot([], [], '-', color='#62A0CA',lw=2)[0] for _ in range(num_vectors)]
    # Replace black_line with a quiver plot
    black_arrow = ax.quiver([0], [0], [0], [0], color='k', 
                          angles='xy', scale_units='xy', scale=1,
                          width=0.008, headwidth=6, headlength=8)  
    legend_elements = [
        Line2D([0], [0], color='#62A0CA', lw=2, label='Component Vectors'),
        Line2D([0], [0], color='k', lw=2, label='Sum Vector')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
  
    def init():
        for line in red_lines:
            line.set_data([], [])
        black_arrow.set_UVC([0], [0])
        return red_lines + [black_arrow]
    
    def animate(frame):
        t = t_vals[frame]
        positions = [0+0j]  # start at the origin
        # Compute endpoints for each red vector sequentially
        for amp, freq in zip(amplitudes, frequencies):
            new_pos = positions[-1] + amp * np.exp(1j * freq * t)
            positions.append(new_pos)
        # Update red vectors: each segment from positions[i] to positions[i+1]
        for i in range(num_vectors):
            start = positions[i]
            end = positions[i+1]
            red_lines[i].set_data([start.real, end.real], [start.imag, end.imag])
        # Update black vector with arrow
        final_pos = positions[-1]
        black_arrow.set_UVC([final_pos.real], [final_pos.imag])
        return red_lines + [black_arrow]
        
    # Create the animation using FuncAnimation and save as GIF
    ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init,
                                  blit=True, interval=interval)
    ani.save(save_path, writer='pillow')
    plt.close(fig)

# Example usage:
if __name__ == '__main__':
    animate_complex_vectors(num_vectors=4, frames=300, interval=40, save_path='images/compare_complex_vectors_summed.gif')
