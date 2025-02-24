import numpy as np
import matplotlib.pyplot as plt
import os
import string

def plot_phase_ambiguity(magnitudes, real_value, figure_number=1, show_plots=True, save_plots=False, 
                         legend_locs=None, same_axis_limits=True, unit_circle_radius=1,
                         negative_fourth_quadrant=False, label_mode='direction'):
    """
    Demonstrates phase ambiguity when inferring phase angle of a rotating vector from its real (x-coordinate) value.

    Parameters:
        magnitudes (list of float): List of magnitudes for the rotating vector.
        real_value (float): Positive real value on the x-axis for phase calculations (must be ≤ each magnitude).
        figure_number (int): Number to include in subplot titles.
        show_plots (bool): Whether to display the plots.
        save_plots (bool): Whether to save the plots as PNG and SVG.
        legend_locs (dict): Dictionary specifying legend locations per subplot.
        same_axis_limits (bool): Whether to enforce the same axis limits across each row.
        unit_circle_radius (float): Radius of the dotted reference circle in the top row plots.
        negative_fourth_quadrant (bool): If True, displays fourth quadrant angle as a negative value.
        label_mode (str): If set to 'direction', labels the vectors as "Counterclockwise (CCW)" (black) and "Clockwise (CW)" (gray).
                           Otherwise, labels them as "Vector 1" (black) and "Vector 2" (gray).
    """
    vec_ccw_color = 'black'
    vec_cw_color = 'gray'
    
    num_plots = len(magnitudes)
    fig, axs = plt.subplots(2, num_plots, figsize=(5 * num_plots, 10))

    if num_plots == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    theta_full_deg = np.linspace(0, 360, 360)
    theta_full_rad = np.deg2rad(theta_full_deg)

    real_value_limits = []
    phase_angle_limits = []
    top_x_limits = []
    top_y_limits = []
    
    # Letter sequence A, B, C... for subplot titles
    title_labels = string.ascii_uppercase[:2 * num_plots]
    
    for i, m in enumerate(magnitudes):
        ax_top = axs[0, i]
        ax_bottom = axs[1, i]
        
        # Compute title labels
        top_label = f"{figure_number}{title_labels[i]}: "
        bottom_label = f"{figure_number}{title_labels[num_plots + i]}: "

        if real_value > m:
            ax_top.text(0.5, 0.5, "real_value > magnitude", horizontalalignment='center',
                        verticalalignment='center', fontsize=12)
            ax_top.set_title(f"{top_label}m = {m}")
            ax_bottom.text(0.5, 0.5, "real_value > magnitude", horizontalalignment='center',
                           verticalalignment='center', fontsize=12)
            continue

        theta_rad = np.arccos(real_value / m)
        theta_deg = np.rad2deg(theta_rad)
        
        theta_fourth_deg_positive = 360 - theta_deg
        theta_fourth_deg = -theta_deg if negative_fourth_quadrant else theta_fourth_deg_positive

        y1, y2 = m * np.sin(theta_rad), -m * np.sin(theta_rad)
        arrow_width = 0.05
        # Calculate arrow head adjustments for CCW and CW vectors
        head_length = arrow_width * 5
        dx = real_value
        dy1 = y1
        dy2 = y2
        vector_length1 = np.sqrt(dx**2 + dy1**2)
        vector_length2 = np.sqrt(dx**2 + dy2**2)
        scale1 = (vector_length1 - head_length) / vector_length1
        scale2 = (vector_length2 - head_length) / vector_length2

        # Set labels based on label_mode
        if label_mode == 'direction':
            label_black = 'Counterclockwise (CCW)'
            label_gray = 'Clockwise (CW)'
        else:
            label_black = 'Vector 1 (V1)'
            label_gray = 'Vector 2 (V2)'
        
        ax_top.arrow(0, 0, real_value * scale1, y1 * scale1, 
                    head_width=arrow_width*4, head_length=head_length, 
                    width=arrow_width, fc=vec_ccw_color, ec=vec_ccw_color, 
                    label=label_black)
        ax_top.arrow(0, 0, real_value * scale2, y2 * scale2,  
                    head_width=arrow_width*4, head_length=head_length, 
                    width=arrow_width, fc=vec_cw_color, ec=vec_cw_color, 
                    label=label_gray)        

        unit_circle = plt.Circle((0, 0), unit_circle_radius, color='gray', fill=False, linestyle=':', alpha=0.5, linewidth=3)
        ax_top.add_artist(unit_circle)
        if label_mode == 'direction':
            ax_top.scatter([real_value], [0], color='red', marker='x', s=2*50, label='CCW Real Value')
            ax_top.scatter([real_value], [0], color='red', marker='o', s=2*50, facecolors='none', edgecolors='r', label='CW Real Value')
        else:
            ax_top.scatter([real_value], [0], color='red', marker='x', s=2*50, label='V1 Real Value')
            ax_top.scatter([real_value], [0], color='red', marker='o', s=2*50, facecolors='none', edgecolors='r', label='V2 Real Value')
        
        ax_top.axhline(0, color='black', linewidth=0.8)
        ax_top.axvline(0, color='black', linewidth=0.8)

        ax_top.set_aspect('equal', adjustable='datalim')
        ax_top.set_xlabel("Real (x-coordinate)")
        ax_top.set_ylabel("Imaginary (y-coordinate)")
        
        if label_mode == 'direction':
            ax_top.set_title(f"{top_label} CCW θ = {theta_deg:.2f}°, CW θ = {theta_fourth_deg:.2f}°")
        else:
            ax_top.set_title(f"{top_label} V1 θ = {theta_deg:.2f}°, V2 θ = {theta_fourth_deg:.2f}°")

        legend_loc = legend_locs.get(f'top_{i}', 'upper right') if legend_locs else 'upper right'
        ax_top.legend(loc=legend_loc)

        top_x_limits.append((-m * 1.1, m * 1.1))
        top_y_limits.append((-m * 1.1, m * 1.1))

        real_values_full = m * np.cos(theta_full_rad)
        ax_bottom.plot(theta_full_deg, real_values_full, color='red')
                
        marker1 = m * np.cos(np.deg2rad(theta_deg))
        marker2 = m * np.cos(np.deg2rad(theta_fourth_deg if not negative_fourth_quadrant else 360 + theta_fourth_deg))

        # Set bottom labels based on label_mode
        if label_mode == 'direction':
            bottom_label_black = f"CCW Vector θ = {theta_deg:.2f}°"
            bottom_label_gray = f"CW Vector θ = {theta_fourth_deg:.2f}°"
        else:
            bottom_label_black = f"Vector 1 θ = {theta_deg:.2f}°"
            bottom_label_gray = f"Vector 2 θ = {theta_fourth_deg:.2f}°"

        ax_bottom.plot(theta_deg, marker1, 'x', mfc='none', mec='red', ms=10, mew=2, 
                    label=bottom_label_black)
        ax_bottom.plot(theta_fourth_deg_positive, marker2, 'o', mfc='none', mec='red', ms=10, mew=2,
                    label=bottom_label_gray)
                        
        ax_bottom.axhline(0, color='black', linestyle=':', linewidth=0.8)
        ax_bottom.set_xlabel("Phase Angle (°)")
        ax_bottom.set_ylabel("Real (x-coordinate)")
        ax_bottom.set_title(f"{bottom_label}Real (x-coordinate) vs Phase Angle")

        legend_loc = legend_locs.get(f'bottom_{i}', 'upper right') if legend_locs else 'upper right'
        ax_bottom.legend(loc=legend_loc)

        real_value_limits.append((min(real_values_full), max(real_values_full)))
        phase_angle_limits.append((0, 360))

    if same_axis_limits:
        min_real, max_real = min(v[0] for v in real_value_limits), max(v[1] for v in real_value_limits)
        min_x_top, max_x_top = min(v[0] for v in top_x_limits), max(v[1] for v in top_x_limits)
        min_y_top, max_y_top = min(v[0] for v in top_y_limits), max(v[1] for v in top_y_limits)

        for ax in axs[1]:
            ax.set_ylim(min_real, max_real)
            ax.set_xlim(0, 360)

        for ax in axs[0]:
            ax.set_xlim(min_x_top, max_x_top)
            ax.set_ylim(min_y_top, max_y_top)

    plt.tight_layout()

    if save_plots:
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/fig2_phase_ambiguity_visualization.png", dpi=300)
        plt.savefig("images/fig2_phase_ambiguity_visualization.svg", format='svg')

    if show_plots:
        plt.show()

# Example Usage:
if __name__ == '__main__':
    magnitudes = [1, 2, 3]
    real_value = 0.8
    figure_number = 2  # Example figure number
    legend_positions = {
        'top_0': 'upper left', 'bottom_0': 'upper left',
        'top_1': 'upper left', 'bottom_1': 'upper left',
        'top_2': 'upper left', 'bottom_2': 'upper left'
    }
    plot_phase_ambiguity(magnitudes, real_value, figure_number, show_plots=True, save_plots=True, 
                         legend_locs=legend_positions, same_axis_limits=True, 
                         unit_circle_radius=1.0, negative_fourth_quadrant=False,
                         label_mode='numeric')  # or use label_mode='numeric'
