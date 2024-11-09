import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

def gaussian_with_constant(x, A, mean, sigma, C):
    """Gaussian function with a constant offset."""
    return A * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + C

def calculate_profiles(image):
    """Calculate horizontal and vertical intensity profiles from the image."""
    k = 1.26  # Scaling factor

    # Adaptive Slicing based on High-Intensity Region around Maximum Intensity
    max_pos = np.unravel_index(np.argmax(image), image.shape)
    window_size = 10 # Adjust the window size if needed

    # Calculate slicing bounds, ensuring they stay within the image dimensions
    slice_start_row = max(0, max_pos[0] - window_size)
    slice_end_row = min(image.shape[0], max_pos[0] + window_size)
    slice_start_col = max(0, max_pos[1] - window_size)
    slice_end_col = min(image.shape[1], max_pos[1] + window_size)

    # Extract adaptively sliced images
    sliced_img_hor = image[slice_start_row:slice_end_row, :]
    sliced_img_vert = image[:, slice_start_col:slice_end_col]

    # Compute profiles by summing intensities
    profile_sliced_hor = np.sum(sliced_img_hor, axis=0)
    profile_sliced_vert = np.sum(sliced_img_vert, axis=1)

    # Smooth profiles for Gaussian fitting
    smoothed_prof_hor = gaussian_filter(profile_sliced_hor, sigma=3)
    smoothed_prof_vert = gaussian_filter(profile_sliced_vert, sigma=3)

    return smoothed_prof_hor, smoothed_prof_vert

def plot_beam_profiles(image, ax_hor, ax_vert):
    """Plot horizontal and vertical beam profiles with Gaussian fits."""
    k = 1.26  # Scaling factor

    # Calculate profiles
    smoothed_prof_hor, smoothed_prof_vert = calculate_profiles(image)

    # Define fitting function
    def fit_gaussian(profile):
        x = np.arange(len(profile))
        initial_guess = [np.max(profile), np.argmax(profile), 1.0, np.min(profile)]
        try:
            popt, pcov = curve_fit(gaussian_with_constant, x, profile, p0=initial_guess)
            errors = np.sqrt(np.diag(pcov))
            fwhm = 2.355 * popt[2]
            fwhm_error = 2.355 * errors[2]
            return popt, fwhm, fwhm_error
        except RuntimeError:
            print("Fit failed; returning zeros.")
            return [0, 0, 0, 0], 0, 0

    # Fit both profiles
    params_hor, fwhm_hor, fwhm_error_hor = fit_gaussian(smoothed_prof_hor)
    params_vert, fwhm_vert, fwhm_error_vert = fit_gaussian(smoothed_prof_vert)

    # Plot horizontal profile
    x_hor = np.arange(len(smoothed_prof_hor )) * k
    ax_hor.clear()
    ax_hor.plot(x_hor, smoothed_prof_hor, label='Горизонтальный профиль', color='blue')
    ax_hor.plot(x_hor, gaussian_with_constant(np.arange(len(smoothed_prof_hor )), *params_hor), 'r-',
                label=f'Аппроксимация (FWHM={fwhm_hor:.2f} ± {fwhm_error_hor:.2f})')
    ax_hor.set_xlabel('пс')
    ax_hor.set_ylabel('Интенсивность')
    ax_hor.legend()

    # Plot vertical profile
    x_vert = np.arange(len(smoothed_prof_vert)) * k
    ax_vert.clear()
    ax_vert.plot(smoothed_prof_vert, x_vert, label='Вертикальный профиль', color='blue')
    ax_vert.plot(gaussian_with_constant(np.arange(len(smoothed_prof_vert)), *params_vert), x_vert, 'r-',
                 label=f'Аппроксимация(FWHM={fwhm_vert:.2f} ± {fwhm_error_vert:.2f})')

    ax_vert.set_xlabel('Интенсивность')
    ax_vert.set_ylabel('пс')
    ax_vert.legend()

    return params_hor, params_vert

