import numpy as np
import cv2
from scipy import ndimage as nd
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.optimize import curve_fit



def gaussian_with_constant(x, A, mean, sigma, C):
    return A * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + C


def calculate_npy_profile(image):
    """
    Process a .npy image file to denoise, slice, and fit Gaussian profiles for horizontal and vertical cuts.

    Parameters:
        filepath (str): Path to the .npy file.

    Returns:
        tuple: (horizontal FWHM, vertical FWHM)
    """

    img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Denoise using Non-Local Means
    sigma_est = np.mean(estimate_sigma(img_normalized, channel_axis=-1))
    img_nlm_denoised = denoise_nl_means(img_normalized, h=1.15 * sigma_est, fast_mode=True, patch_size=5,
                                        patch_distance=6)

    # Determine adaptive slicing bounds around the max intensity
    max_pos = np.unravel_index(np.argmax(img_nlm_denoised), img_nlm_denoised.shape)
    window_size = 2  # Window size around the maximum intensity

    # Define the slicing bounds
    slice_start_row = max(0, max_pos[0] - window_size)
    slice_end_row = min(img_nlm_denoised.shape[0], max_pos[0] + window_size)
    slice_start_col = max(0, max_pos[1] - window_size)
    slice_end_col = min(img_nlm_denoised.shape[1], max_pos[1] + window_size)

    # Slice images horizontally and vertically
    sliced_img_hor = img_nlm_denoised[slice_start_row:slice_end_row, :]
    sliced_img_vert = img_nlm_denoised[:, slice_start_col:slice_end_col]

    # Compute profiles and smooth for fitting
    profile_sliced_hor = np.sum(sliced_img_hor, axis=0)
    profile_sliced_vert = np.sum(sliced_img_vert, axis=1)
    smoothed_prof_hor = nd.gaussian_filter(profile_sliced_hor, sigma=2)
    smoothed_prof_vert = nd.gaussian_filter(profile_sliced_vert, sigma=2)

    return smoothed_prof_hor, smoothed_prof_vert

def plot_beam_profiles_npy(image, ax_hor, ax_vert):

    k = 1.26

    # Calculate profiles
    smoothed_prof_hor, smoothed_prof_vert = calculate_npy_profile(image)

    #scale
    x_hor = np.arange(len(smoothed_prof_hor))
    x_vert = np.arange(len(smoothed_prof_vert))


    # Fit Gaussian to horizontal profile
    initial_guess_hor = [np.max(smoothed_prof_hor), np.argmax(smoothed_prof_hor), 1.0, np.min(smoothed_prof_hor)]
    bounds_hor = (0, [np.inf, len(smoothed_prof_hor), np.inf, np.inf])
    params_hor, _ = curve_fit(gaussian_with_constant, x_hor, smoothed_prof_hor, p0=initial_guess_hor, bounds=bounds_hor)
    fwhm_hor = 2.355 * params_hor[2]  # FWHM calculation for horizontal

    # Fit Gaussian to vertical profile

    initial_guess_vert = [np.max(smoothed_prof_vert), np.argmax(smoothed_prof_vert), 1.0, np.min(smoothed_prof_vert)]
    bounds_vert = (0, [np.inf, len(smoothed_prof_vert), np.inf, np.inf])
    params_vert, _ = curve_fit(gaussian_with_constant, x_vert, smoothed_prof_vert, p0=initial_guess_vert,
                               bounds=bounds_vert)
    fwhm_vert = 2.355 * params_vert[2]  # FWHM calculation for vertical

    ax_hor.clear()
    ax_vert.clear()

    #plot the profiles
    ax_hor.plot(x_hor, smoothed_prof_hor, label='Горизонтальный профиль', color='blue')
    ax_hor.plot(x_hor, gaussian_with_constant(x_hor, *params_hor), 'r-', label=f'Аппроксимация(FWHM={fwhm_hor:.2f})')
    ax_hor.set_xlabel('пс')
    ax_hor.set_ylabel('Интенсивность')
    ax_hor.legend()

    ax_vert.plot(smoothed_prof_vert, x_vert, label='Вертикальный профиль', color='blue')
    ax_vert.plot(gaussian_with_constant(x_vert, *params_vert), x_vert, 'r-', label=f'Аппроксимация(FWHM={fwhm_vert:.2f})')
    ax_vert.set_ylabel('пс')
    ax_vert.set_xlabel('Интенсивность')
    ax_vert.legend()


    return params_hor, params_vert
