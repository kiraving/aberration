from aberration.model import PhaseNet, Data
from phasenet.zernike import ZernikeWavefront
from csbdeep.utils import normalize, download_and_extract_zip_file
import matplotlib.pyplot as plt
from pathlib import Path
from tifffile import imread
import numpy as np
from sklearn import metrics
from datetime import datetime

def single_mode(test_mode,model,amp_range=0.075,jitter='default',phantom='default'):
    amps = dict(zip([test_mode], [amp_range]))
    if jitter=='default':
        jitter = model.config.jitter
    if phantom=='default':
        phantom = model.config.phantom_params
    data = Data(
        batch_size           = 50,
        amplitude_ranges     = amps,
        order                = model.config.zernike_order,
        normed               = model.config.zernike_normed,
        psf_shape            = model.config.psf_shape,
        units                = model.config.psf_units,
        na_detection         = model.config.psf_na_detection,
        lam_detection        = model.config.psf_lam_detection,
        n                    = model.config.psf_n,
        noise_mean           = model.config.noise_mean,
        noise_snr            = model.config.noise_snr,
        noise_sigma          = model.config.noise_sigma,
        noise_perlin_flag    = model.config.noise_perlin_flag,
        crop_shape           = model.config.crop_shape,
        jitter               = jitter,
        #phantom_params       = phantom_params, #ph, #
        phantom_params       = phantom, #model.config.phantom_params,
        planes               = model.config.planes,
    )
    psfs, amps = next(data.generator())
    return psfs, amps

def gt_array(test_mode,zern,Y):
    num_samples=Y.shape[0]
    y_gt = np.zeros((num_samples,len(zern)))
    i=0
    for zer in zern:
        #zer = int(zer)
        #print(f"\n   zer {zer} vs zph {zph}, type(zer)= {type(zer)}, type(zph)= {type(zph)}")
        #y_pred[i] = prepare_plot_values(random_mode_result,exp_amps, zer)
        if str(zer)==str(test_mode):
            print("GT Mse mae")
            y_gt[:,i] = Y.flatten() # _?
        else:
            y_gt[:,i] = np.zeros(num_samples)
        i+=1
    return y_gt


def base_plot(dir_name, model, title, fig_name_end, phantom, jitter, amp_range=0.075, fontsize=18, fontsize_small=14):
    mses = []
    rmses = []
    maes = []
    mses_all = []
    rmses_all = []
    maes_all = []

    zern_ansi = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    zern = zern_ansi

    timestamp = datetime.now()
    timestr = timestamp.strftime("%d%m")
    # dir_name = basedir+name
    with open(dir_name + '/metrics.txt', 'a+') as file:
        file.write("\nDate: " + timestr)
        file.write("\n" + title)

        file.write("\nTesting range: " + str(amp_range))

        for test_mode in zern:
            # test_mode = 6

            X, Y = single_mode(test_mode, model, amp_range, phantom=phantom, jitter=jitter)

            fig = plt.figure(figsize=(10, 8))
            """for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(fontsize)"""

            ind = np.argsort(Y.flatten())  # .ravel())
            # print(ind.shape)
            amps_pred = np.array([model.predict(psf) for psf in X])
            print(Y.shape, amps_pred.shape)

            # plt.axhline(color = 'r',label='gt for other modes')
            plt.plot(Y.flatten()[ind], Y.flatten()[ind], marker='*', label=f'gt for mode {test_mode}')
            # plt.plot(Y.flatten()[ind],amps_pred.flatten()[ind], '--', marker='*', label='pred')

            i = np.where(np.array(zern_ansi) == test_mode)[0]
            print(i)
            plt.plot(Y.flatten()[ind], amps_pred[ind, i], '--', marker='o', label=f'predicted for mode {test_mode}')
            # plt.plot(exp_amps_, exp_amps_, marker='o', label='gt')
            plt.xlabel("Input PSF amplitude ($\mu m$)", fontsize=fontsize)
            plt.ylabel("Predicted amplitude ($\mu m$)",
                       fontsize=fontsize)  # tuple(model.config.zernike_amplitude_ranges.keys())[0]}')

            limit = amp_range + 0.01
            lower_limit, upper_limit = -limit, limit  # -0.1,0.1
            plt.ylim(lower_limit, upper_limit)
            plt.xlim(lower_limit, upper_limit)

            y_gt = gt_array(test_mode, zern_ansi, Y)

            mse = metrics.mean_squared_error(amps_pred[ind, i], y_gt[ind, i])
            mae = metrics.mean_absolute_error(amps_pred[ind, i], y_gt[ind, i])
            rmse = metrics.mean_squared_error(amps_pred[ind, i], y_gt[ind, i], squared=False)
            file.write("\nMSE [zern " + str(test_mode) + " alone] = " + str(mse))
            file.write("\nRMSE [zern " + str(test_mode) + " alone] = " + str(rmse))
            file.write("\nMAE [zern " + str(test_mode) + " alone] = " + str(mae) + "\n")
            mses.append(mse)
            rmses.append(rmse)
            maes.append(mae)

            plt.legend(title="MSE on target mode= " + str(np.round(mse, 5)) + "\nMAE = " + str(np.round(mae, 5)), loc=2,
                       prop={'size': fontsize_small}, title_fontsize=fontsize_small)
            plt.title(title,
                      fontsize=fontsize)  # , {model.config.zernike_order}, {model.config.train_epochs} epochs, {model.config.crop_shape[0]} crop')

            mse_all = metrics.mean_squared_error(amps_pred[ind], y_gt[ind])
            mae_all = metrics.mean_absolute_error(amps_pred[ind], y_gt[ind])
            rmse_all = metrics.mean_squared_error(amps_pred[ind], y_gt[ind], squared=False)
            file.write("\nMSE [zern " + str(test_mode) + " all modes] = " + str(mse_all))
            file.write("\nRMSE [zern " + str(test_mode) + " all modes] = " + str(rmse_all))
            file.write("\nMAE [zern " + str(test_mode) + " all modes] = " + str(mae_all) + "\n")
            mses_all.append(mse_all)
            rmses_all.append(rmse_all)
            maes_all.append(mae_all)
            plt.xticks(fontsize=fontsize_small)
            plt.yticks(fontsize=fontsize_small)

            ax2 = fig.add_axes([0.58, 0.2, 0.3, 0.2])
            ax2.boxplot(amps_pred[ind])
            plt.xticks(range(1, len(zern_ansi) + 1), zern_ansi, fontsize=fontsize_small)
            plt.yticks(fontsize=fontsize_small)
            # ax2.x
            plt.ylim(lower_limit, upper_limit)
            # ax2.xlim(lower_limit,upper_limit)
            ax2.set_ylabel("Predicted amplitude ($\mu m$)", size=fontsize_small, labelpad=-0.5)
            ax2.set_xlabel("Zernike mode", size=fontsize_small, labelpad=-0.5)
            plt.legend(title="MSE = " + str(np.round(mse_all, 5)) + "\nMAE = " + str(np.round(mae_all, 5))  # )
                       # ,prop={'size': fontsize_small})
                       , title_fontsize=fontsize_small - 2)
            plt.title("Predictions for all modes", fontsize=fontsize_small)

            plt.savefig(dir_name + f'/test_z{test_mode}_range{amp_range}{fig_name_end}.png')

        MSE = np.array(mses).mean()
        print(MSE)
        file.write("\nMean MSE target mode " + str(amp_range) + "range= " + str(MSE))
        RMSE = np.array(rmses).mean()
        print(RMSE)
        file.write("\nMean RMSE target mode " + str(amp_range) + " range= " + str(RMSE))
        MAE = np.array(maes).mean()
        print(MAE)
        file.write("\nMean MAE target mode " + str(amp_range) + " range  = " + str(MAE))
        mse = np.array(mses_all).mean()
        rmse = np.array(rmses_all).mean()
        mae = np.array(maes_all).mean()
        print("across all modes: ", mse, rmse, mae)
        file.write("\nMSE across all modes on " + str(amp_range) + " range = " + str(mse))
        file.write("\nRMSE across all modes on " + str(amp_range) + " range = " + str(rmse))
        file.write("\nMAE across all modes on " + str(amp_range) + " range = " + str(mae) + "\n")
        median = np.median(np.array(rmses_all))
        file.write("\nMedian RMSE = " + str(median) + "\n")
        print(median)
    return np.round(mse, 6)


def central_crop(dir_name, model, amp_range=0.075, fontsize=18, fontsize_small=14):
    title = 'Test on central crop of the first 3D image'
    fig_name_end = 'ph1center'
    phantom = {'name': 'images',
               'filepath': './images/ly17_crop50.tif',
               'shape': [64, 64, 64],
               'units': [0.2, 0.068519, 0.068519]}
    jitter = False
    mse = base_plot(dir_name, model, title, fig_name_end, phantom, jitter, amp_range
                    , fontsize=fontsize, fontsize_small=fontsize_small)
    return mse


def random_crop(dir_name, model, amp_range=0.075, fontsize=18, fontsize_small=14):
    title = 'Test on random crops of the first 3D image'
    fig_name_end = 'ph1jitter'
    phantom = {'name': 'images',
               'filepath': './images/ly17_crop50.tif',
               'shape': [64, 64, 64],
               'units': [0.2, 0.068519, 0.068519]}
    jitter = True
    mse = base_plot(dir_name, model, title, fig_name_end, phantom, jitter, amp_range
                    , fontsize=fontsize, fontsize_small=fontsize_small)
    return mse


def second_random_crop(dir_name, model, amp_range=0.075, fontsize=18, fontsize_small=14):
    title = 'Test on random crops of the second 3D image'
    fig_name_end = 'ph2jitter'
    phantom = {'name': 'images',
               'filepath': './images/ly22_crop50.tif',
               'shape': [64, 64, 64],
               'units': [0.2, 0.068519, 0.068519]}
    jitter = True
    mse = base_plot(dir_name, model, title, fig_name_end, phantom, jitter, amp_range
                    , fontsize=fontsize, fontsize_small=fontsize_small)
    return mse