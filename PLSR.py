import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_spectrum_data_from_multiple_files(spectrum_files):
    X_list = []
    y_list = []

    for spectrum_file in spectrum_files:


        data = np.loadtxt(spectrum_file)


        X = data[:-1]


        y = data[-1]

        X_list.append(X)
        y_list.append(y)


    X_all = np.array(X_list)
    y_all = np.array(y_list)
    return X_all, y_all



def load_wavelengths(wavelength_file):

    wavelengths = np.loadtxt(wavelength_file)
    return wavelengths


def plsrearly(X, y, num_repeats=100, num_folds=10, num_folds2=10, num_lv_range=range(1, 21)):
    num_samples = len(y)

    yestimation = np.zeros((num_samples, num_repeats))

    regresscoeff = np.zeros((X.shape[1], num_repeats))


    for repeat_idx in range(num_repeats):

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=repeat_idx)

        predictions_for_repeat = np.empty(num_samples)

        beta_list = []


        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]


            kf2 = KFold(n_splits=num_folds2, shuffle=True, random_state=repeat_idx)
            r2_fold = np.zeros((num_folds2, len(num_lv_range)))
            for m, (train_idx2, test_idx2) in enumerate(kf2.split(X_train)):
                X_train2, y_train2 = X_train[train_idx2], y_train[train_idx2]
                X_val2, y_val2 = X_train[test_idx2], y_train[test_idx2]

                for j, num_lvs in enumerate(num_lv_range):
                    pls = PLSRegression(n_components=num_lvs)
                    pls.fit(X_train2, y_train2)
                    y_pred2 = pls.predict(X_val2)
                    r2_fold[m, j] = r2_score(y_val2, y_pred2)


            rdcv_r2 = np.mean(r2_fold, axis=0)
            best_num_lvs = num_lv_range[np.argmax(rdcv_r2)]


            pls_best = PLSRegression(n_components=best_num_lvs)
            pls_best.fit(X_train, y_train)
            y_pred = pls_best.predict(X_test).flatten()


            predictions_for_repeat[test_idx] = y_pred


            beta_list.append(pls_best.coef_.reshape(-1))


            fold_r2 = r2_score(y_test, y_pred)
            print(f"Repeat {repeat_idx + 1}, Fold R²: {fold_r2:.4f}")


        yestimation[:, repeat_idx] = predictions_for_repeat

        beta_array = np.array(beta_list)
        regresscoeff[:, repeat_idx] = np.mean(beta_array, axis=0)


    mean_final_est = np.mean(yestimation, axis=1)
    mean_final_rc = np.mean(regresscoeff, axis=1)


    SSE = np.sum((y - mean_final_est) ** 2)
    SST = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - SSE / SST
    RMSE = np.sqrt(np.mean((y - mean_final_est) ** 2))

    print(f"Final R²: {R2:.4f}, RMSE: {RMSE:.4f}")

    return R2, mean_final_est, mean_final_rc, RMSE



def plot_prediction_results(y_true, y_pred, r2, rmse, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--',
             label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'MDA\nR² = {r2:.2f}, RMSE = {rmse:.2f}')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()



if __name__ == "__main__":
    # 主程序：设置数据文件夹和文件路径
    spectrum_directory = r""
    wavelength_file = r""


    spectrum_files = [os.path.join(spectrum_directory, f) for f in os.listdir(spectrum_directory) if f.endswith('.txt')]


    X, y = load_spectrum_data_from_multiple_files(spectrum_files)
    wavelengths = load_wavelengths(wavelength_file)


    R2, mean_final_est, mean_final_rc, RMSE = plsrearly(X, y)
    print("Overall R²:", R2, "RMSE:", RMSE)


    prediction_plot_path = r''
    plot_prediction_results(y, mean_final_est, R2, RMSE, prediction_plot_path)
