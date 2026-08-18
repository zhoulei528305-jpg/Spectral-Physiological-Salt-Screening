import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from tqdm import tqdm



def load_training_data(train_folder):
    train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.txt')]
    train_files.sort()
    X_list, y_list = [], []
    for file in train_files:
        data = np.loadtxt(file)
        X_list.append(data[:-1])
        y_list.append(data[-1])
    return np.array(X_list), np.array(y_list)



def load_prediction_samples(pred_folder):
    pred_files = [os.path.join(pred_folder, f) for f in os.listdir(pred_folder) if f.endswith('.txt')]
    pred_files.sort()
    sample_names = [os.path.splitext(f)[0] for f in os.listdir(pred_folder) if f.endswith('.txt')]
    X_pred = []
    for file in pred_files:
        data = np.loadtxt(file)
        X_pred.append(data)
    return np.array(X_pred), sample_names



def pls_with_rdcv_ensemble(X, y, X_pred, num_repeats=100, num_folds=10, num_folds2=10, num_lv_range=range(1, 31)):
    num_samples_pred = len(X_pred)

    total_models = num_repeats * num_folds
    y_all_preds = np.zeros((num_samples_pred, total_models))

    best_num_lvs_list = []
    model_counter = 0

    for repeat_idx in tqdm(range(num_repeats), desc="Processing rdCV Ensemble"):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=repeat_idx)

        for train_idx, test_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]


            kf2 = KFold(n_splits=num_folds2, shuffle=True, random_state=repeat_idx)
            r2_inner = np.zeros((num_folds2, len(num_lv_range)))

            for m, (train_idx2, test_idx2) in enumerate(kf2.split(X_train)):
                X_tr2, y_tr2 = X_train[train_idx2], y_train[train_idx2]
                X_va2, y_va2 = X_train[test_idx2], y_train[test_idx2]

                for j, num_lvs in enumerate(num_lv_range):
                    pls = PLSRegression(n_components=num_lvs)
                    if X_tr2.shape[0] <= num_lvs:
                        continue
                    pls.fit(X_tr2, y_tr2)
                    y_pred2 = pls.predict(X_va2)

                    r2_inner[m, j] = r2_score(y_va2, y_pred2) if len(y_va2) > 1 else 0


            mean_r2_inner = np.mean(r2_inner, axis=0)
            best_num_lvs = num_lv_range[np.argmax(mean_r2_inner)]
            best_num_lvs_list.append(best_num_lvs)


            pls_final = PLSRegression(n_components=best_num_lvs)
            pls_final.fit(X_train, y_train)


            y_all_preds[:, model_counter] = pls_final.predict(X_pred).flatten()
            model_counter += 1


    final_mean = np.mean(y_all_preds, axis=1)
    final_std = np.std(y_all_preds, axis=1)

    return final_mean, final_std, best_num_lvs_list



def save_predictions_to_excel(sample_names, predictions, std_devs, output_file):
    df = pd.DataFrame({
        'Sample_Name': sample_names,
        'Predicted_Mean': predictions,
        'Predicted_Std': std_devs,
        'CV_Percent': (std_devs / predictions) * 100
    })
    df.to_excel(output_file, index=False)
    print(f"Ensemble predictions saved to {output_file}")


if __name__ == "__main__":

    train_folder = r""
    pred_folder = r""
    output_excel = r""

    X_train, y_train = load_training_data(train_folder)
    X_pred, sample_names = load_prediction_samples(pred_folder)


    final_mean, final_std, lv_list = pls_with_rdcv_ensemble(X_train, y_train, X_pred)


    unique_lvs, counts = np.unique(lv_list, return_counts=True)
    print("\nOptimal LV Distribution across 1000 models:")
    for lv, count in zip(unique_lvs, counts):
        print(f"LV {lv}: {count} times")


    save_predictions_to_excel(sample_names, final_mean, final_std, output_excel)