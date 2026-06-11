import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering


def _lagged_correlation_matrix(signals):
    """
    Computes a symmetric distance matrix based on max correlation (lag >= 0)
    using vectorized FFT broadcasting.
    """
    n_comps, n_samples = signals.shape
    # 1. Standardize signals along the time axis
    # This ensures the cross-correlation equals the correlation coefficient
    norm_signals = zscore(signals, axis=1)

    # 2. Pad for FFT to avoid circular convolution (length 2n - 1)
    pad_len = 2 * n_samples - 1
    # Use rfft for real-valued signals (faster and uses less memory)
    signals_fft = np.fft.rfft(norm_signals, n=pad_len, axis=1)

    max_corrs = np.zeros((n_comps, n_comps))

    for i in range(n_comps):
        # 3. Multiply spectrum of signal 'i' with ALL other signals
        # np.conj handles the 'correlation' vs 'convolution' logic
        combined_fft = signals_fft[i] * np.conj(signals_fft)

        # 4. Batch Inverse FFT to get all correlations for this row
        # This returns an (n_comps, pad_len) matrix
        corrs = np.fft.irfft(combined_fft, n=pad_len, axis=1)

        # 5. Extract lags >= 0 
        # In the 'full' correlation result, lag 0 is at index 0 
        # due to how np.fft.irfft aligns the output.
        # However, to match 'scipy.signal.correlate' logic:
        # Lags [0...n_samples-1] are at the start of the array.
        lags_ge_zero = corrs[:, :n_samples]

        # 6. Find max absolute correlation for each pair
        # We divide by n_samples to normalize the correlation to [-1, 1]
        max_corrs[i, :] = np.max(np.abs(lags_ge_zero), axis=1) / n_samples

    # 7. Make the matrix symmetric
    # Since we want distance(A,B) == distance(B,A), we take the 
    # maximum correlation found regardless of which signal was 'leading'.
    max_corrs = np.maximum(max_corrs, max_corrs.T)

    return 1 - max_corrs


def _cluster_components(ica_stats, n_components=5, n_clusters=3):
    """
    Subroutine for grouping top n_components into clusters and heuristics-based assignment.
    """
    df = pd.DataFrame(ica_stats).iloc[:n_components]
    features = {'mean_hr': 'mean', 'snr': 'max', 'sdnn': 'min', 'hr_outlier_rate': 'min', 'num_peaks': 'mean'}
    sort_features = {'sdnn': True, 'mean_hr': False, 'snr': False}
    if 'energy' in df.columns:
        features['energy'] = 'max'
        sort_features['energy'] = False

    # # Clustering 
    # X = df[['mean_hr']]
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # df['cluster'] = kmeans.fit_predict(X_scaled)

    ica_signals = np.array([e['signal'] for e in ica_stats[:n_components]])
    cluster_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='complete'
    )
    df['cluster'] = cluster_model.fit_predict(_lagged_correlation_matrix(ica_signals))

    # Assign heuristic
    df_cluster = df.groupby('cluster').agg(features).sort_values(list(sort_features.keys()), ascending=list(sort_features.values())).reset_index()
    fetal = df[df.cluster == df_cluster.loc[0].cluster]['index'].iloc[0]
    maternal = df[df.cluster == df_cluster.loc[1].cluster]['index'].iloc[0]


    df['assigned_label'] = df['cluster'].map({
        df_cluster.loc[0].cluster: 'Fetal',
        df_cluster.loc[1].cluster: 'Maternal'
    }).fillna('Other')

    # sanity check
    # maternal hr should be lower
    if df[df['index'] == maternal].mean_hr.mean() >= df[df['index'] == fetal].mean_hr.mean():
        print("Warning: Heuristic assignment may be incorrect based on mean HR. Please review the clustering results.")
        print(f"Maternal mean HR: {df[df['index'] == maternal].mean_hr.mean()}")
        print(f"Fetal mean HR: {df[df['index'] == fetal].mean_hr.mean()}")
    print(df_cluster)


    return maternal, fetal, df