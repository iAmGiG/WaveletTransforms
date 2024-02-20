# %%
import numpy as np
import pywt
from sklearn.cluster import KMeans
# Generate random weight matrix
weight_mat = np.random.randn(512, 512)
# %%
# Apply DWT to matrix
coeffs = pywt.dwt2(weight_mat, 'haar')
ll, (lh, hl, hh) = coeffs

# Threshold small values to 0
threshold = 0.1
lh_thresh = lh * (abs(lh) >= threshold)
hl_thresh = hl * (abs(hl) >= threshold)
hh_thresh = hh * (abs(hh) >= threshold)

# Quantize remaining coefficients to centroids
# kmeans = KMeans(n_clusters=8)
# lh_q, hl_q, hh_q = kmeans.fit_predict(lh_thresh), kmeans.fit_predict(
#     hl_thresh), kmeans.fit_predict(hh_thresh)

# Cluster thresholded coefficients and replace with cluster centers
kmeans_lh = KMeans(n_clusters=8).fit(lh_thresh.reshape(-1, 1))
kmeans_hl = KMeans(n_clusters=8).fit(hl_thresh.reshape(-1, 1))
kmeans_hh = KMeans(n_clusters=8).fit(hh_thresh.reshape(-1, 1))

lh_clustered = kmeans_lh.cluster_centers_[
    kmeans_lh.labels_].reshape(lh_thresh.shape)
hl_clustered = kmeans_hl.cluster_centers_[
    kmeans_hl.labels_].reshape(hl_thresh.shape)
hh_clustered = kmeans_hh.cluster_centers_[
    kmeans_hh.labels_].reshape(hh_thresh.shape)

# Reconstruct matrix
weight_mat_compressed = pywt.idwt2(
    (ll, (lh_clustered, hl_clustered, hh_clustered)), 'haar')

# Quantize original matrix to 8 bits (optional)
weight_mat_quant = (weight_mat/np.max(weight_mat) * 255).astype(np.uint8)

print("Sparsity DWT:", np.mean(weight_mat_compressed == 0))
print("Max quant error:", np.max(np.abs(weight_mat_compressed - weight_mat_quant)))

# %%
