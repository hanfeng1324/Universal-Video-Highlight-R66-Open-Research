# Modified 2026: only get_optimal_out_size retained from rdroste/unisal.
# Upstream 7f8ca1a7ecb3dbc1bf7fca837cb6186a85ac27d9; Apache-2.0.
# Original full data.py is excluded; upstream copyright/notices in THIRD_PARTY_NOTICES.md.
import numpy as np

def get_optimal_out_size(img_size):
	ar = img_size[0] / img_size[1]
	min_prod = 100
	max_prod = 120
	ar_array = []
	size_array = []
	for n1 in range(7, 14):
		for n2 in range(7, 14):
			if min_prod <= n1 * n2 <= max_prod:
				this_ar = n1 / n2
				this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
				ar_array.append(this_ar_ratio)
				size_array.append((n1, n2))

	max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
	bn_size = size_array[max_ar_ratio_idx]
	out_size = tuple(r * 32 for r in bn_size)
	return out_size

