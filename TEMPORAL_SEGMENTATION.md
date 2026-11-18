# Temporal segmentation

The historical DSNet video preprocessor formed a linear Gram matrix
(XXᵀ) from L2-normalized GoogLeNet descriptors. It called the original
automatic selection with up to (N−1) changes for (N) descriptors,
`vmax=1`, descriptor rate 1, and a one-sample minimum segment. The original
count criterion combines kernel scatter with a nonlinear count penalty.
The historical R66 inference path imported only the feature extractor from that
preprocessor module. Its highlighter used the proposals from DSNet, CG-DETR
and UVCOM and did not call the segmentation method.

- `open` uses ruptures 1.1.10 `KernelCPD(kernel="linear", min_size=1)`
  directly on descriptors. The kernel corresponds to the same dot product.
  PELT chooses the count with an independent per-cut penalty of
  `0.5 * (1 + log(N))` in summed within-segment squared-error units. The
  value is fixed by sample count, with no video-specific fit. A
  single sample yields one segment directly.
- `legacy_external` forms the Gram matrix using the original feature
  dtype and calls the original selection via the user-supplied directory.
  The loader reads and executes `cpd_nonlin.py` and `cpd_auto.py` from
  that directory and records their hashes in the local receipt.

Both modes return interior cut indices with the cut sample beginning the next
segment. The adapter removes the terminal endpoint from the ruptures result,
then maps cuts through the actual sample-frame indices. Ranges are inclusive
and partition every decoded native frame, including an unsampled tail.

The full CLI executes the selected backend on the DSNet feature stream and
writes one call record per input in `TEMPORAL_SEGMENTS.json`. These are
diagnostic scene partitions. R66's proposal selection and crop scoring retain
their existing inputs and logic. The requested segmentation backend is
therefore observable even when the selected frames and boxes agree.

Seven synthetic descriptor cases with 1 to 384 samples were compared.
Six produced identical cut positions. An alternating sequence produced
no cuts with the fixed PELT penalty and 63 cuts with the original
criterion. A shared lack of cuts in the two-sample excursion case does
not establish scene-level recall. This is an independent research variant.


## Public BBB end-to-end check

A fresh run used the existing Big Buck Bunny 30–42 s segment (Blender
Foundation, CC BY 3.0), 9:16, on Windows with Python 3.12 and a local 8 GB
RTX 5060 Laptop GPU with CPU offload. The full pipeline executed all seven
capabilities and passed the independent output contract. From 288 input
frames it selected 287 and decoded all 287 from the rendered 360×640,
24-fps silent H.264 MP4. Retained frame indices, crop boxes and MP4 SHA256
matched the earlier run for the same input and aspect ratio.

The open backend processed the 20 freshly sampled descriptors in one
recorded call. Both backends returned a single native frame range [0, 287]
when the external legacy backend was separately run on those same
descriptors. This test covers one public input and one aspect ratio.

The eleven external binaries were previously fetched from pinned upstream
locations and reused by local path for this validation. They are excluded
from this candidate's inventory, which includes the three project research
parameter files.
