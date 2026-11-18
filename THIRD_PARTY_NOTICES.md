# Third-party and research data notices

Project MIT is issued and limited by MIT_SCOPE_APPROVED.json, including the
project-authored temporal segmentation adapter approved on 2026-11-18. No third-party
original, GAIC port, CG adapter, dataset or model weight is relicensed by it.
The following bundled project model parameters are for noncommercial research
and evaluation only. No commercial-use rights are granted, and these parameters
are not covered by project MIT:

- `local_assets/models/r66/decision.joblib` (R66 decision parameters)
- `local_assets/models/snapshot/root/autodl-fs/r60_20261022/fcdb_ranker.joblib` (FCDB ranker)
- `local_assets/models/snapshot/root/autodl-fs/r64_20261023/dino_projection.pt` (DINO projection)

The distribution also includes model configuration files.
The eleven externally hosted model binaries are acquired from their pinned
upstream sources for use under their terms. Retain the model and dataset
notices. No datasets, labels or media are bundled.

Third-party runtime files are documented in `third_party/SOURCE_PROVENANCE.json` and retain
their existing notices. Retain every original header and the texts in
third_party/licenses. SmartVidCrop is the separately bound RetargetVid file,
commit b16ab6bb92d4eb028406220ce14c49fcf03bf86c, MIT; upstream placeholder
copyright text is preserved as published, not replaced by hanfeng1324.

The UNISAL data.py in this candidate is a modified, function-only extraction of
get_optimal_out_size; its AST matches the phase-three pinned upstream function.
Source: rdroste/unisal@7f8ca1a7ecb3dbc1bf7fca837cb6186a85ac27d9/unisal/data.py,
Apache-2.0. Other UNISAL files keep their original headers, including MobileNet
and ConvGRU references. This does not license the excluded original full file.

GAIC r59_gaic.py is a local faithful PyTorch inference port, not an original
project-only implementation. Source: HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch
@344e86fb54b4b083d83c405aa63d7fbeeed4fcdc, roi_align/src/roi_align_kernel.cu,
rod_align/src/rod_align_kernel.cu, corresponding modules and croppingModel.py.
Upstream MIT Copyright (c) 2017 Max deGroot, Ellis Brown is retained in
GAIC_LICENSE.txt. Changes: CUDA forward geometry, endpoint sampling, extrapolation,
ROD exclusion and 2x2 pooling transcribed into tensor operations; MobileNet ranker
adapted locally. MobileNet retains tonylins Apache-2.0 and inherited notices.
The original CUDA binaries and model weights are absent. Local contribution
rights for preparation follow Phase 6; this file remains outside project MIT.

Both r22_lighthouse_runtime.py files are local adapters to Lighthouse/CG-DETR.
Sources: line/lighthouse@9b04316efe436d0cd4082e565e22b75f5f8bdb98:
lighthouse/frame_loaders/clip_loader.py and lighthouse/models.py; Apache-2.0,
Copyright $today.year LY Corporation (literal upstream notice preserved).
Inherited MIT: CG-DETR Copyright (c) 2023 WonJun Moon; QD-DETR Copyright (c) 2022
WonJun Moon; Moment-DETR Copyright (c) 2021 Jie Lei. Full texts are included.
Changes: PyAV FFmpeg graph, Hugging Face CLIP, training normalization retained,
zero padding for text, 75-cell chunking and output conversion; historical local
smoke drivers removed. Local contribution preparation follows Phase 6. These files
are outside the project MIT proposal. Checkpoints are separate assets.

Temporal segmentation uses the separately installed ruptures 1.1.10 package
by ENS Paris-Saclay and CNRS. BSD-2-Clause. See
[ruptures license](third_party/licenses/RUPTURES_BSD_2_CLAUSE.txt).
The candidate carries no LEAR source. The optional legacy comparison
loads user-supplied code from outside the distribution.
The DSNet video helper keeps its MIT attribution and is locally adapted to
defer segmentation to the selected backend. The feature extractor is unchanged.
SAM2 files use Apache-2.0; SAM2_CCTORCH_LICENSE.txt retains optional BSD inheritance,
without bundling its binary. Lighthouse UVCOM/CIM preserve EasonXiao-888 notices.

Research training data (no data is shipped):

- GNMC: Aneesh Vartakavi, The Gracenote Multi-Crop Dataset (0.0.1), 2022,
  https://doi.org/10.5281/zenodo.6228834 ;
  https://github.com/aneeshvartakavi/GNMC ; PolyForm Noncommercial 1.0.0,
  full terms in GNMC_LICENSE.txt. New works and their distribution are permitted
  for permitted purposes. This candidate adopts noncommercial research/evaluation;
  retain actual licensor Required Notice lines if supplied with the applicable
  copy (the illustrative example in the license is not a licensor notice).
- FCDB: Yi-Ling Chen et al., Quantitative Analysis of Automatic Image Cropping
  Algorithms: A Dataset and Comparative Study, WACV 2017,
  https://github.com/yiling-chen/flickr-cropping-dataset . Repository annotations
  and code carry Copyright (c) 2016, Yi-Ling Chen, MIT (FCDB_MIT_LICENSE.txt).
  Flickr images carry a research-purpose statement; training-result distribution
  for the actual image copy remains an open scope question. No image license
  is represented as the project's learned-parameter license.
- RetargetVid: Konstantinos Apostolidis and Vasileios Mezaris,
  A fast smart-cropping method and dataset for video retargeting, ICIP 2021,
  https://github.com/bmezaris/RetargetVid ; repository MIT, original text preserved.
- DHF1K: Wenguan Wang and collaborators, Revisiting Video Saliency:
  A Large-Scale Benchmark and a New Model, CVPR 2018; Revisiting Video Saliency
  Prediction in the Deep Learning Era, TPAMI 2019,
  https://github.com/wenguanwang/DHF1K . Dataset and code are offered under
  CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ .
  Training transformation: frozen visual features and human crop-IoU supervision
  were used to fit ranking/calibration and trajectory parameters. The parameters
  are new training outputs, not supplied media or gaze annotations. Retain this
  attribution, license link and change description with any permitted release;
  no author endorsement is implied. CC BY has no share-alike requirement and
  does not automatically apply as the license of all learned parameters.

Existing licenses cover only rights their licensors can grant. Do not repackage
datasets, cached features, original checkpoints or local research history here.

Phase 5: UNISAL weights_best.pth at upstream commit 7f8ca1a7ecb3dbc1bf7fca837cb6186a85ac27d9 is covered by the retained upstream Apache-2.0. Its exact frozen SHA256 was independently matched; no weight is included. Project MIT applies only to MIT_SCOPE_APPROVED.json.
