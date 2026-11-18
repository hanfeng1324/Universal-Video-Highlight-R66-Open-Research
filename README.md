# Universal-Video-Highlight · R66 Open Research

This is the R66 Open Research variant with an open temporal segmentation
backend (ruptures). It runs the R66 temporal proposal and spatial crop
pipeline and emits native-frame decisions, crop boxes and an H.264 MP4. See
[temporal segmentation](TEMPORAL_SEGMENTATION.md) for the input and output
contract and the scope of the comparison with the historical R66.

The historical competition score of 35.05 belongs to the original frozen R66
submission, not this variant. This variant has not been officially scored and
does not claim numerical equivalence to that result. The public Big Buck Bunny
check covers only one clip and one aspect ratio.

## Install and acquire model assets

Use Python 3.12 and an NVIDIA CUDA GPU. The reference installer creates a
virtual environment with PyTorch 2.9.0, CUDA 12.8 and the pinned runtime
dependencies. Activate that environment before the subsequent commands.

```sh
python tools/install_runtime.py
python tools/fetch_models.py --list
python tools/fetch_models.py --model qwen --model dinov2 --model sam2 --model clip --model clip_tokenizer --model dsnet --model googlenet --model cg_detr --model uvcom --model unisal --model gaic --accept-qwen-research-license
python tools/preflight.py --inference --environment
```

Review the Qwen research license and other model terms before acquisition.
The external model binaries come from the pinned upstream sources. The
fetcher checks their hashes and uses bounded HTTP Range reads for upstream
ZIP members. If a direct download is interrupted after writing a prefix, it
attempts to resume with strict HTTP Range checks. If the fetch still exits
with a preserved direct-file partial, rerun the same fetch command with
`--resume-partials`; the complete file must still match its pinned SHA-256.
Use `--models` for an existing authorized local
installation, with `config/models.example.json` as the configuration template.

## Run from a video

```sh
python infer.py --video PATH_TO_VIDEO.mp4 --ratio 9:16 --work-dir NEW_OUTPUT_DIR --temporal-backend open --low-memory
```

Use `--public` when the input is public demonstration material. Use a new
output directory per run. The low-memory path uses the same BF16 Qwen
checkpoint with CPU offload. Outputs include `predictions.jsonl`,
`TEMPORAL_SEGMENTS.json`, `CAPABILITIES.json`, `highlight.mp4` and its
decode verification receipt.

`legacy_external` is an optional local comparison mode. To select it, use
`--temporal-backend legacy_external --legacy-kts-dir PATH_TO_YOUR_KTS_DIRECTORY`
with the two original modules you obtained lawfully. Treat the directory as
trusted Python source. The default command above requires only the declared
runtime dependencies and model assets.

Project MIT coverage is limited to the authorized source listed in
[LICENSE_SCOPE.md](LICENSE_SCOPE.md). The bundled project model parameters are
for noncommercial research and evaluation only and are not licensed under MIT.
Third-party material retains its own terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the parameter inventory and notices.

