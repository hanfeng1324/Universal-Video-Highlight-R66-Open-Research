"""Segment sampled visual descriptors and map starts to inclusive frame ranges."""

from importlib import import_module
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
import hashlib


def add_backend_arguments(parser):
    parser.add_argument('--temporal-backend', choices=('open', 'legacy_external'),
                        default='open')
    parser.add_argument('--legacy-kts-dir', type=Path,
                        help='Directory of your legally obtained external KTS modules')


def backend_options(args, parser):
    if (args.temporal_backend == 'legacy_external') != (args.legacy_kts_dir is not None):
        parser.error('--legacy-kts-dir is required only for legacy_external')
    if args.legacy_kts_dir is not None:
        args.legacy_kts_dir = args.legacy_kts_dir.resolve()


def load_legacy(directory):
    """Evaluate only the explicitly selected source files, without writing bytecode."""
    modules = {name: ModuleType(name) for name in ('kts', 'kts.cpd_nonlin', 'kts.cpd_auto')}
    if any(name == 'kts' or name.startswith('kts.') for name in sys.modules):
        raise ValueError('A kts module is already loaded; use a fresh process')
    modules['kts'].__path__ = []
    hashes = {}
    try:
        sys.modules.update(modules)
        for name in ('cpd_nonlin', 'cpd_auto'):
            path = directory / (name + '.py')
            data = path.read_bytes()
            hashes[path.name] = hashlib.sha256(data).hexdigest()
            module = modules['kts.' + name]
            module.__file__ = str(path)
            module.__package__ = 'kts'
            exec(compile(data, str(path), 'exec'), module.__dict__)
        return modules['kts.cpd_auto'].cpd_auto, hashes
    finally:
        for name in modules:
            sys.modules.pop(name, None)


class TemporalSegmenter:
    def __init__(self, backend='open', legacy_dir=None):
        self.backend = backend
        if backend == 'open' and legacy_dir is None:
            ruptures = import_module('ruptures')
            self.kernel_cpd = ruptures.KernelCPD
            self.implementation = dict(package='ruptures', version=ruptures.__version__.strip('v'),
                                       kernel='linear', min_size=1, selection='PELT')
        elif backend == 'legacy_external' and legacy_dir is not None:
            self.cpd_auto, hashes = load_legacy(Path(legacy_dir))
            self.implementation = dict(source_sha256=hashes, vmax=1, desc_rate=1,
                                       selection='original cpd_auto')
        else:
            raise ValueError('Select open or provide an explicit directory for legacy_external')

    def segment(self, features, picks, frame_count):
        """Return cut indices, inclusive frame ranges, lengths and execution metadata."""
        import numpy as np

        started = perf_counter()
        features = np.asarray(features)
        picks = np.asarray(picks)
        if (features.ndim != 2 or min(features.shape) == 0
                or not np.issubdtype(features.dtype, np.floating)
                or not np.isfinite(features).all()):
            raise ValueError('Expected nonempty finite floating-point descriptor matrix')
        samples = len(features)
        if (picks.shape != (samples,) or not np.issubdtype(picks.dtype, np.integer)
                or picks[0] != 0 or picks[-1] >= frame_count or not np.all(picks[1:] > picks[:-1])):
            raise ValueError('Expected strictly increasing sample frames within the video, starting at 0')

        if self.backend == 'legacy_external':
            # The legacy function receives X X^T in the original descriptor dtype.
            cuts, _ = self.cpd_auto(features @ features.T, samples - 1, 1, verbose=False)
            settings = dict(max_cuts=samples - 1)
        else:
            # Penalty in summed within-segment squared-error units. For unit
            # descriptors this gives a fixed, reproducible log-length criterion.
            penalty = 0.5 * (1 + np.log(samples))
            if samples == 1:
                cuts = []
            else:
                ends = self.kernel_cpd(kernel='linear', min_size=1).fit(features).predict(pen=penalty)
                if not ends or ends[-1] != samples:
                    raise RuntimeError('KernelCPD did not return the terminal sample endpoint')
                cuts = ends[:-1]
            settings = dict(penalty=float(penalty))

        cuts = np.asarray(cuts)
        if (cuts.ndim != 1 or cuts.size and (
                not np.issubdtype(cuts.dtype, np.integer) or cuts[0] < 1 or cuts[-1] >= samples
                or not np.all(cuts[1:] > cuts[:-1]))):
            raise RuntimeError('Temporal backend returned invalid internal cut indices')
        cuts = cuts.astype(np.int64)
        edges = np.r_[0, picks[cuts], frame_count]
        segments = np.column_stack((edges[:-1], edges[1:] - 1))
        return dict(backend=self.backend, implementation=self.implementation, settings=settings,
                    sample_count=samples, frame_count=int(frame_count), cut_samples=cuts.tolist(),
                    frame_ranges_inclusive=segments.tolist(), frame_lengths=np.diff(edges).tolist(),
                    seconds=perf_counter() - started)

