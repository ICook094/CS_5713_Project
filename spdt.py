from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from histogram import Histogram

# ------------------------------------------------------------
# Impurity helpers
# ------------------------------------------------------------
def gini_from_counts(counts: np.ndarray) -> float:
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts / n
    return 1.0 - np.sum(p * p)

def entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = np.clip(counts / n, eps, 1.0)
    return -np.sum(p * np.log(p))

# ------------------------------------------------------------
# Tree node
# ------------------------------------------------------------
@dataclass
class _Node:
    id: int
    depth: int
    is_leaf: bool = True
    prediction: Optional[int] = None
    split_feature: Optional[int] = None
    split_threshold: Optional[float] = None
    left: Optional[int] = None
    right: Optional[int] = None

# ------------------------------------------------------------
# SPDT classifier (single-process)d
# ------------------------------------------------------------
class SPDT:
    
    def __init__(
        self,
        B: int,
        # W: int,
        max_depth: int = 100,
        min_samples_leaf: int = 10,
        impurity: str = "gini",  # "gini" or "entropy"
        candidates_per_feature: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.B = int(B)
        # self.W = int(W)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.candidates_per_feature = candidates_per_feature
        self._rng = np.random.default_rng(random_state)

        if impurity not in ("gini", "entropy"):
            raise ValueError("impurity must be 'gini' or 'entropy'")
        self._impurity_fn = gini_from_counts if impurity == "gini" else entropy_from_counts

        self._nodes: Dict[int, _Node] = {}
        self._root_id: int = 0
        self._n_classes: int = 0
        self._n_features: int = 0

    # ---------------- fit ----------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        n, d = X.shape
        self._n_features = d
        self._n_classes = int(np.max(y)) + 1

        # init root
        self._nodes = {0: _Node(id=0, depth=0, is_leaf=True, prediction=self._majority(y, self._n_classes))}
        active_leaves = {0}

        # BFS growth: one full data pass per depth
        for depth in range(self.max_depth):
            if not active_leaves:
                break

            # histograms (leaf, feat, cls) -> Histogram
            H: Dict[Tuple[int, int, int], Histogram] = {}
            # per-leaf class totals (for parent impurity)
            leaf_cls_counts: Dict[int, np.ndarray] = {lid: np.zeros(self._n_classes, dtype=float)
                                                      for lid in active_leaves}

            # ---- single pass over data: route each sample to current active leaf and update ----
            for xi, yi in zip(X, y):
                lid = self._route_to_leaf(self._root_id, xi)
                if lid not in active_leaves:
                    continue
                leaf_cls_counts[lid][yi] += 1.0
                for f in range(d):
                    key = (lid, f, yi)
                    if key not in H:
                        H[key] = Histogram(self.B)
                    H[key].update(float(xi[f]))

            # ---- choose split per active leaf ----
            splits: Dict[int, Tuple[Optional[int], Optional[float], float]] = {}
            for lid in list(active_leaves):
                parent_counts = leaf_cls_counts[lid]
                n_node = parent_counts.sum()

                # stopping: small node or pure
                if n_node < 2 * self.min_samples_leaf or parent_counts.max() == n_node:
                    splits[lid] = (None, None, 0.0)
                    continue

                parent_imp = self._impurity_fn(parent_counts)
                best_gain, best_feat, best_thr = 0.0, None, None

                for f in range(d):
                    # merged histogram over classes (for candidate thresholds)
                    merged = None
                    per_class = {}
                    totals = np.zeros(self._n_classes, dtype=float)
                    for c in range(self._n_classes):
                        key = (lid, f, c)
                        if key in H:
                            per_class[c] = H[key]
                            totals[c] = sum(m for _, m in H[key].bins)
                            if merged is None:
                                merged = Histogram(self.B)
                                merged.merge(H[key])
                            else:
                                merged.merge(H[key])

                    if merged is None:
                        continue

                    thresholds = merged.uniform()
                    if not thresholds:
                        continue
                    if self.candidates_per_feature is not None and len(thresholds) > self.candidates_per_feature:
                        idx = np.linspace(0, len(thresholds) - 1, self.candidates_per_feature, dtype=int)
                        thresholds = [thresholds[i] for i in idx]

                    total_sum = totals.sum()
                    if total_sum <= 0:
                        continue

                    for thr in thresholds:
                        left_counts = np.array([
                            (0.0 if c not in per_class else per_class[c].sum(thr))
                            for c in range(self._n_classes)
                        ], dtype=float)
                        nl = left_counts.sum()
                        nr = total_sum - nl
                        if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                            continue

                        right_counts = totals - left_counts
                        tau = nl / (nl + nr)
                        gain = parent_imp - (tau * self._impurity_fn(left_counts)
                                             + (1.0 - tau) * self._impurity_fn(right_counts))
                        if gain > best_gain:
                            best_gain, best_feat, best_thr = gain, f, float(thr)

                splits[lid] = (best_feat, best_thr, best_gain)

            # ---- materialize splits ----
            new_active = set()
            next_id = max(self._nodes) + 1 if self._nodes else 0

            for lid, (feat, thr, gain) in splits.items():
                node = self._nodes[lid]
                if feat is None or thr is None or gain <= 0.0:
                    node.is_leaf = True
                    # keep prediction as majority at creation time (or recompute if desired)
                    continue

                node.is_leaf = False
                node.split_feature = feat
                node.split_threshold = thr

                left_id, right_id = next_id, next_id + 1
                next_id += 2

                maj = node.prediction if node.prediction is not None else 0
                self._nodes[left_id]  = _Node(id=left_id,  depth=node.depth + 1, is_leaf=True, prediction=maj)
                self._nodes[right_id] = _Node(id=right_id, depth=node.depth + 1, is_leaf=True, prediction=maj)
                node.left, node.right = left_id, right_id

                if node.depth + 1 < self.max_depth:
                    new_active.add(left_id)
                    new_active.add(right_id)

            if not new_active:
                break
            active_leaves = new_active

            # (Optional) refresh leaf majorities with a routing pass—cheap and improves predictions
            leaf_counts = {lid: np.zeros(self._n_classes, dtype=int) for lid in active_leaves}
            for xi, yi in zip(X, y):
                lid = self._route_to_leaf(self._root_id, xi)
                if lid in leaf_counts:
                    leaf_counts[lid][yi] += 1
            for lid, cnt in leaf_counts.items():
                if cnt.sum() > 0:
                    self._nodes[lid].prediction = int(np.argmax(cnt))

        return self

    # ---------------- predict ----------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        out = np.empty(X.shape[0], dtype=int)
        for i, xi in enumerate(X):
            lid = self._route_to_leaf(self._root_id, xi)
            pred = self._nodes[lid].prediction
            out[i] = 0 if pred is None else int(pred)
        return out

    # ---------------- helpers ----------------
    def _route_to_leaf(self, node_id: int, x: np.ndarray) -> int:
        node = self._nodes[node_id]
        while not node.is_leaf:
            f = node.split_feature
            t = node.split_threshold
            node_id = node.left if x[f] < t else node.right
            node = self._nodes[node_id]
        return node.id

    @staticmethod
    def _majority(y: np.ndarray, n_classes: int) -> int:
        return int(np.argmax(np.bincount(y, minlength=n_classes)))


from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ------------ helpers to serialize the tree snapshot for workers ------------
def _snapshot_tree(nodes: Dict[int, Any]) -> Dict[int, Tuple[bool, Optional[int], Optional[float], Optional[int], Optional[int]]]:
    """
    Return a compact, picklable snapshot:
    id -> (is_leaf, split_feature, split_threshold, left, right)
    """
    snap = {}
    for nid, n in nodes.items():
        snap[nid] = (n.is_leaf, n.split_feature, n.split_threshold, n.left, n.right)
    return snap

def _route_to_leaf_snapshot(root_id: int, tree_snap: Dict[int, Tuple[bool, Optional[int], Optional[float], Optional[int], Optional[int]]], x: np.ndarray) -> int:
    nid = root_id
    is_leaf, f, t, l, r = tree_snap[nid]
    while not is_leaf:
        nid = l if x[f] < t else r
        is_leaf, f, t, l, r = tree_snap[nid]
    return nid

# ------------ worker function ------------
def _build_histograms_worker(
    X: np.ndarray,
    y: np.ndarray,
    active_leaves: List[int],
    tree_snapshot: Dict[int, Tuple[bool, Optional[int], Optional[float], Optional[int], Optional[int]]],
    root_id: int,
    B: int,
    n_classes: int,
    n_features: int,
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int,int,int], List[Tuple[float, float]]]]:
    """
    Return:
      leaf_class_counts: leaf_id -> counts[C]
      hist_bins: (leaf, feat, cls) -> list of (p, m) tuples (raw bins)
    """
    active_set = set(active_leaves)
    leaf_class_counts = {lid: np.zeros(n_classes, dtype=float) for lid in active_set}
    histos: Dict[Tuple[int,int,int], Histogram] = {}

    for xi, yi in zip(X, y):
        lid = _route_to_leaf_snapshot(root_id, tree_snapshot, xi)
        if lid not in active_set:
            continue
        leaf_class_counts[lid][int(yi)] += 1.0
        for f in range(n_features):
            key = (lid, f, int(yi))
            h = histos.get(key)
            if h is None:
                h = Histogram(B)
                histos[key] = h
            h.update(float(xi[f]))

    # convert to raw bins for pickling efficiency
    hist_bins = {k: v.bins for k, v in histos.items()}
    return leaf_class_counts, hist_bins

# ------------ Parallel SPDT ------------
class SPDTClassifierParallel(SPDT):
    """
    Parallel (W workers) version of SPDTClassifier.
    Uses data-parallel histogram building per depth; master merges and decides splits.
    """

    def __init__(
        self,
        B: int = 32,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        impurity: str = "gini",
        candidates_per_feature: Optional[int] = None,
        random_state: Optional[int] = None,
        W: int = 4,
        backend: str = "process",   # "process" or "thread"
        shard_strategy: str = "round_robin"  # or "contiguous"
    ):
        super().__init__(B=B, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         impurity=impurity, candidates_per_feature=candidates_per_feature,
                         random_state=random_state)
        assert backend in ("process", "thread")
        self.W = int(max(1, W))
        self.backend = backend
        self.shard_strategy = shard_strategy

    # ---------------- fit (overrides) ----------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        self._n_features = d
        self._n_classes = int(np.max(y)) + 1

        # init root
        self._nodes = {0: _Node(id=0, depth=0, is_leaf=True, prediction=self._majority(y, self._n_classes))}
        active_leaves = {0}

        # pre-shard data indices
        shards = self._make_shards(n)

        # level-by-level training
        for depth in range(self.max_depth):
            if not active_leaves:
                break

            # Broadcast snapshot
            tree_snap = _snapshot_tree(self._nodes)
            root_id = self._root_id

            # Run workers to build histograms on shards
            leaf_counts_agg: Dict[int, np.ndarray] = {lid: np.zeros(self._n_classes, dtype=float)
                                                      for lid in active_leaves}
            Hraw_agg: Dict[Tuple[int,int,int], List[Tuple[float, float]]] = {}

            if self.W == 1:
                # fall back to local worker call
                lc, hb = _build_histograms_worker(X, y, list(active_leaves), tree_snap, root_id,
                                                  self.B, self._n_classes, self._n_features)
                _merge_worker_results(leaf_counts_agg, Hraw_agg, lc, hb)
            else:
                Executor = ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
                with Executor(max_workers=self.W) as ex:
                    futures = []
                    for sidx in shards:
                        Xi, yi = X[sidx], y[sidx]
                        fut = ex.submit(
                            _build_histograms_worker,
                            Xi, yi,
                            list(active_leaves),
                            tree_snap,
                            root_id,
                            self.B,
                            self._n_classes,
                            self._n_features,
                        )
                        futures.append(fut)
                    for fut in as_completed(futures):
                        lc, hb = fut.result()
                        _merge_worker_results(leaf_counts_agg, Hraw_agg, lc, hb)

            # Turn raw bins into merged Histograms per (leaf, feat, cls)
            H: Dict[Tuple[int,int,int], Histogram] = {}
            for key, bins in Hraw_agg.items():
                h = Histogram(self.B)
                # inject and reduce
                h.bins = list(bins)
                h._reduce_bins()
                H[key] = h

            # ---- choose split per active leaf (same as base; uses H) ----
            splits: Dict[int, Tuple[Optional[int], Optional[float], float]] = {}
            for lid in list(active_leaves):
                parent_counts = leaf_counts_agg[lid]
                n_node = parent_counts.sum()

                # stopping: small node or pure
                if n_node < 2 * self.min_samples_leaf or parent_counts.max() == n_node:
                    splits[lid] = (None, None, 0.0)
                    continue

                parent_imp = self._impurity_fn(parent_counts)
                best_gain, best_feat, best_thr = 0.0, None, None

                for f in range(self._n_features):
                    # merged histogram across classes for candidates
                    merged_all = None
                    totals = np.zeros(self._n_classes, dtype=float)
                    per_class = {}
                    for c in range(self._n_classes):
                        key = (lid, f, c)
                        if key in H:
                            per_class[c] = H[key]
                            totals[c] = sum(m for _, m in H[key].bins)
                            if merged_all is None:
                                merged_all = Histogram(self.B)
                                merged_all.merge(H[key])
                            else:
                                merged_all.merge(H[key])

                    if merged_all is None:
                        continue

                    thresholds = merged_all.uniform()
                    if not thresholds:
                        continue
                    if self.candidates_per_feature is not None and len(thresholds) > self.candidates_per_feature:
                        idx = np.linspace(0, len(thresholds)-1, self.candidates_per_feature, dtype=int)
                        thresholds = [thresholds[i] for i in idx]

                    total_sum = totals.sum()
                    if total_sum <= 0:
                        continue

                    for thr in thresholds:
                        left_counts = np.array([
                            (0.0 if c not in per_class else per_class[c].sum(thr))
                            for c in range(self._n_classes)
                        ], dtype=float)

                        nl = left_counts.sum()
                        nr = total_sum - nl
                        if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                            continue

                        right_counts = totals - left_counts
                        tau = nl / (nl + nr)
                        gain = parent_imp - (tau * self._impurity_fn(left_counts)
                                             + (1.0 - tau) * self._impurity_fn(right_counts))
                        if gain > best_gain:
                            best_gain, best_feat, best_thr = gain, f, float(thr)

                splits[lid] = (best_feat, best_thr, best_gain)

            # ---- materialize splits ----
            new_active = set()
            next_id = max(self._nodes) + 1 if self._nodes else 0

            for lid, (feat, thr, gain) in splits.items():
                node = self._nodes[lid]
                if feat is None or thr is None or gain <= 0.0:
                    node.is_leaf = True
                    continue

                node.is_leaf = False
                node.split_feature = feat
                node.split_threshold = thr

                left_id, right_id = next_id, next_id + 1
                next_id += 2

                maj = node.prediction if node.prediction is not None else 0
                self._nodes[left_id]  = _Node(id=left_id,  depth=node.depth + 1, is_leaf=True, prediction=maj)
                self._nodes[right_id] = _Node(id=right_id, depth=node.depth + 1, is_leaf=True, prediction=maj)
                node.left, node.right = left_id, right_id

                if node.depth + 1 < self.max_depth:
                    new_active.add(left_id)
                    new_active.add(right_id)

            if not new_active:
                break
            active_leaves = new_active

            # optional refresh of leaf majorities
            leaf_counts_refresh = {lid: np.zeros(self._n_classes, dtype=int) for lid in active_leaves}
            for xi, yi in zip(X, y):
                lid = self._route_to_leaf(self._root_id, xi)
                if lid in leaf_counts_refresh:
                    leaf_counts_refresh[lid][yi] += 1
            for lid, cnt in leaf_counts_refresh.items():
                if cnt.sum() > 0:
                    self._nodes[lid].prediction = int(np.argmax(cnt))

        return self

    # ------------ sharding helpers ------------
    def _make_shards(self, n: int) -> List[np.ndarray]:
        idx = np.arange(n)
        if self.shard_strategy == "contiguous":
            # contiguous slices
            splits = np.array_split(idx, self.W)
            return [s for s in splits if s.size > 0]
        else:
            # round-robin for better class mixing
            rr = [idx[i::self.W] for i in range(self.W)]
            return [r for r in rr if r.size > 0]


def _merge_worker_results(
    leaf_counts_agg: Dict[int, np.ndarray],
    Hraw_agg: Dict[Tuple[int,int,int], List[Tuple[float, float]]],
    leaf_counts_local: Dict[int, np.ndarray],
    hist_bins_local: Dict[Tuple[int,int,int], List[Tuple[float, float]]],
):
    # sum leaf class counts
    for lid, cnt in leaf_counts_local.items():
        if lid not in leaf_counts_agg:
            leaf_counts_agg[lid] = cnt.astype(float, copy=True)
        else:
            leaf_counts_agg[lid] += cnt
    # concatenate raw bins; we'll reduce later
    for key, bins in hist_bins_local.items():
        if key not in Hraw_agg:
            Hraw_agg[key] = list(bins)
        else:
            Hraw_agg[key].extend(bins)
