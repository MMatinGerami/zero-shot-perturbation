import numpy as np
import pandas as pd
import scipy.sparse as sp

from zsp.data import Context, align, cp10k, log_fold_change
from zsp.emit import emit_cells
from zsp.models import (
    CalibratedTransfer,
    decompose,
    lfc_to_counts,
    predict_mean_transfer,
)


def _ctx(name, genes, symbols, control, pert, n=100):
    idx = pd.Index(genes, name="gene_id")
    return Context(
        name,
        idx,
        pd.Series(symbols, index=idx),
        np.asarray(control, float),
        pd.DataFrame(pert, columns=idx),
        pd.Series({p: n for p in pert.index}),
    )


def test_log_fold_change_zero_for_identical_profiles():
    x = np.array([[10.0, 5.0, 0.0]])
    np.testing.assert_allclose(log_fold_change(x, x[0]), 0.0)


def test_cp10k_sums_to_1e4():
    np.testing.assert_allclose(cp10k(np.array([3.0, 7.0])).sum(), 1e4)


def test_align_intersects_genes_in_first_order():
    a = _ctx(
        "a",
        ["g1", "g2", "g3"],
        ["A", "B", "C"],
        [1, 1, 1],
        pd.DataFrame([[1, 2, 3]], index=["P"], columns=["g1", "g2", "g3"]),
    )
    b = _ctx(
        "b",
        ["g3", "g1"],
        ["C", "A"],
        [1, 1],
        pd.DataFrame([[3, 1]], index=["P"], columns=["g3", "g1"]),
    )
    aa, bb = align([a, b])
    assert list(aa.genes) == ["g1", "g3"] and list(bb.genes) == ["g1", "g3"]
    assert bb.pert.loc["P"].tolist() == [1, 3]


def test_decompose_removes_generic_component():
    genes = [f"g{i}" for i in range(4)]
    control = np.array([10.0, 10.0, 10.0, 10.0])
    # every perturbation halves gene 0 (generic); P2 additionally doubles gene 1 (specific)
    pert = pd.DataFrame(
        [[5, 10, 10, 10], [5, 10, 10, 10], [5, 20, 10, 10]],
        index=["P0", "P1", "P2"],
        columns=genes,
    )
    src = decompose(_ctx("s", genes, ["P0", "P1", "P2", "X"], control, pert))
    assert src.generic[0] < -0.5  # halving is visible in the generic response
    assert abs(src.specific.at["P0", "g1"]) < 1e-6 and src.specific.at["P2", "g1"] > 0.5
    assert src.own_lfc["P0"] < 0  # target gene knocked down


def test_mean_transfer_falls_back_to_generic_for_unseen_perturbation():
    genes = ["g0", "g1"]
    pert = pd.DataFrame([[5.0, 10.0]], index=["P0"], columns=genes)
    src = decompose(_ctx("s", genes, ["P0", "Y"], [10.0, 10.0], pert))
    pred = predict_mean_transfer(
        [src], cp10k(np.array([10.0, 10.0]))[0], ["P0", "NEW"], pd.Index(genes)
    )
    assert pred.lfc.loc["NEW"].tolist() == src.generic.tolist()


def test_calibrated_transfer_gates_unexpressed_genes_and_sets_own_target():
    genes = ["g0", "g1", "g2"]
    symbols = ["P0", "B", "C"]
    pert = pd.DataFrame([[5.0, 20.0, 10.0]], index=["P0"], columns=genes)
    src = decompose(_ctx("s", genes, symbols, [10.0, 10.0, 10.0], pert))
    target_basal = cp10k(np.array([10.0, 10.0, 0.0]))[0]  # g2 silent in the target
    pred = CalibratedTransfer()([src], target_basal, ["P0"], pd.Index(genes), symbols)
    assert pred.lfc.at["P0", "g2"] == 0.0
    assert pred.lfc.at["P0", "g0"] < 0
    assert len(pred.uncertainty) == 1


def test_lfc_to_counts_roundtrip():
    control = np.array([100.0, 50.0, 25.0])
    lfc = np.array([[1.0, 0.0, -1.0]])
    out = lfc_to_counts(lfc, control)
    np.testing.assert_allclose(out[0, 1], 50.0, rtol=1e-3)
    assert out[0, 0] > 150 and out[0, 2] < 15


def test_emit_cells_matches_target_mean_and_integer_counts():
    rng = np.random.default_rng(0)
    ctrl = sp.csr_matrix(rng.poisson(20, size=(200, 5)).astype(np.float32))
    ctrl_mean = np.asarray(ctrl.mean(0)).ravel()
    target = ctrl_mean * np.array([2.0, 1.0, 0.5, 1.0, 1.0])
    cells = emit_cells(ctrl, ctrl_mean, target, 2000, rng)
    got = np.asarray(cells.mean(0)).ravel()
    np.testing.assert_allclose(got / ctrl_mean, [2.0, 1.0, 0.5, 1.0, 1.0], rtol=0.05)
    assert np.all(cells.data == np.round(cells.data))


def test_symbol_index_is_unique_and_named():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench", "scripts/02_local_benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    idx = bench.symbol_index(pd.Series(["TP53", "MYC", "TP53"]))
    assert idx.is_unique and idx.name == "gene" and idx[0] == "TP53"


def test_streaming_h5ad_roundtrip(tmp_path):
    from zsp.submission_io import StreamingH5ad

    var = pd.DataFrame(index=pd.Index(["A", "B", "C"], name="gene"))
    w = StreamingH5ad(tmp_path / "s.h5ad", var, n_obs=3)
    w.append(
        sp.csr_matrix([[1.0, 0.0, 2.0]]), pd.DataFrame({"target_gene": ["X"], "context": ["A"]})
    )
    w.append(
        sp.csr_matrix([[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]),
        pd.DataFrame({"target_gene": ["Y", "Y"], "context": ["B", "B"]}),
    )
    w.close()
    assert w.nnz == 3
    import anndata as ad

    a = ad.read_h5ad(tmp_path / "s.h5ad")
    assert a.shape == (3, 3) and a.var_names.tolist() == ["A", "B", "C"]
    np.testing.assert_array_equal(a.X.toarray(), [[1, 0, 2], [0, 3, 0], [0, 0, 0]])
    assert a.obs["target_gene"].tolist() == ["X", "Y", "Y"]


def test_align_union_keeps_unmeasured_genes_as_nan():
    from zsp.data import align_union

    a = _ctx(
        "a",
        ["g1", "g2"],
        ["A", "B"],
        [10, 10],
        pd.DataFrame([[5, 10]], index=["P"], columns=["g1", "g2"]),
    )
    b = _ctx(
        "b",
        ["g2", "g3"],
        ["B", "C"],
        [10, 10],
        pd.DataFrame([[10, 20]], index=["P"], columns=["g2", "g3"]),
    )
    aa, bb = align_union([a, b])
    assert list(aa.genes) == ["g1", "g2", "g3"] and list(bb.genes) == ["g1", "g2", "g3"]
    assert np.isnan(aa.control[2]) and np.isnan(bb.control[0])
    assert aa.measured.tolist() == [True, True, False]
    # cp10k ignores unmeasured genes; the decomposition leaves them NaN
    np.testing.assert_allclose(np.nansum(cp10k(bb.control)), 1e4)
    src_a, src_b = decompose(aa), decompose(bb)
    assert np.isnan(src_a.lfc.at["P", "g3"]) and np.isfinite(src_a.lfc.at["P", "g1"])
    # mean transfer averages measured sources per gene; g3 comes from b alone, no NaN output
    pred = predict_mean_transfer(
        [src_a, src_b], cp10k(np.array([10.0, 10.0, 10.0]))[0], ["P"], aa.genes
    )
    assert np.isfinite(pred.lfc.to_numpy()).all()
    assert pred.lfc.at["P", "g3"] > 0 and pred.lfc.at["P", "g1"] < 0


def _two_sources():
    genes = ["g0", "g1", "g2"]
    ctrl = [10.0, 10.0, 10.0]
    a = decompose(
        _ctx(
            "a",
            genes,
            ["P0", "B", "C"],
            ctrl,
            pd.DataFrame(
                [[5, 20, 10], [10, 10, 5], [10, 10, 10]],
                index=["P0", "P1", "P2"],
                columns=genes,
            ),
        )
    )
    b = decompose(
        _ctx(
            "b",
            genes,
            ["P0", "B", "C"],
            ctrl,
            pd.DataFrame(
                [[5, 10, 10], [10, 10, 20], [10, 10, 10]],
                index=["P0", "P1", "P2"],
                columns=genes,
            ),
        )
    )
    return genes, a, b


def test_weighted_transfer_interpolates_between_sources():
    from zsp.models import WeightedTransfer

    genes, a, b = _two_sources()
    basal_like_a = cp10k(np.array([10.0, 10.0, 10.0]))[0]
    pred = WeightedTransfer(temperature=1e6)(
        [a, b], basal_like_a, ["P0"], pd.Index(genes)
    )  # huge temperature = equal weights = mean transfer
    ref = predict_mean_transfer([a, b], basal_like_a, ["P0"], pd.Index(genes))
    np.testing.assert_allclose(pred.lfc.to_numpy(), ref.lfc.to_numpy(), atol=1e-6)


def test_scaled_and_gene_scaled_transfer():
    from zsp.models import GeneScaledTransfer, ScaledTransfer

    genes, a, b = _two_sources()
    basal = cp10k(np.array([10.0, 10.0, 10.0]))[0]
    ref = predict_mean_transfer([a, b], basal, ["P0", "P1"], pd.Index(genes)).lfc.to_numpy()
    half = ScaledTransfer(0.5)([a, b], basal, ["P0", "P1"], pd.Index(genes)).lfc.to_numpy()
    np.testing.assert_allclose(half, 0.5 * ref)
    m = GeneScaledTransfer(lam=1.0).fit([a, b])
    assert m.beta.shape == (3,) and np.all(m.beta <= 1.0 + 1e-9)
    # gene g0 responds identically in both sources -> transfers -> beta near 1;
    # genes g1/g2 disagree between sources -> damped
    assert m.beta[0] > m.beta[1] and m.beta[0] > m.beta[2]


def test_uncertainty_scores_columns_and_missing_source():
    from zsp.models import uncertainty_scores

    _, a, b = _two_sources()
    basal = cp10k(np.array([10.0, 10.0, 10.0]))[0]
    u = uncertainty_scores([a, b], basal, ["P0", "P1", "P2", "NEW"])
    assert list(u.columns) == [
        "disagreement",
        "disagreement_norm",
        "n_sources_inv",
        "effect_magnitude",
        "basal_distance",
    ]
    assert u.at["NEW", "n_sources_inv"] == 2.0 and u.at["P0", "n_sources_inv"] == 0.5
    assert u.at["P1", "disagreement"] > u.at["P2", "disagreement"]  # sources agree on P2
    assert np.isfinite(u.to_numpy()).all()


def test_loso_proxies_perfect_prediction_scores_one():
    from zsp.loso import proxies

    rng = np.random.default_rng(0)
    truth = rng.normal(size=(20, 500))
    out = proxies(truth.copy(), truth, k=50)
    for key in ("pds", "fid", "reach", "jac"):
        assert abs(out[key] - 1.0) < 1e-9
    assert out["mse"] < 1e-9 and out["nmae"] < 1e-9 and abs(out["mean_oriented"] - 1) < 1e-9
    zero = proxies(np.zeros_like(truth), truth, k=50)
    assert abs(zero["mse"] - 1.0) < 1e-9 and abs(zero["nmae"] - 1.0) < 1e-9
    # a prediction with the right signs but negligible magnitude earns no fidelity credit
    tiny = proxies(0.01 * truth, truth, k=50)
    assert tiny["fid"] == 0.0 and abs(tiny["pds"] - 1.0) < 1e-9


def test_norm_restored_and_median_transfer_keep_magnitude():
    from zsp.models import MedianTransfer, NormRestoredTransfer, WeightedTransfer

    genes = ["g0", "g1", "g2", "g3"]
    ctrl = [10.0, 10.0, 10.0, 10.0]
    # two sources that agree on g0 and disagree in sign on g1 -> the mean shrinks g1
    a = decompose(
        _ctx(
            "a",
            genes,
            ["P0", "B", "C", "D"],
            ctrl,
            pd.DataFrame([[5, 20, 10, 10]], index=["P0"], columns=genes),
        )
    )
    b = decompose(
        _ctx(
            "b",
            genes,
            ["P0", "B", "C", "D"],
            ctrl,
            pd.DataFrame([[5, 5, 10, 10]], index=["P0"], columns=genes),
        )
    )
    basal = cp10k(np.array(ctrl))[0]
    idx = pd.Index(genes)
    mean = WeightedTransfer(1e6)([a, b], basal, ["P0"], idx).lfc.to_numpy()
    restored = NormRestoredTransfer(1e6)([a, b], basal, ["P0"], idx).lfc.to_numpy()
    median = MedianTransfer(1e6)([a, b], basal, ["P0"], idx).lfc.to_numpy()
    src_norm = np.mean([np.linalg.norm(a.lfc.to_numpy()), np.linalg.norm(b.lfc.to_numpy())])
    assert np.linalg.norm(mean) < src_norm
    np.testing.assert_allclose(np.linalg.norm(restored), src_norm, rtol=1e-6)
    assert np.sign(restored[0, 0]) == np.sign(mean[0, 0]) == -1
    # equal weights: the weighted median of two values is the lower one -> keeps a real value
    assert median[0, 1] in (a.lfc.at["P0", "g1"], b.lfc.at["P0", "g1"])
    assert np.isfinite(median).all()


def test_depth_aware_transfer_weights_and_normalises_by_knockdown_depth():
    from zsp.models import DepthAwareTransfer, WeightedTransfer, knockdown_depth

    genes = ["g0", "g1", "g2"]
    ctrl = [10.0, 10.0, 10.0]
    # P0 targets g0: source a knocks it down deeply and g1 rises a lot; source b is a
    # shallow-knockdown screen with a proportionally smaller downstream response
    deep = decompose(
        _ctx(
            "a",
            genes,
            ["P0", "B", "C"],
            ctrl,
            pd.DataFrame([[5, 20, 10]], index=["P0"], columns=genes),
        )
    )
    shallow = decompose(
        _ctx(
            "b",
            genes,
            ["P0", "B", "C"],
            ctrl,
            pd.DataFrame([[8, 12, 10]], index=["P0"], columns=genes),
        )
    )
    assert knockdown_depth(deep) > knockdown_depth(shallow) > 0
    basal = cp10k(np.array(ctrl))[0]
    idx = pd.Index(genes)
    plain = WeightedTransfer(1e6)([deep, shallow], basal, ["P0"], idx).lfc.to_numpy()
    same = DepthAwareTransfer(1e6)([deep, shallow], basal, ["P0"], idx).lfc.to_numpy()
    np.testing.assert_allclose(same, plain, atol=1e-9)  # defaults = weighted transfer
    deep_only = DepthAwareTransfer(1e6, depth_weight=50)([deep, shallow], basal, ["P0"], idx)
    np.testing.assert_allclose(deep_only.lfc.to_numpy(), deep.lfc.to_numpy(), atol=1e-6)
    restored = DepthAwareTransfer(1e6, normalise=True)([deep, shallow], basal, ["P0"], idx)
    # the shallow screen's response is scaled up to the common depth, so the consensus
    # is larger than the plain mean and still smaller than the deep source alone
    assert np.linalg.norm(plain) < np.linalg.norm(restored.lfc.to_numpy())
    assert np.linalg.norm(restored.lfc.to_numpy()) <= np.linalg.norm(deep.lfc.to_numpy()) + 1e-9


def test_basal_modulated_transfer_reduces_to_weighted_and_modulates_by_delta():
    from zsp.models import BasalModulatedTransfer, WeightedTransfer

    genes, a, b = _two_sources()
    basal = cp10k(np.array([10.0, 10.0, 10.0]))[0]
    idx = pd.Index(genes)
    # sources share one basal profile: every delta is 0, one bin, multiplier 1 -> weighted
    same = BasalModulatedTransfer(1e6).fit([a, b])([a, b], basal, ["P0", "P1"], idx)
    ref = WeightedTransfer(1e6)([a, b], basal, ["P0", "P1"], idx)
    np.testing.assert_allclose(same.lfc.to_numpy(), ref.lfc.to_numpy(), atol=1e-9)
    # a target line that barely expresses g1 (delta << 0) with a hand-set multiplier of 0
    # for that bin: g1's response is removed, and "consensus" restores the row norm
    model = BasalModulatedTransfer(1e6, restore="consensus")
    model.multipliers_ = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    low_g1 = cp10k(np.array([10.0, 0.01, 10.0]))[0]
    out = model([a, b], low_g1, ["P0"], idx).lfc.to_numpy()
    plain = WeightedTransfer(1e6)([a, b], low_g1, ["P0"], idx).lfc.to_numpy()
    assert out[0, 1] == 0.0 and plain[0, 1] != 0.0
    np.testing.assert_allclose(np.linalg.norm(out), np.linalg.norm(plain), rtol=1e-9)
