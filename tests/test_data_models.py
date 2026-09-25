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
