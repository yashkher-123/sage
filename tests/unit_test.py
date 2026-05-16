import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sage_base import Sage_Explainer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })

@pytest.fixture
def linear_explainer(sample_data):
    """Explainer with a known linear predict function: pred = 2a + 3b"""
    predict_func = lambda df: (df["a"] * 2 + df["b"] * 3).values
    explainer = Sage_Explainer(predict_func=predict_func)
    explainer.fit(sample_data)
    return explainer

@pytest.fixture
def linear_explainer_with_data(sample_data):
    """Returns both the explainer and sample_data for tests that need both"""
    predict_func = lambda df: (df["a"] * 2 + df["b"] * 3).values
    explainer = Sage_Explainer(predict_func=predict_func)
    explainer.fit(sample_data)
    return explainer, sample_data



class TestGetScaledStdRanges:

    def test_output_keys_match_columns(self, sample_data):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        result = explainer.get_scaled_std_ranges(sample_data, perturbation_factor=0.3)
        assert set(result.keys()) == set(sample_data.columns)

    def test_scaling_is_correct(self, sample_data):
        factor = 0.3
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        result = explainer.get_scaled_std_ranges(sample_data, perturbation_factor=factor)
        raw_std = sample_data["a"].std(ddof=0)
        assert abs(result["a"] - raw_std * factor) < 1e-10

    def test_single_column(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        explainer = Sage_Explainer(predict_func=lambda df: df["x"].values)
        result = explainer.get_scaled_std_ranges(data, perturbation_factor=0.5)
        assert "x" in result
        assert len(result) == 1



class TestGetPerturbations:

    def test_original_value_excluded(self):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        ranges = {"a": (0.7, 1.3)}  # midpoint (original val) = 1.0
        result = explainer.get_perturbations(ranges, num_samples=10)
        for val in result["a"]:
            assert not np.isclose(val, 1.0), f"original value 1.0 found in perturbations"

    def test_perturbations_within_range(self):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        low, high = 0.7, 1.3
        ranges = {"a": (low, high)}
        result = explainer.get_perturbations(ranges, num_samples=10)
        for val in result["a"]:
            assert low <= val <= high

    def test_correct_num_samples(self):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        # midpoint is 1.0, linspace(0.7, 1.3, 10) — one point should be removed (the midpoint)
        ranges = {"a": (0.7, 1.3)}
        result = explainer.get_perturbations(ranges, num_samples=10)
        # should have at most num_samples - 1 points (one removed if midpoint lands exactly)
        assert len(result["a"]) <= 10

    def test_multiple_features(self):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        ranges = {"a": (0.7, 1.3), "b": (2.0, 4.0)}
        result = explainer.get_perturbations(ranges, num_samples=8)
        assert "a" in result and "b" in result


# ── fit ──────────────────────────────────────────────────────────────────────

class TestFit:

    def test_std_dict_populated(self, sample_data):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        explainer.fit(sample_data)
        assert hasattr(explainer, "std_dict")
        assert len(explainer.std_dict) == len(sample_data.columns)

    def test_used_features_defaults_to_all(self, sample_data):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        explainer.fit(sample_data)
        assert set(explainer.used_features) == set(sample_data.columns)

    def test_ignore_features_defaults_to_empty(self, sample_data):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        explainer.fit(sample_data)
        assert explainer.ignore_features == []

    def test_non_numeric_columns_excluded_from_std_dict(self):
        data = pd.DataFrame({
            "a": np.random.randn(50),
            "label": ["x"] * 50  # non-numeric
        })
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        explainer.fit(data)
        assert "label" not in explainer.std_dict
        assert "a" in explainer.std_dict

    def test_custom_perturbation_strength(self, sample_data):
        explainer = Sage_Explainer(predict_func=lambda df: df["a"].values)
        explainer.fit(sample_data, perturbation_strength=0.5)
        assert explainer.perturbation_factor == 0.5


# ── explain ──────────────────────────────────────────────────────────────────

class TestExplain:

    def test_accepts_series(self, linear_explainer):
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = linear_explainer.explain(instance)
        assert isinstance(result, dict)

    def test_accepts_dict(self, linear_explainer):
        instance = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = linear_explainer.explain(instance)
        assert isinstance(result, dict)

    def test_ignored_feature_absent_from_output(self, sample_data):
        predict_func = lambda df: (df["a"] * 2 + df["b"] * 3).values
        explainer = Sage_Explainer(predict_func=predict_func)
        explainer.fit(sample_data, ignore_features=["c"])
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = explainer.explain(instance)
        assert "c" not in result

    def test_used_features_restriction(self, sample_data):
        predict_func = lambda df: (df["a"] * 2 + df["b"] * 3).values
        explainer = Sage_Explainer(predict_func=predict_func)
        explainer.fit(sample_data, used_features=["a"])
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = explainer.explain(instance)
        assert set(result.keys()) == {"a"}

    def test_output_keys_are_subset_of_features(self, linear_explainer):
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = linear_explainer.explain(instance)
        assert set(result.keys()).issubset({"a", "b", "c"})

    def test_output_values_are_scalars(self, linear_explainer):
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = linear_explainer.explain(instance)
        for val in result.values():
            assert isinstance(val, float)


# ── regress_sensitivity / core accuracy ─────────────────────────────────────

class TestRegressSensitivity:

    def test_linear_model_sensitivity_feature_a(self, linear_explainer):
        """for pred = 2a + 3b, sensitivity of a should be ~2"""
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = linear_explainer.explain(instance)
        assert abs(result["a"] - 2.0) < 0.1

    def test_linear_model_sensitivity_feature_b(self, linear_explainer):
        """for pred = 2a + 3b, sensitivity of b should be ~3"""
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = linear_explainer.explain(instance)
        assert abs(result["b"] - 3.0) < 0.1

    def test_negative_sensitivity(self, sample_data):
        """for pred = -2a, sensitivity of a should be ~-2"""
        predict_func = lambda df: (-2 * df["a"]).values
        explainer = Sage_Explainer(predict_func=predict_func)
        explainer.fit(sample_data)
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = explainer.explain(instance)
        assert abs(result["a"] - (-2.0)) < 0.1

    def test_zero_sensitivity_for_unused_feature(self, sample_data):
        """for pred = 2a, sensitivity of b and c should be ~0"""
        predict_func = lambda df: (2 * df["a"]).values
        explainer = Sage_Explainer(predict_func=predict_func)
        explainer.fit(sample_data)
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        result = explainer.explain(instance)
        assert abs(result["b"]) < 0.1
        assert abs(result["c"]) < 0.1



class TestGraph:

    def test_graph_after_explain_runs_without_error(self, linear_explainer):
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        linear_explainer.explain(instance)
        linear_explainer.graph()  # should not raise

    def test_graph_with_instance_arg_runs_without_error(self, linear_explainer):
        instance = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        linear_explainer.graph(instance)  # should not raise

    def test_graph_without_prior_explain_raises(self, sample_data):
        """Calling graph() cold (no explain, no instance arg) should raise AttributeError"""
        predict_func = lambda df: (df["a"] * 2).values
        explainer = Sage_Explainer(predict_func=predict_func)
        explainer.fit(sample_data)
        with pytest.raises(AttributeError):
            explainer.graph()