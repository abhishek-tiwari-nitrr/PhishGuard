import os, sys, pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.exception import ApplicationException
from src.utils.model_utils import PhishGuard
from src.utils.utils import get_classification_score
from src.entity.artifact_entity import ClassificationModelArtifact


class TestApplicationException:
    def test_str_contains_file_and_line(self):
        try:
            raise ValueError("test")
        except ValueError as e:
            exc = ApplicationException(e, sys)

        result = str(exc)
        assert "|" in result, "Expected 'filename:lineno | message' format"
        parts = result.split("|")
        assert len(parts) == 2
        assert "test" in parts[1]

    def test_message_preserved(self):
        try:
            raise RuntimeError("original message here")
        except RuntimeError as e:
            exc = ApplicationException(e, sys)

        assert "original message here" in str(exc)

    def test_no_sys_arg_falls_back_gracefully(self):
        exc = ApplicationException("plain error")
        assert "plain error" in str(exc)


class TestPhishGuardWrapper:
    def _make_model(self, predictions, probas=None):
        preprocessor = MagicMock()
        preprocessor.transform.side_effect = lambda x: x
        model = MagicMock()
        model.predict.return_value = np.array(predictions)
        if probas is not None:
            model.predict_proba.return_value = np.array(probas)
        return preprocessor, model

    def test_predict_calls_transform_then_predict(self):
        preprocessor, model = self._make_model([1, 0, 1])
        pg = PhishGuard(preprocessor=preprocessor, model=model)

        X = pd.DataFrame({"a": [1, 2, 3]})
        result = pg.predict(X)

        preprocessor.transform.assert_called_once_with(X)
        model.predict.assert_called_once()
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_predict_proba_returns_probabilities(self):
        probas = [[0.2, 0.8], [0.9, 0.1]]
        preprocessor, model = self._make_model([1, 0], probas=probas)
        pg = PhishGuard(preprocessor=preprocessor, model=model)

        X = pd.DataFrame({"a": [1, 2]})
        result = pg.predict_proba(X)

        np.testing.assert_array_almost_equal(result, probas)


class TestGetClassificationScore:
    def test_perfect_predictions(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        artifact = get_classification_score(y_true, y_pred)

        assert isinstance(artifact, ClassificationModelArtifact)
        assert artifact.f1_score == pytest.approx(1.0)
        assert artifact.precision_score == pytest.approx(1.0)
        assert artifact.recall_score == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = [1, 1, 1, 0, 0]
        y_pred = [0, 0, 0, 1, 1]
        artifact = get_classification_score(y_true, y_pred)

        assert artifact.f1_score == pytest.approx(0.0, abs=1e-6)

    def test_partial_predictions(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        artifact = get_classification_score(y_true, y_pred)

        assert 0.0 < artifact.f1_score < 1.0
        assert 0.0 <= artifact.precision_score <= 1.0
        assert 0.0 <= artifact.recall_score <= 1.0


class TestModelPusher:

    def _make_pusher(self, is_accepted: bool):
        from src.components.model_pusher import ModelPusher

        eval_artifact = MagicMock()
        eval_artifact.is_model_accepted = is_accepted
        eval_artifact.trained_model_file_path = "/tmp/trained_model.pkl"

        eval_config = MagicMock()
        eval_config.production_model_file_path = "/tmp/production_model/model.pkl"
        eval_config.production_model_dir = "/tmp/production_model"

        return ModelPusher(eval_artifact, eval_config)

    def test_pusher_copies_when_accepted(self):
        pusher = self._make_pusher(is_accepted=True)

        with (
            patch("src.components.model_pusher.os.makedirs") as mock_mkdir,
            patch("src.components.model_pusher.shutil.copy2") as mock_copy,
        ):
            result = pusher.initiate_model_pusher()

        assert result is True
        mock_mkdir.assert_called_once()
        mock_copy.assert_called_once_with(
            "/tmp/trained_model.pkl",
            "/tmp/production_model/model.pkl",
        )

    def test_pusher_skips_when_rejected(self):
        pusher = self._make_pusher(is_accepted=False)

        with patch("src.components.model_pusher.shutil.copy2") as mock_copy:
            result = pusher.initiate_model_pusher()

        assert result is False
        mock_copy.assert_not_called()
