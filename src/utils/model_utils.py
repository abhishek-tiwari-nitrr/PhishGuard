import sys
from src.exception import ApplicationException


class PhishGuard:
    """
    Wrapper class for phishing detection predictions.

    Attributes:
        - preprocessor(Any): Preprocessing object used to transform raw input data before prediction.
        - model(Any): Trained machine learning model used for classification.

    """

    def __init__(self, preprocessor, model):
        """
        Initialize the PhishGuard prediction pipeline.

        Args:
            - preprocessor(Any): Preprocessing object used to transform raw input data before prediction.
            - model(Any): Trained machine learning model used for classification.

        Raises:
            - ApplicationException: If initialization of attributes fails.
        """
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise ApplicationException(e, sys)

    def predict(self, x):
        """
        Predict class labels for input data.

        Args:
            - x (Any): Input feature data.

        Raises:
            - ApplicationException: If prediction or preprocessing fails.

        Returns:
            - np.ndarray: Predicted class labels.

        """
        try:
            x_transformed = self.preprocessor.transform(x)
            return self.model.predict(x_transformed)
        except Exception as e:
            raise ApplicationException(e, sys)

    def predict_proba(self, x):
        """
        Predict class probabilities for input data.

        Args:
            - x (Any): Input feature data.

        Raises:
            - AttributeError: If the model does not support probability prediction.
            - ApplicationException: If preprocessing or prediction fails.

        Returns:
            - np.ndarray: Predicted probability scores.

        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Underlying model does not support predict_proba")
        try:
            x_transformed = self.preprocessor.transform(x)
            return self.model.predict_proba(x_transformed)
        except Exception as e:
            raise ApplicationException(e, sys)
