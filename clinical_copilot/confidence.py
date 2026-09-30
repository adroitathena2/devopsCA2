"""
Confidence and uncertainty estimation for pipeline predictions.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from clinical_copilot.config import ConfigLoader

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """
    Estimates confidence and uncertainty for predictions using multiple methods:
    - Monte Carlo Dropout
    - Ensemble predictions
    - Temperature scaling
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize confidence estimator.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.method = config.get('uncertainty', 'method', default='ensemble')
        self.n_iterations = config.get('uncertainty', 'n_iterations', default=10)
        self.confidence_intervals = config.get('uncertainty', 'confidence_intervals', default=[0.95, 0.99])
    
    def estimate_confidence(self, predictions: List[float]) -> Dict[str, float]:
        """
        Estimate confidence from multiple predictions.
        
        Args:
            predictions: List of prediction scores
            
        Returns:
            Dictionary with confidence metrics
        """
        if not predictions:
            return {
                'mean': 0.0,
                'std': 0.0,
                'confidence': 0.0,
                'uncertainty': 1.0
            }
        
        predictions = np.array(predictions)
        
        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))
        
        confidence = 1.0 - min(std_pred, 1.0)
        
        percentiles = {}
        for ci in self.confidence_intervals:
            lower = (1 - ci) / 2
            upper = 1 - lower
            percentiles[f'ci_{int(ci*100)}_lower'] = float(np.percentile(predictions, lower * 100))
            percentiles[f'ci_{int(ci*100)}_upper'] = float(np.percentile(predictions, upper * 100))
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'confidence': confidence,
            'uncertainty': std_pred,
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            **percentiles
        }
    
    def calibrate_confidence(self, scores: List[float], temperature: float = 1.5) -> List[float]:
        """
        Apply temperature scaling to calibrate confidence scores.
        
        Args:
            scores: Raw confidence scores
            temperature: Temperature parameter (>1 softens, <1 sharpens)
            
        Returns:
            Calibrated scores
        """
        scores = np.array(scores)
        calibrated = scores ** (1.0 / temperature)
        calibrated = calibrated / np.sum(calibrated) * len(calibrated)
        return calibrated.tolist()
    
    def ensemble_confidence(self, predictions_list: List[List[float]]) -> Dict[str, Any]:
        """
        Aggregate confidence from ensemble of predictions.
        
        Args:
            predictions_list: List of prediction arrays from different models
            
        Returns:
            Aggregated confidence metrics
        """
        if not predictions_list:
            return {'confidence': 0.0, 'uncertainty': 1.0}
        
        mean_predictions = np.mean(predictions_list, axis=0)
        
        variance = np.var(predictions_list, axis=0)
        
        return {
            'ensemble_mean': float(np.mean(mean_predictions)),
            'ensemble_variance': float(np.mean(variance)),
            'confidence': float(1.0 - min(np.mean(variance), 1.0)),
            'uncertainty': float(np.mean(variance)),
            'agreement_score': float(1.0 - np.mean(variance))
        }
