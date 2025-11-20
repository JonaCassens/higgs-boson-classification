import xgboost as xgb
import yaml
import numpy as np


class XGBoostClassifier:
    """
    Unified XGBoost classifier that auto-detects binary vs multiclass mode.
    
    Binary mode (2 classes): Uses objective='binary:logistic', eval_metric=['aucpr', 'logloss']
    Multiclass mode (>2 classes): Uses objective='multi:softprob', eval_metric=['mlogloss', 'merror']
    """
    
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        self.model = None
        self.num_classes = None  # Determined during training
        
    def train(self, x_train, y_train, x_val=None, y_val=None):
        # Auto-detect number of classes
        self.num_classes = len(np.unique(y_train))
        
        # Build params based on number of classes
        params = self.config.copy()
        
        if self.num_classes == 2:
            # Binary classification
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = ['aucpr', 'logloss']
            # scale_pos_weight is useful for binary imbalanced data
        else:
            # Multiclass classification - override objective, eval_metric, and add num_class
            params['objective'] = 'multi:softprob'
            params['num_class'] = self.num_classes
            params['eval_metric'] = ['mlogloss', 'merror']
            # Remove binary-specific parameters
            params.pop('scale_pos_weight', None)
        
        self.model = xgb.XGBClassifier(**params)
        
        # Train with or without validation set
        if x_val is not None and y_val is not None:
            eval_set = [(x_train, y_train), (x_val, y_val)]
            self.model.fit(x_train, y_train, eval_set=eval_set, verbose=True)
        else:
            self.model.fit(x_train, y_train)
        
        return self.model
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    
    def get_feature_importance(self):
        return self.model.feature_importances_