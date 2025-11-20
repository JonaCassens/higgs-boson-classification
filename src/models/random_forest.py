import yaml
from sklearn.ensemble import RandomForestClassifier as RFC


class RandomForestClassifier:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                config.pop('model', None)
                self.config = config
        else:
            self.config = {}
        self.model = None
        
    def train(self, x_train, y_train):
        self.model = RFC(**self.config)
        self.model.fit(x_train, y_train)
        return self.model
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    
    def get_feature_importance(self):
        return self.model.feature_importances_
