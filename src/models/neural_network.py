import yaml
import torch
import torch.nn as nn


class NeuralNetwork:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        self.model = None
        self.optimizer = None
        
    def build_model(self, input_dim):
        layers = []
        prev_units = input_dim
        
        for layer_config in self.config.get('hidden_layers', []):
            layers.append(nn.Linear(prev_units, layer_config['units']))
            layers.append(nn.ReLU())
            prev_units = layer_config['units']
        
        layers.append(nn.Linear(prev_units, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        lr = self.config.get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        return self.model
    
    def train(self, x_train, y_train, x_val=None, y_val=None):
        if self.model is None:
            self.build_model(x_train.shape[1])
        
        if hasattr(x_train, 'values'):
            x_train = x_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
            
        X = torch.FloatTensor(x_train)
        y = torch.FloatTensor(y_train).view(-1, 1)
        
        batch_size = self.config.get('batch_size', 64)
        epochs = self.config.get('epochs', 50)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            
            avg_loss = total_loss / len(loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    def predict(self, x_test):
        self.model.eval()
        if hasattr(x_test, 'values'):
            x_test = x_test.values
        X = torch.FloatTensor(x_test)
        
        with torch.no_grad():
            proba = self.model(X).numpy()
        return (proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, x_test):
        self.model.eval()
        if hasattr(x_test, 'values'):
            x_test = x_test.values
        X = torch.FloatTensor(x_test)
        
        with torch.no_grad():
            return self.model(X).numpy()
