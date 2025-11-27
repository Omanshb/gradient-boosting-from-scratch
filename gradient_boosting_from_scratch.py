import numpy as np
from collections import Counter


class DecisionTreeRegressor:
    """Decision Tree for Regression based tasks - used as base learner in Gradient Boosting"""
    
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _calculate_mse(self, y):
        """Calculate Mean Squared Error"""
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)
    
    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Performance optimization: limit number of split candidates
            # Use percentiles instead of all unique values for large datasets
            unique_values = np.unique(feature_values)
            if len(unique_values) > 100:
                # Use 100 percentile-based thresholds for efficiency
                percentiles = np.linspace(0, 100, 101)[1:-1]  # Exclude 0 and 100
                thresholds = np.percentile(feature_values, percentiles)
                thresholds = np.unique(thresholds)  # Remove duplicates
            else:
                thresholds = unique_values
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_mse = self._calculate_mse(y[left_mask])
                right_mse = self._calculate_mse(y[right_mask])
                total_mse = left_mse + right_mse
                
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Leaf node: return mean of y values
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'value': np.mean(y)}
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return {'value': np.mean(y)}
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Fit the decision tree"""
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Predict a single sample"""
        if 'value' in tree:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Predict for multiple samples"""
        X = np.array(X)
        return np.array([self._predict_sample(x, self.tree) for x in X])


class GradientBoostingRegressor:
    """
    Gradient Boosting for Regression
    
    Mathematical Formulation:
    F_0(x) = mean(y)
    For m = 1 to M:
        1. Compute residuals: r_i = y_i - F_{m-1}(x_i)
        2. Fit a tree h_m(x) to residuals
        3. Update: F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X, y):
        """Fit the gradient boosting model"""
        X = np.array(X)
        y = np.array(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Step 1: Initialize with mean of y (F_0(x))
        self.initial_prediction = np.mean(y)
        
        # Current predictions
        F = np.full(len(y), self.initial_prediction)
        
        # Step 2: Iteratively add trees
        for m in range(self.n_estimators):
            # Compute negative gradient (residuals for MSE loss)
            residuals = y - F
            
            # Subsample if needed
            if self.subsample < 1.0:
                n_samples = int(len(X) * self.subsample)
                indices = np.random.choice(len(X), n_samples, replace=False)
                X_subset = X[indices]
                residuals_subset = residuals[indices]
            else:
                X_subset = X
                residuals_subset = residuals
            
            # Fit a regression tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, residuals_subset)
            
            # Update predictions: F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Predict using the gradient boosting model"""
        X = np.array(X)
        
        # Start with initial prediction
        F = np.full(len(X), self.initial_prediction)
        
        # Add predictions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        return F
    
    def staged_predict(self, X):
        """Generate predictions at each stage (useful for visualization)"""
        X = np.array(X)
        F = np.full(len(X), self.initial_prediction)
        
        predictions_by_stage = [F.copy()]
        
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
            predictions_by_stage.append(F.copy())
        
        return predictions_by_stage


class GradientBoostingClassifier:
    """
    Gradient Boosting for Binary Classification
    
    Mathematical Formulation:
    Uses log-loss (binary cross-entropy) as the loss function
    
    F_0(x) = log(p / (1-p)) where p = mean(y)
    For m = 1 to M:
        1. Compute negative gradient: r_i = y_i - p_i where p_i = sigmoid(F_{m-1}(x_i))
        2. Fit a tree h_m(x) to the negative gradient
        3. Update: F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
    
    Final prediction: p(x) = sigmoid(F_M(x))
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None
        self.classes_ = None
    
    def _sigmoid(self, x):
        """Sigmoid function with numerical stability"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _log_odds(self, p):
        """Convert probability to log-odds"""
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))
    
    def fit(self, X, y):
        """Fit the gradient boosting classifier"""
        X = np.array(X)
        y = np.array(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Store classes
        self.classes_ = np.unique(y)
        
        # Convert to binary 0/1
        y_binary = (y == self.classes_[1]).astype(int)
        
        # Step 1: Initialize with log-odds of positive class
        p_positive = np.mean(y_binary)
        p_positive = np.clip(p_positive, 1e-15, 1 - 1e-15)
        self.initial_prediction = self._log_odds(p_positive)
        
        # Current predictions in log-odds space
        F = np.full(len(y), self.initial_prediction)
        
        # Step 2: Iteratively add trees
        for m in range(self.n_estimators):
            # Compute probabilities
            probabilities = self._sigmoid(F)
            
            # Compute negative gradient (residuals)
            residuals = y_binary - probabilities
            
            # Subsample if needed
            if self.subsample < 1.0:
                n_samples = int(len(X) * self.subsample)
                indices = np.random.choice(len(X), n_samples, replace=False)
                X_subset = X[indices]
                residuals_subset = residuals[indices]
            else:
                X_subset = X
                residuals_subset = residuals
            
            # Fit a regression tree to the negative gradient
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, residuals_subset)
            
            # Update predictions
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            
            self.trees.append(tree)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        
        # Start with initial prediction
        F = np.full(len(X), self.initial_prediction)
        
        # Add predictions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        # Convert to probabilities
        prob_positive = self._sigmoid(F)
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]
    
    def staged_predict_proba(self, X):
        """Generate probability predictions at each stage"""
        X = np.array(X)
        F = np.full(len(X), self.initial_prediction)
        
        probabilities_by_stage = [self._sigmoid(F.copy())]
        
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
            probabilities_by_stage.append(self._sigmoid(F.copy()))
        
        return probabilities_by_stage


class GradientBoostingMultiClassifier:
    """
    Gradient Boosting for Multi-class Classification
    
    Uses one-vs-rest strategy with multiple GradientBoostingClassifier instances
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.classifiers = []
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit multi-class gradient boosting"""
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification
            clf = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                random_state=self.random_state
            )
            clf.fit(X, y)
            self.classifiers.append(clf)
        else:
            # Multi-class: one-vs-rest
            for i, class_label in enumerate(self.classes_):
                y_binary = (y == class_label).astype(int)
                
                clf = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    subsample=self.subsample,
                    random_state=self.random_state if self.random_state is None else self.random_state + i
                )
                clf.fit(X, y_binary)
                self.classifiers.append(clf)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        
        if len(self.classes_) == 2:
            return self.classifiers[0].predict_proba(X)
        
        # Get probabilities from each one-vs-rest classifier
        all_probs = []
        for clf in self.classifiers:
            probs = clf.predict_proba(X)[:, 1]  # Probability of positive class
            all_probs.append(probs)
        
        all_probs = np.column_stack(all_probs)
        
        # Normalize to sum to 1
        row_sums = all_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        all_probs = all_probs / row_sums
        
        return all_probs
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]

