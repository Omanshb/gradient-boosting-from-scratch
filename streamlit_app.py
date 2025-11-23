import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

from gradient_boosting_from_scratch import (
    GradientBoostingRegressor, 
    GradientBoostingClassifier,
    GradientBoostingMultiClassifier
)


def load_kaggle_loan_dataset():
    """Load the Kaggle loan prediction dataset from playground-series-s5e11"""
    dataset_path = os.path.join(os.path.dirname(__file__), 'playground-series-s5e11', 'train.csv')
    
    try:
        df = pd.read_csv(dataset_path)
        return df, 'loan_paid_back', 'classification'
    except FileNotFoundError:
        st.error(f"Dataset not found at {dataset_path}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None, None


def encode_categorical_features(df, categorical_cols):
    """Encode categorical features using label encoding"""
    df_encoded = df.copy()
    encodings = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            unique_vals = df_encoded[col].unique()
            encoding = {val: idx for idx, val in enumerate(unique_vals)}
            df_encoded[col] = df_encoded[col].map(encoding)
            encodings[col] = encoding
    
    return df_encoded, encodings


def calculate_metrics_classification(y_true, y_pred, y_proba=None):
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            pass
    
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm


def calculate_metrics_regression(y_true, y_pred):
    """Calculate regression metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix"""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500
    )
    
    return fig


def plot_feature_importance(feature_names, importances):
    """Plot feature importance"""
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
    fig = go.Figure(data=[
        go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title='Feature Importance (Top 20)',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=max(400, len(indices) * 20),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_learning_curve(train_scores, test_scores):
    """Plot learning curve"""
    iterations = list(range(1, len(train_scores) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=train_scores,
        mode='lines',
        name='Training Score',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=test_scores,
        mode='lines',
        name='Test Score',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Learning Curve (Score vs Number of Trees)',
        xaxis_title='Number of Trees',
        yaxis_title='Score',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_predictions_regression(y_true, y_pred):
    """Plot actual vs predicted for regression"""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual',
        yaxis_title='Predicted',
        height=500,
        width=600
    )
    
    return fig


def plot_residuals(y_true, y_pred):
    """Plot residuals for regression"""
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Residuals Distribution')
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='blue', size=6, opacity=0.6),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Zero line
    fig.add_trace(
        go.Scatter(
            x=[y_pred.min(), y_pred.max()],
            y=[0, 0],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Histogram of residuals
    fig.add_trace(
        go.Histogram(
            x=residuals,
            marker=dict(color='lightblue'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Predicted', row=1, col=1)
    fig.update_yaxes(title_text='Residuals', row=1, col=1)
    fig.update_xaxes(title_text='Residuals', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)
    
    fig.update_layout(height=400, title_text='Residual Analysis')
    
    return fig


def main():
    st.set_page_config(
        page_title="Gradient Boosting from Scratch",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Gradient Boosted Decision Trees from Scratch")
    
    st.markdown("""
    This application demonstrates **Gradient Boosted Decision Trees (GBDT)** implemented from scratch.
    """)
    
    st.sidebar.header("Loan Prediction Dataset")
    st.sidebar.info("**Kaggle Competition:** playground-series-s5e11\n\nPredict whether a borrower will pay back their loan")
    
    # Load the dataset
    df, target_col, task = load_kaggle_loan_dataset()
    dataset_name = "Loan Prediction (playground-series-s5e11)"
    
    if df is not None:
        st.sidebar.success(f"✓ Loaded {df.shape[0]} training samples")
        st.sidebar.write(f"**Features:** {df.shape[1] - 1}")
        st.sidebar.write(f"**Target:** {target_col}")
    
    if df is not None and target_col is not None:
        st.header(f"Dataset: {dataset_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1] - 1)
        with col3:
            st.metric("Task", task.capitalize())
        
        with st.expander("Dataset Preview"):
            st.dataframe(df.head(10))
        
        # Data Preprocessing
        st.subheader("Data Preprocessing")
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        if categorical_cols:
            st.info(f"Categorical features detected: {', '.join(categorical_cols)}")
            df, encodings = encode_categorical_features(df, categorical_cols)
            st.success(f"Categorical features encoded using **Label Encoding** (each unique category is assigned a unique integer)")
        
        # Handle missing values
        if df.isnull().any().any():
            st.warning("Missing values detected. Removing rows with missing values...")
            df = df.dropna()
            st.info(f"Dataset reduced to {len(df)} samples")
        
        # Feature selection
        feature_cols = [col for col in df.columns if col != target_col]
        
        selected_features = st.multiselect(
            "Select features for training:",
            feature_cols,
            default=feature_cols
        )
        
        if len(selected_features) == 0:
            st.error("Please select at least one feature")
            st.stop()
        
        # Prepare data
        X = df[selected_features].values
        y = df[target_col].values
        
        # Scaling option
        scale_features = st.checkbox("Scale features (recommended)", value=True)
        
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # Train/Test Split
        st.subheader("Model Configuration")
        
        random_state = st.number_input("Random State", 0, 1000, 42, 1)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        st.info(f"Training samples: {len(X_train)} | Validation samples: {len(X_test)} (80/20 split)")
        
        # Model Parameters
        st.subheader("Gradient Boosting Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider("Number of Trees", 10, 300, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        
        with col2:
            max_depth = st.slider("Max Tree Depth", 1, 10, 3, 1)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)
        
        with col3:
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1, 1)
            subsample = st.slider("Subsample", 0.1, 1.0, 1.0, 0.1)
        
        # Train Model
        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training Gradient Boosting {task.capitalize()} model..."):
                try:
                    if task == 'classification':
                        n_classes = len(np.unique(y))
                        
                        if n_classes == 2:
                            model = GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                subsample=subsample,
                                random_state=random_state
                            )
                        else:
                            model = GradientBoostingMultiClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                subsample=subsample,
                                random_state=random_state
                            )
                        
                        model.fit(X_train, y_train)
                        
                        # Predictions
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        y_train_proba = model.predict_proba(X_train)
                        y_test_proba = model.predict_proba(X_test)
                        
                        # Calculate metrics
                        train_metrics, train_cm = calculate_metrics_classification(
                            y_train, y_train_pred, y_train_proba
                        )
                        test_metrics, test_cm = calculate_metrics_classification(
                            y_test, y_test_pred, y_test_proba
                        )
                        
                        st.session_state['model'] = model
                        st.session_state['task'] = task
                        st.session_state['train_metrics'] = train_metrics
                        st.session_state['test_metrics'] = test_metrics
                        st.session_state['train_cm'] = train_cm
                        st.session_state['test_cm'] = test_cm
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test
                        st.session_state['y_train_pred'] = y_train_pred
                        st.session_state['y_test_pred'] = y_test_pred
                        st.session_state['y_test_proba'] = y_test_proba
                        st.session_state['feature_names'] = selected_features
                        st.session_state['n_classes'] = n_classes
                        
                    else:  # Regression
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            subsample=subsample,
                            random_state=random_state
                        )
                        
                        model.fit(X_train, y_train)
                        
                        # Predictions
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        train_metrics = calculate_metrics_regression(y_train, y_train_pred)
                        test_metrics = calculate_metrics_regression(y_test, y_test_pred)
                        
                        st.session_state['model'] = model
                        st.session_state['task'] = task
                        st.session_state['train_metrics'] = train_metrics
                        st.session_state['test_metrics'] = test_metrics
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test
                        st.session_state['y_train_pred'] = y_train_pred
                        st.session_state['y_test_pred'] = y_test_pred
                        st.session_state['feature_names'] = selected_features
                    
                    st.success("Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
        
        # Display Results
        if 'model' in st.session_state:
            st.markdown("---")
            st.header("Results")
            
            task = st.session_state['task']
            train_metrics = st.session_state['train_metrics']
            test_metrics = st.session_state['test_metrics']
            
            if task == 'classification':
                # Metrics
                st.subheader("Model Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Training Set")
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.metric("Accuracy", f"{train_metrics['accuracy']:.4f}")
                        st.metric("Precision", f"{train_metrics['precision']:.4f}")
                    with metric_cols[1]:
                        st.metric("Recall", f"{train_metrics['recall']:.4f}")
                        st.metric("F1 Score", f"{train_metrics['f1']:.4f}")
                    
                    if 'roc_auc' in train_metrics:
                        st.metric("ROC AUC", f"{train_metrics['roc_auc']:.4f}")
                
                with col2:
                    st.markdown("#### Test Set")
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.metric("Accuracy", f"{test_metrics['accuracy']:.4f}")
                        st.metric("Precision", f"{test_metrics['precision']:.4f}")
                    with metric_cols[1]:
                        st.metric("Recall", f"{test_metrics['recall']:.4f}")
                        st.metric("F1 Score", f"{test_metrics['f1']:.4f}")
                    
                    if 'roc_auc' in test_metrics:
                        st.metric("ROC AUC", f"{test_metrics['roc_auc']:.4f}")
                
                # Confusion Matrices
                st.subheader("Confusion Matrices")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Training Set")
                    fig_train_cm = plot_confusion_matrix(st.session_state['train_cm'])
                    st.plotly_chart(fig_train_cm, use_container_width=True)
                
                with col2:
                    st.markdown("#### Test Set")
                    fig_test_cm = plot_confusion_matrix(st.session_state['test_cm'])
                    st.plotly_chart(fig_test_cm, use_container_width=True)
                
            else:  # Regression
                st.subheader("Model Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Training Set")
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.metric("R² Score", f"{train_metrics['r2']:.4f}")
                        st.metric("RMSE", f"{train_metrics['rmse']:.4f}")
                    with metric_cols[1]:
                        st.metric("MSE", f"{train_metrics['mse']:.4f}")
                        st.metric("MAE", f"{train_metrics['mae']:.4f}")
                
                with col2:
                    st.markdown("#### Test Set")
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.metric("R² Score", f"{test_metrics['r2']:.4f}")
                        st.metric("RMSE", f"{test_metrics['rmse']:.4f}")
                    with metric_cols[1]:
                        st.metric("MSE", f"{test_metrics['mse']:.4f}")
                        st.metric("MAE", f"{test_metrics['mae']:.4f}")
                
                # Prediction plots
                st.subheader("Prediction Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pred = plot_predictions_regression(
                        st.session_state['y_test'],
                        st.session_state['y_test_pred']
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                with col2:
                    fig_resid = plot_residuals(
                        st.session_state['y_test'],
                        st.session_state['y_test_pred']
                    )
                    st.plotly_chart(fig_resid, use_container_width=True)
    
    else:
        st.error("Failed to load the Kaggle Loan Prediction dataset. Please check that the 'playground-series-s5e11/train.csv' file exists.")


if __name__ == "__main__":
    main()

