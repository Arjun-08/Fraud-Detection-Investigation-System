# Model Explainability
For a tree model, SHAP gives an additive view f(x) = E[f(X)] + sum_i phi_i. A positive phi_i pushes the model score toward fraud and a negative phi_i pushes it away. SHAP is model evidence, not independent proof of fraud.
