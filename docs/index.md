---
title: open-mlpipe — Production ML Pipeline
---

<div class="hero-banner">
  <h1>🔥 open-mlpipe</h1>
  <p class="subtitle">Production-level automated ML pipeline — from raw data to deployed model in one line.</p>
  
  <div class="hero-badges">
    <a href="https://pypi.org/project/open-mlpipe/"><img src="https://img.shields.io/pypi/v/open-mlpipe.svg?style=flat-square&color=6366f1" alt="PyPI"></a>
    <a href="https://pypi.org/project/open-mlpipe/"><img src="https://img.shields.io/pypi/pyversions/open-mlpipe.svg?style=flat-square" alt="Python"></a>
    <a href="https://github.com/loisekk/open-mlpipe"><img src="https://img.shields.io/github/stars/loisekk/open-mlpipe?style=flat-square&color=f59e0b" alt="GitHub Stars"></a>
    <a href="https://pypi.org/project/open-mlpipe/"><img src="https://img.shields.io/pypi/dm/open-mlpipe?style=flat-square" alt="Downloads"></a>
    <a href="https://github.com/loisekk/open-mlpipe/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/open-mlpipe?style=flat-square&color=22c55e" alt="License"></a>
  </div>
  
  <div class="hero-buttons">
    <a href="getting-started/quickstart/" class="btn-primary">🚀 Get Started</a>
    <a href="https://github.com/loisekk/open-mlpipe" class="btn-secondary">⭐ Star on GitHub</a>
  </div>
</div>

<div class="stats-bar">
  <div class="stat-item">
    <div class="stat-value">14+</div>
    <div class="stat-label">ML Models</div>
  </div>
  <div class="stat-item">
    <div class="stat-value">12</div>
    <div class="stat-label">Pipeline Stages</div>
  </div>
  <div class="stat-item">
    <div class="stat-value">1</div>
    <div class="stat-label">Line of Code</div>
  </div>
  <div class="stat-item">
    <div class="stat-value">106</div>
    <div class="stat-label">Tests Passing</div>
  </div>
</div>

## Why open-mlpipe?

Every ML project starts the same way: load data, clean it, try 10 models, pick the best one. **Hours wasted. Every. Single. Time.**

open-mlpipe does it all in **one line of Python**:

<div class="code-showcase">
  <h2>One Line. Done.</h2>
  <div class="code-block-wrapper">
    <div class="code-block-header">
      <div class="code-block-dot red"></div>
      <div class="code-block-dot yellow"></div>
      <div class="code-block-dot green"></div>
    </div>
    <div class="code-block-content">
      <span class="keyword">from</span> <span class="function">open_mlpipe</span> <span class="keyword">import</span> run<br><br>
      <span class="comment"># One line — full pipeline</span><br>
      ctx = <span class="function">run</span>(<span class="string">"dataset.csv"</span>, target=<span class="string">"price"</span>)<br><br>
      <span class="comment"># Access results</span><br>
      <span class="keyword">print</span>(<span class="string">f"Best model: </span>{ctx.best_model_name}<span class="string">"</span>)<br>
      <span class="keyword">print</span>(<span class="string">f"Test R²: </span>{ctx.metrics[<span class="string">'test_r2'</span>]:.4f}<span class="string">"</span>)
    </div>
  </div>
</div>

<div class="feature-grid">
  <a href="guide/overview/" class="feature-card">
    <div class="icon">📊</div>
    <h3>Auto EDA</h3>
    <p>Column hygiene, quality assessment, distributions, statistical tests, VIF analysis — all automatic.</p>
  </a>
  
  <a href="guide/overview/" class="feature-card">
    <div class="icon">🔧</div>
    <h3>Feature Engineering</h3>
    <p>Missingness flags, interaction features, log transforms, datetime extraction — no manual work.</p>
  </a>
  
  <a href="concepts/supported-models/" class="feature-card">
    <div class="icon">🤖</div>
    <h3>14+ Models</h3>
    <p>Ridge, XGBoost, LightGBM, RandomForest, Stacking, Voting — auto-selected and compared.</p>
  </a>
  
  <a href="guide/overview/" class="feature-card">
    <div class="icon">🎯</div>
    <h3>Optuna Tuning</h3>
    <p>Hyperparameter optimization with 18 model-specific search spaces — finds the best parameters.</p>
  </a>
  
  <a href="guide/overview/" class="feature-card">
    <div class="icon">💡</div>
    <h3>SHAP Explainability</h3>
    <p>Global + local feature importance, dependence plots, waterfall explanations — understand your model.</p>
  </a>
  
  <a href="guide/overview/" class="feature-card">
    <div class="icon">💾</div>
    <h3>Production Ready</h3>
    <p>Full inference pipeline saved — raw data → prediction with no manual preprocessing.</p>
  </a>
</div>

<div class="quickstart">
  <h2>🚀 Get Started in 30 Seconds</h2>
  <div class="steps-grid">
    <div class="step-card">
      <div class="step-number">1</div>
      <h4>Install</h4>
      <p>pip install open-mlpipe</p>
    </div>
    <div class="step-card">
      <div class="step-number">2</div>
      <h4>Run</h4>
      <p>run("data.csv")</p>
    </div>
    <div class="step-card">
      <div class="step-number">3</div>
      <h4>Results</h4>
      <p>ctx.best_model_name</p>
    </div>
    <div class="step-card">
      <div class="step-number">4</div>
      <h4>Predict</h4>
      <p>model.predict(new)</p>
    </div>
  </div>
</div>

<div class="models-section">
  <h2>🤖 Supported Models</h2>
  <div class="models-grid">
    <div class="model-tag">Ridge</div>
    <div class="model-tag">Lasso</div>
    <div class="model-tag">ElasticNet</div>
    <div class="model-tag">DecisionTree</div>
    <div class="model-tag">RandomForest</div>
    <div class="model-tag">ExtraTrees</div>
    <div class="model-tag">XGBoost</div>
    <div class="model-tag">LightGBM</div>
    <div class="model-tag">GradientBoosting</div>
    <div class="model-tag">HistGradientBoosting</div>
    <div class="model-tag">AdaBoost</div>
    <div class="model-tag">KNN</div>
    <div class="model-tag">SVM</div>
    <div class="model-tag">Stacking</div>
    <div class="model-tag">Voting</div>
    <div class="model-tag">NaiveBayes</div>
  </div>
</div>

<div class="cta-section">
  <h2>Ready to Build?</h2>
  <p>Start building production ML pipelines in minutes, not hours.</p>
  <a href="getting-started/quickstart/">🚀 Get Started Now</a>
</div>

## Links

- [Getting Started](getting-started/installation.md) — Install and run your first pipeline
- [Python API](guide/python-api.md) — Complete API documentation
- [CLI Usage](guide/cli.md) — Command-line interface
- [YAML Config](guide/yaml-config.md) — Config-driven pipelines
- [API Reference](api/open_mlpipe.md) — Auto-generated from docstrings
- [GitHub](https://github.com/loisekk/open-mlpipe) — Source code and issues
