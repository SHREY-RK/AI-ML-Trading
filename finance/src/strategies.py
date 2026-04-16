import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

FEATURES = [
    'feat_ema9_dist',
    'feat_ema21_dist',
    'feat_ema50_dist',
    'feat_rsi',
    'feat_macd',
    'feat_macd_hist',
    'feat_bb_pos',
    'feat_bb_width',
    'feat_atr_pct',
    'feat_momentum_3',
    'feat_momentum_5',
    'feat_momentum_10',
    'feat_vol_ratio',
]


def build_features(df):
    """
    Adds normalised ML features to the DataFrame.
    All features are expressed as percentages or ratios so they are
    scale-invariant across different instruments and time periods.
    """
    df = df.copy()

    # Distance from EMAs (mean-reversion / trend signals)
    df['feat_ema9_dist'] = (df['close'] - df['ema9']) / df['ema9']
    df['feat_ema21_dist'] = (df['close'] - df['ema21']) / df['ema21']
    df['feat_ema50_dist'] = (df['close'] - df['ema50']) / df['ema50']

    # Oscillators
    df['feat_rsi'] = df['rsi'] / 100

    # MACD (normalised by price)
    df['feat_macd'] = df['macd'] / df['close']
    df['feat_macd_hist'] = df['macd_hist'] / df['close']

    # Bollinger Band position (0 = at lower band, 1 = at upper band)
    df['feat_bb_pos'] = df['bb_pos']
    df['feat_bb_width'] = df['bb_width']

    # ATR as a % of price (regime filter)
    df['feat_atr_pct'] = df['atr'] / df['close']

    # Momentum (pure price return over N bars)
    df['feat_momentum_3'] = df['close'].pct_change(3)
    df['feat_momentum_5'] = df['close'].pct_change(5)
    df['feat_momentum_10'] = df['close'].pct_change(10)

    # Volume ratio vs rolling average (is this bar unusual activity?)
    if 'volume' in df.columns:
        vol_mean = df['volume'].rolling(20).mean()
        # avoid division by zero when rolling mean is 0
        df['feat_vol_ratio'] = df['volume'] / vol_mean.replace(0, np.nan)
    else:
        df['feat_vol_ratio'] = 1.0

    # ── Sanitise all feature columns ──────────────────────────────────────
    # Replace inf/-inf with NaN first, then forward-fill, then back-fill,
    # then fill any remaining NaN with 0. This handles:
    #   • Division by zero (ema=0, bb_width=0, close=0)
    #   • Indicator warm-up period at the start of the series
    #   • Any volume spikes producing extreme ratios
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].ffill().bfill().fillna(0)

    # Hard-clip extreme values to ±10 (e.g. momentum during circuit breakers)
    # Keeps the scaler from being thrown off by rare outliers
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].clip(-10, 10)

    return df


# ─────────────────────────────────────────────
# LABEL GENERATION
# ─────────────────────────────────────────────

def build_labels(df, lookahead=10, profit_target=0.005):
    """
    Forward-return label.

    FIX: The label is built BEFORE the train/test split so that the future
    return never leaks into the model's feature set — we later drop the last
    `lookahead` rows from each training window.

    profit_target: minimum move (both ways) to generate a signal.
    Setting this above 0.1% ensures the expected win covers brokerage.
    """
    df = df.copy()
    df['future_return'] = (df['close'].shift(-lookahead) - df['close']) / df['close']
    conditions = [
        df['future_return'] > profit_target,
        df['future_return'] < -profit_target,
    ]
    df['target'] = np.select(conditions, [1, -1], default=0)
    return df


# ─────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

def walk_forward_signals(
    df,
    lookahead=10,
    profit_target=0.005,
    n_splits=5,
    confidence_threshold=0.55,
    model_type='rf',          # 'rf' or 'gb'
):
    """
    Walk-forward (anchored) cross-validation.

    Instead of a single 80/20 split, we train on expanding windows and
    predict on the next out-of-sample slice.  This mimics how a real live
    system would operate and avoids look-ahead bias from the naive split.

    Returns the original DataFrame with a 'signal' column populated only
    for the out-of-sample periods.
    """
    df = build_features(df)
    df = build_labels(df, lookahead=lookahead, profit_target=profit_target)
    df.dropna(subset=FEATURES + ['target'], inplace=True)
    df['signal'] = 0

    all_true = []
    all_pred = []

    n = len(df)
    min_train = int(n * 0.4)          # need at least 40% of data to start training
    fold_size = int((n - min_train) / n_splits)

    if fold_size < 50:
        print("⚠️  Not enough data for walk-forward. Falling back to single split.")
        return single_split_signals(df, confidence_threshold, model_type)

    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        train_df = df.iloc[:train_end - lookahead]   # drop last `lookahead` rows to prevent label leak
        test_df = df.iloc[test_start:test_end]

        if len(train_df) < 100 or len(test_df) < 10:
            continue

        # Final safety — drop any rows still containing NaN/inf at fold boundaries
        train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
        test_df  = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
        if len(train_df) < 50:
            print(f"  Fold {fold+1}: skipped — too few clean rows ({len(train_df)})")
            continue

        model = _build_model(model_type)
        model.fit(train_df[FEATURES], train_df['target'])

        probabilities = model.predict_proba(test_df[FEATURES])
        classes = list(model.classes_)

        for i, idx in enumerate(test_df.index):
            probs = probabilities[i]
            if 1 in classes:
                buy_prob = probs[classes.index(1)]
                if buy_prob > confidence_threshold:
                    df.loc[idx, 'signal'] = 1
                    continue
            if -1 in classes:
                sell_prob = probs[classes.index(-1)]
                if sell_prob > confidence_threshold:
                    df.loc[idx, 'signal'] = -1

        fold_true = test_df['target'].astype(int)
        fold_pred = df.loc[test_df.index, 'signal'].astype(int)
        all_true.extend(fold_true.tolist())
        all_pred.extend(fold_pred.tolist())

        fold_acc = accuracy_score(fold_true, fold_pred)
        fold_cm = confusion_matrix(fold_true, fold_pred, labels=[-1, 0, 1])

        print(f"  Fold {fold+1}/{n_splits} — train rows: {len(train_df)}, "
              f"test rows: {len(test_df)}, "
              f"buy signals: {(df.loc[test_df.index, 'signal'] == 1).sum()}, "
              f"sell signals: {(df.loc[test_df.index, 'signal'] == -1).sum()}")
        print(f"    Accuracy: {fold_acc:.3f}")
        print(f"    Confusion matrix ([-1,0,1]):\n{fold_cm}")

    if all_true:
        overall_acc = accuracy_score(all_true, all_pred)
        overall_cm = confusion_matrix(all_true, all_pred, labels=[-1, 0, 1])
        print(f"\n  Overall OOS accuracy: {overall_acc:.3f}")
        print(f"  Overall confusion matrix ([-1,0,1]):\n{overall_cm}")

    return df


def single_split_signals(df, confidence_threshold=0.55, model_type='rf'):
    """Fallback when data is too short for walk-forward."""
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    model = _build_model(model_type)
    model.fit(train_df[FEATURES], train_df['target'])

    probabilities = model.predict_proba(test_df[FEATURES])
    classes = list(model.classes_)

    df['signal'] = 0
    for i, idx in enumerate(test_df.index):
        probs = probabilities[i]
        if 1 in classes and probs[classes.index(1)] > confidence_threshold:
            df.loc[idx, 'signal'] = 1
        elif -1 in classes and probs[classes.index(-1)] > confidence_threshold:
            df.loc[idx, 'signal'] = -1

    y_true = test_df['target'].astype(int)
    y_pred = df.loc[test_df.index, 'signal'].astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

    print(f"\n  Single split accuracy: {acc:.3f}")
    print(f"  Single split confusion matrix ([-1,0,1]):\n{cm}")

    return df


def _build_model(model_type='rf'):
    """
    Returns a sklearn Pipeline with scaling + classifier.
    StandardScaler ensures gradient boosting (and in general) works well
    regardless of feature magnitudes.
    """
    if model_type == 'gb':
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            max_features='sqrt',
            class_weight='balanced',   # handles class imbalance (rare signals)
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf),
    ])


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def generate_signals(
    df,
    lookahead=10,
    profit_target=0.005,
    confidence_threshold=0.55,
    n_splits=5,
    model_type='rf',
):
    """
    Main entry point — drop-in replacement for the old generate_signals().

    Parameters
    ----------
    df                   : DataFrame with OHLCV + all indicator columns
    lookahead            : bars ahead to measure future return
    profit_target        : minimum return (each direction) to label a trade
    confidence_threshold : minimum model probability to generate a signal
    n_splits             : walk-forward folds (set to 1 for simple split)
    model_type           : 'rf' (Random Forest) or 'gb' (Gradient Boosting)
    """
    print(f"\n🔍 Generating signals — model: {model_type.upper()}, "
          f"lookahead: {lookahead}, target: {profit_target*100:.2f}%")

    df = walk_forward_signals(
        df,
        lookahead=lookahead,
        profit_target=profit_target,
        n_splits=n_splits,
        confidence_threshold=confidence_threshold,
        model_type=model_type,
    )

    total_signals = (df['signal'] != 0).sum()
    buy_signals = (df['signal'] == 1).sum()
    sell_signals = (df['signal'] == -1).sum()
    print(f"\n✅ Signal generation complete:")
    print(f"   Total signals : {total_signals}")
    print(f"   Buy  (long)   : {buy_signals}")
    print(f"   Sell (exit)   : {sell_signals}")

    return df