"""
CUSTOMER SEGMENTATION - Unsupervised Machine Learning project
Algorithms: K-Means, Hierarchical (Agglomerative, Ward), DBSCAN
Everything lives in this single file: loading, preprocessing, training,
evaluation, plotting and the terminal menu.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
RANDOM_STATE = 42
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
ID_COLUMN = "CustomerID"
X_COL, Y_COL = "Annual Income (k$)", "Spending Score (1-100)"  # plot axes
CSV_NAMES = ["Mall_Customers.csv", "Customer.csv"]  # first one found is used
K_RANGE = range(2, 11)                # cluster counts to test (2..10)
DBSCAN_MIN_SAMPLES = [3, 4, 5, 6, 8, 10]
DBSCAN_MAX_NOISE = 0.20               # reject configs with >20% noise
DBSCAN_MAX_BIGGEST = 0.80             # reject if one cluster holds >80% of points
DBSCAN_MIN_CLUSTER_SIZE = 5           # reject configs with tiny "clusters"
ALGORITHMS = ["K-Means", "Hierarchical", "DBSCAN"]


# ----------------------------------------------------------------------
# 1. Loading and preprocessing
# ----------------------------------------------------------------------
def load_data():
    """Load the CSV (next to this script), validate it and clean the features."""
    script_folder = Path(__file__).resolve().parent
    csv_path = next((script_folder / n for n in CSV_NAMES
                     if (script_folder / n).exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"Dataset not found. Put '{CSV_NAMES[0]}' in: {script_folder}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    missing = [c for c in [ID_COLUMN] + FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Turn text like "abc" into NaN instead of crashing, then drop bad rows.
    for col in [ID_COLUMN] + FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[ID_COLUMN] + FEATURES)
    df = df[(df[FEATURES] >= 0).all(axis=1)]          # negative values are invalid
    df = df.drop_duplicates(subset=ID_COLUMN).reset_index(drop=True)
    df[ID_COLUMN] = df[ID_COLUMN].astype(int)
    if before != len(df):
        print(f"Note: removed {before - len(df)} invalid/duplicate row(s).")

    if len(df) < 12:
        raise ValueError("Not enough valid rows to cluster (need at least 12).")
    return df


def scale_features(df):
    """Scale the three features. The original df is NOT modified."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    return X_scaled, scaler


# ----------------------------------------------------------------------
# 2. Metrics helper
# ----------------------------------------------------------------------
def compute_metrics(X, labels):
    """Silhouette / Davies-Bouldin / Calinski-Harabasz on NON-NOISE points only.
    Returns NaN for metrics that cannot be computed (fewer than 2 clusters)."""
    labels = np.asarray(labels)
    mask = labels != -1                       # DBSCAN noise is labeled -1
    n_noise = int((~mask).sum())
    out = {"silhouette": np.nan, "davies_bouldin": np.nan,
           "calinski_harabasz": np.nan,
           "n_clusters": len(set(labels[mask])),
           "n_noise": n_noise, "noise_pct": 100 * n_noise / len(labels),
           "n_used": int(mask.sum())}
    n_used = int(mask.sum())
    if 2 <= out["n_clusters"] <= n_used - 1:
        out["silhouette"] = silhouette_score(X[mask], labels[mask])
        out["davies_bouldin"] = davies_bouldin_score(X[mask], labels[mask])
        out["calinski_harabasz"] = calinski_harabasz_score(X[mask], labels[mask])
    return out


# ----------------------------------------------------------------------
# 3. The three algorithms
# ----------------------------------------------------------------------
def run_kmeans(X, scaler):
    max_k = min(max(K_RANGE), len(X) - 1)
    ks = [k for k in K_RANGE if k <= max_k]
    inertias, silhouettes = [], []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    best_k = ks[int(np.argmax(silhouettes))]       # highest silhouette
    final = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = final.fit_predict(X)
    # Centers are in scaled units -> convert back to original units for plotting
    centers_original = scaler.inverse_transform(final.cluster_centers_)
    return {"model": final, "labels": labels, "k": best_k, "ks": ks,
            "inertias": inertias, "silhouettes": silhouettes,
            "centers_original": centers_original,
            "params": f"k={best_k}"}


def run_hierarchical(X):
    max_k = min(max(K_RANGE), len(X) - 1)
    ks = [k for k in K_RANGE if k <= max_k]
    silhouettes = []
    for k in ks:
        # Ward linkage always uses Euclidean distance
        model = AgglomerativeClustering(n_clusters=k, linkage="ward",
                                        metric="euclidean")
        silhouettes.append(silhouette_score(X, model.fit_predict(X)))

    best_k = ks[int(np.argmax(silhouettes))]
    final = AgglomerativeClustering(n_clusters=best_k, linkage="ward",
                                    metric="euclidean")
    labels = final.fit_predict(X)
    return {"model": final, "labels": labels, "k": best_k, "ks": ks,
            "silhouettes": silhouettes, "params": f"k={best_k}, ward"}


def k_distances(X, k):
    """Sorted distance from every point to its k-th nearest point (incl. itself
    as the 1st). This is the curve used to pick eps for DBSCAN."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    return np.sort(distances[:, -1])


def run_dbscan(X):
    n = len(X)
    kdist_curves = {}
    tried = []                 # every configuration we evaluated
    best = None

    for min_samples in DBSCAN_MIN_SAMPLES:
        if min_samples >= n:
            continue
        kdist = k_distances(X, min_samples)
        kdist_curves[min_samples] = kdist
        # Candidate eps values come from the k-distance curve itself:
        # from its median up to its 98th percentile.
        eps_values = np.unique(np.round(
            np.linspace(np.percentile(kdist, 50), np.percentile(kdist, 98), 15), 3))
        for eps in eps_values:
            if eps <= 0:
                continue
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            m = compute_metrics(X, labels)
            sizes = np.bincount(labels[labels != -1]) if m["n_used"] else np.array([])
            biggest = sizes.max() / n if len(sizes) else 1.0
            smallest = sizes.min() if len(sizes) else 0
            valid = (m["n_clusters"] >= 2
                     and m["noise_pct"] / 100 <= DBSCAN_MAX_NOISE
                     and biggest <= DBSCAN_MAX_BIGGEST
                     and smallest >= DBSCAN_MIN_CLUSTER_SIZE
                     and not np.isnan(m["silhouette"]))
            tried.append({"eps": float(eps), "min_samples": min_samples,
                          "valid": valid, **m})
            if valid and (best is None or m["silhouette"] > best["silhouette"]):
                best = {"eps": float(eps), "min_samples": min_samples,
                        "labels": labels, "silhouette": m["silhouette"]}

    if best is None:
        raise ValueError(
            "No valid DBSCAN configuration found (every tested setting gave "
            "<2 clusters, too much noise, or one giant cluster).")
    return {"labels": best["labels"], "eps": best["eps"],
            "min_samples": best["min_samples"], "tried": tried,
            "kdist_curves": kdist_curves,
            "params": f"eps={best['eps']}, min_samples={best['min_samples']}"}


def train_all(df, X, scaler):
    """Train every algorithm once. One failure does not stop the others."""
    results = {}
    runners = {
        "K-Means": lambda: run_kmeans(X, scaler),
        "Hierarchical": lambda: run_hierarchical(X),
        "DBSCAN": lambda: run_dbscan(X),
    }
    for name, runner in runners.items():
        try:
            res = runner()
            res["metrics"] = compute_metrics(X, res["labels"])
            results[name] = res
        except Exception as error:               # keep going with the others
            print(f"  [!] {name} failed: {error}")
            results[name] = None
    return results


# ----------------------------------------------------------------------
# 4. Reporting: comparison table, profiles, insights
# ----------------------------------------------------------------------
def fmt(value, digits=3):
    return "n/a" if value is None or np.isnan(value) else f"{value:.{digits}f}"


def print_comparison(results):
    print("\n" + "=" * 98)
    print("MODEL COMPARISON")
    print("=" * 98)
    header = (f"{'Algorithm':<14}{'Clusters':>9}{'Silhouette':>12}{'Davies-B.':>11}"
              f"{'Calinski-H.':>13}{'Noise pts':>11}{'Noise %':>9}  Parameters")
    print(header)
    print("-" * 98)
    for name in ALGORITHMS:
        res = results.get(name)
        if res is None:
            print(f"{name:<14}  (failed - see message above)")
            continue
        m = res["metrics"]
        noise_pts = m["n_noise"] if name == "DBSCAN" else "-"
        noise_pct = f"{m['noise_pct']:.1f}%" if name == "DBSCAN" else "-"
        print(f"{name:<14}{m['n_clusters']:>9}{fmt(m['silhouette']):>12}"
              f"{fmt(m['davies_bouldin']):>11}{fmt(m['calinski_harabasz'], 1):>13}"
              f"{noise_pts:>11}{noise_pct:>9}  {res['params']}")
    print("-" * 98)
    print("Silhouette: higher = better | Davies-Bouldin: LOWER = better | "
          "Calinski-Harabasz: higher = better")
    db = results.get("DBSCAN")
    if db is not None:
        m = db["metrics"]
        print(f"DBSCAN metrics use only the {m['n_used']} non-noise points "
              f"({m['n_noise']} noise points excluded).")


def print_summary(results):
    print("\nSUMMARY")
    print("-" * 40)
    ok = {n: r for n, r in results.items() if r is not None}
    if not ok:
        print("No algorithm finished successfully.")
        return
    for name, res in ok.items():
        extra = (f", {res['metrics']['noise_pct']:.1f}% noise"
                 if name == "DBSCAN" else "")
        print(f"- {name}: {res['metrics']['n_clusters']} clusters ({res['params']}){extra}")

    def best(metric, higher_is_better):
        candidates = {n: r["metrics"][metric] for n, r in ok.items()
                      if not np.isnan(r["metrics"][metric])}
        if not candidates:
            return "n/a"
        pick = max if higher_is_better else min
        return pick(candidates, key=candidates.get)

    print(f"\nBest Silhouette:        {best('silhouette', True)}")
    print(f"Best Davies-Bouldin:    {best('davies_bouldin', False)}")
    print(f"Best Calinski-Harabasz: {best('calinski_harabasz', True)}")
    print("\nNo single metric decides the 'winner'. Compare the metrics together")
    print("with the plots and with how useful the groups are for marketing.")
    print("DBSCAN is density-based, so its output (and its noise points) can")
    print("differ structurally from K-Means and Hierarchical clustering.")


def level(value, mean, std):
    """Turn a number into High / Medium / Low relative to the whole dataset."""
    if value >= mean + 0.5 * std:
        return "High"
    if value <= mean - 0.5 * std:
        return "Low"
    return "Medium"


def describe_cluster(avg_age, avg_income, avg_spend, df):
    """Data-driven description. Never uses the cluster ID."""
    income_lvl = level(avg_income, df[X_COL].mean(), df[X_COL].std())
    spend_lvl = level(avg_spend, df[Y_COL].mean(), df[Y_COL].std())
    age_part = ""
    if avg_age <= df["Age"].mean() - 0.5 * df["Age"].std():
        age_part = ", younger"
    elif avg_age >= df["Age"].mean() + 0.5 * df["Age"].std():
        age_part = ", older"
    text = f"{income_lvl}-income, {spend_lvl}-spending{age_part}"
    tips = {("High", "High"): " -> premium / loyalty target",
            ("High", "Low"): " -> under-engaged; try personalised offers",
            ("Low", "High"): " -> enthusiastic spenders; value deals",
            ("Low", "Low"): " -> low priority / budget-focused"}
    return text + tips.get((income_lvl, spend_lvl), "")


def print_profiles(df, results):
    for name in ALGORITHMS:
        res = results.get(name)
        print("\n" + "=" * 98)
        print(f"{name.upper()} - CLUSTER PROFILES (original, unscaled values)")
        print("=" * 98)
        if res is None:
            print("Not available (algorithm failed).")
            continue
        labels = pd.Series(res["labels"], index=df.index)
        print(f"{'Cluster':<9}{'Customers':>10}{'Share':>8}{'Avg Age':>9}"
              f"{'Avg Income':>12}{'Avg Spend':>11}   Interpretation")
        for label in sorted(l for l in labels.unique() if l != -1):
            part = df[labels == label]
            print(f"{label:<9}{len(part):>10}{100 * len(part) / len(df):>7.1f}%"
                  f"{part['Age'].mean():>9.1f}{part[X_COL].mean():>12.1f}"
                  f"{part[Y_COL].mean():>11.1f}   "
                  f"{describe_cluster(part['Age'].mean(), part[X_COL].mean(), part[Y_COL].mean(), df)}")
        noise = df[labels == -1]
        if name == "DBSCAN":
            if len(noise):
                print(f"{'Noise':<9}{len(noise):>10}{100 * len(noise) / len(df):>7.1f}%"
                      f"{noise['Age'].mean():>9.1f}{noise[X_COL].mean():>12.1f}"
                      f"{noise[Y_COL].mean():>11.1f}   Outliers (not assigned to any cluster)")
            else:
                print("Noise: 0 customers")
    print("\nNote: cluster numbers are arbitrary and can change between algorithms;")
    print("the interpretation column is computed from each cluster's averages.")


# ----------------------------------------------------------------------
# 5. Plotting
# ----------------------------------------------------------------------
def axis_limits(df):
    """Same axis limits for all three plots so they can be compared visually."""
    x_pad = 0.05 * (df[X_COL].max() - df[X_COL].min())
    y_pad = 0.05 * (df[Y_COL].max() - df[Y_COL].min())
    return ((df[X_COL].min() - x_pad, df[X_COL].max() + x_pad),
            (df[Y_COL].min() - y_pad, df[Y_COL].max() + y_pad))


def make_cluster_figure(df, name, results):
    """One figure for one algorithm, using that algorithm's real labels."""
    res = results.get(name)
    if res is None:
        print(f"{name}: no plot available (algorithm failed).")
        return None
    labels = res["labels"]
    colors = plt.get_cmap("tab10")
    xlim, ylim = axis_limits(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in sorted(l for l in set(labels) if l != -1):
        mask = labels == label
        ax.scatter(df.loc[mask, X_COL], df.loc[mask, Y_COL], s=45,
                   color=colors(label % 10), edgecolor="white", linewidth=0.5,
                   label=f"Cluster {label} (n={mask.sum()})")
    if name == "DBSCAN":
        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(df.loc[noise_mask, X_COL], df.loc[noise_mask, Y_COL],
                       marker="x", s=55, color="black",
                       label=f"Noise (n={noise_mask.sum()})")
    if name == "K-Means":
        centers = res["centers_original"]          # columns follow FEATURES
        ax.scatter(centers[:, FEATURES.index(X_COL)], centers[:, FEATURES.index(Y_COL)],
                   marker="*", s=350, color="gold", edgecolor="black",
                   linewidth=1.2, label="Cluster centers", zorder=5)

    ax.set_title(f"{name} Clustering ({res['params']})")
    ax.set_xlabel(X_COL)
    ax.set_ylabel(Y_COL)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.text(0.5, 0.01, "Only Income and Spending Score are shown; "
             "Age was also used to train the model.", ha="center", fontsize=8,
             style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def show_cluster_plots(df, results, names):
    figures = [make_cluster_figure(df, n, results) for n in names]
    if any(f is not None for f in figures):
        plt.show()                                  # blocks until windows close
    plt.close("all")


def make_elbow_silhouette_figure(results):
    km, hc = results.get("K-Means"), results.get("Hierarchical")
    if km is None and hc is None:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if km is not None:
        ax1.plot(km["ks"], km["inertias"], "o-", color="tab:blue")
        ax1.axvline(km["k"], color="red", linestyle="--", label=f"Selected k={km['k']}")
        ax1.set_title("K-Means Elbow Method")
        ax1.set_xlabel("Number of clusters (k)")
        ax1.set_ylabel("Inertia (within-cluster sum of squares)")
        ax1.legend()
        ax2.plot(km["ks"], km["silhouettes"], "o-", label="K-Means")
    else:
        ax1.text(0.5, 0.5, "K-Means unavailable", ha="center")
    if hc is not None:
        ax2.plot(hc["ks"], hc["silhouettes"], "s--", label="Hierarchical")
    ax2.set_title("Silhouette Score vs Number of Clusters")
    ax2.set_xlabel("Number of clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_kdistance_figure(results):
    db = results.get("DBSCAN")
    if db is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for min_samples, curve in db["kdist_curves"].items():
        selected = min_samples == db["min_samples"]
        ax.plot(curve, linewidth=2.5 if selected else 1,
                alpha=1 if selected else 0.5,
                label=f"min_samples={min_samples}" + (" (selected)" if selected else ""))
    ax.axhline(db["eps"], color="red", linestyle="--",
               label=f"Selected eps={db['eps']}")
    ax.set_title("DBSCAN k-distance Plot (look for the 'knee')")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel("Distance to k-th nearest neighbor (scaled units)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_dendrogram_figure(X, results):
    hc = results.get("Hierarchical")
    if hc is None:
        return None
    try:
        Z = linkage(X, method="ward")
        k = hc["k"]
        cut_height = (Z[-(k - 1), 2] + Z[-k, 2]) / 2 if k >= 2 else None
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, leaf_rotation=90,
                   leaf_font_size=8, show_contracted=True)
        if cut_height is not None:
            ax.axhline(cut_height, color="red", linestyle="--",
                       label=f"Cut for k={k}")
            ax.legend()
        ax.set_title("Hierarchical Clustering Dendrogram (Ward, last 30 merges)")
        ax.set_xlabel("Cluster size (or customer index)")
        ax.set_ylabel("Ward distance")
        fig.tight_layout()
        return fig
    except Exception as error:
        print(f"Could not create dendrogram: {error}")
        return None


def make_comparison_figure(results):
    ok = [n for n in ALGORITHMS if results.get(n) is not None]
    if not ok:
        return None
    metrics = [("silhouette", "Silhouette\n(higher is better)"),
               ("davies_bouldin", "Davies-Bouldin\n(lower is better)"),
               ("calinski_harabasz", "Calinski-Harabasz\n(higher is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = {"K-Means": "tab:blue", "Hierarchical": "tab:orange", "DBSCAN": "tab:green"}
    for ax, (key, title) in zip(axes, metrics):
        values = [results[n]["metrics"][key] for n in ok]
        plotted = [0 if np.isnan(v) else v for v in values]
        bars = ax.bar(ok, plotted, color=[colors[n] for n in ok])
        for bar, v in zip(bars, values):
            label = "n/a" if np.isnan(v) else f"{v:.2f}" if key != "calinski_harabasz" else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Algorithm Comparison (DBSCAN metrics exclude noise points)")
    fig.tight_layout()
    return fig


def show_evaluation_plots(X, results):
    figures = [make_elbow_silhouette_figure(results), make_kdistance_figure(results),
               make_dendrogram_figure(X, results), make_comparison_figure(results)]
    if any(f is not None for f in figures):
        plt.show()
    plt.close("all")


# ----------------------------------------------------------------------
# 6. Explore one customer
# ----------------------------------------------------------------------
def explore_customer(df, results):
    text = input("Enter a CustomerID: ").strip()
    try:
        customer_id = int(text)
    except ValueError:
        print("Please enter a whole number.")
        return
    row = df[df[ID_COLUMN] == customer_id]
    if row.empty:
        print(f"No customer with ID {customer_id} "
              f"(valid IDs: {df[ID_COLUMN].min()} to {df[ID_COLUMN].max()}).")
        return
    index = row.index[0]
    customer = df.loc[index]
    print(f"\nCustomer {customer_id}")
    if "Gender" in df.columns:
        print(f"  Gender:         {customer['Gender']}")
    for feature in FEATURES:
        print(f"  {feature + ':':<26}{customer[feature]}")
    print("  Cluster assignments:")
    for name in ALGORITHMS:
        res = results.get(name)
        if res is None:
            print(f"    {name:<13} unavailable")
            continue
        label = int(res["labels"][index])
        print(f"    {name:<13} " + ("Noise (outlier)" if label == -1 else f"Cluster {label}"))


# ----------------------------------------------------------------------
# 7. Menu
# ----------------------------------------------------------------------
MENU = """
==================== MENU ====================
1. View model comparison
2. View K-Means clustering plot
3. View Hierarchical clustering plot
4. View DBSCAN clustering plot
5. View all three clustering plots
6. View cluster profiles and business insights
7. View evaluation and parameter-selection plots
8. Explore a customer by CustomerID
9. Exit
=============================================="""


def main():
    print("=" * 40)
    print("      CUSTOMER SEGMENTATION")
    print("=" * 40)
    try:
        df = load_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as error:
        print(f"Error: {error}")
        return

    print(f"Customers: {len(df)}")
    print("Features used: " + ", ".join(FEATURES))
    print("(CustomerID and Gender are not used for clustering.)")

    X, scaler = scale_features(df)
    print("\nTraining K-Means, Hierarchical and DBSCAN ...")
    results = train_all(df, X, scaler)
    if all(r is None for r in results.values()):
        print("All algorithms failed. Nothing to show.")
        return

    print_comparison(results)
    print_summary(results)

    actions = {
        "1": lambda: (print_comparison(results), print_summary(results)),
        "2": lambda: show_cluster_plots(df, results, ["K-Means"]),
        "3": lambda: show_cluster_plots(df, results, ["Hierarchical"]),
        "4": lambda: show_cluster_plots(df, results, ["DBSCAN"]),
        "5": lambda: show_cluster_plots(df, results, ALGORITHMS),
        "6": lambda: print_profiles(df, results),
        "7": lambda: show_evaluation_plots(X, results),
        "8": lambda: explore_customer(df, results),
    }
    while True:
        print(MENU)
        try:
            choice = input("Choose an option (1-9): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if choice == "9":
            print("Goodbye!")
            break
        action = actions.get(choice)
        if action is None:
            print("Invalid choice. Please type a number from 1 to 9.")
            continue
        try:
            action()
        except Exception as error:     # never let one bad action kill the menu
            print(f"Something went wrong: {error}")


if __name__ == "__main__":
    main()
