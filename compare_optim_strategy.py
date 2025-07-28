import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to timing and result directories
timing_base_dir = os.path.join("Figures", "assoc_pp_model")
results_base_dir = os.path.join("Results", "assoc_pp_model")

# Stocker les résultats cumulés
timing_data = []
performance_data = []

# Récupérer tous les sous-dossiers de jeux de données
for dataset in os.listdir(timing_base_dir):
    timing_path = os.path.join(timing_base_dir, dataset, "timing_results.csv")
    result_dir = os.path.join(results_base_dir, dataset)
    
    if not os.path.exists(timing_path):
        continue
    
    # Charger les temps de calcul
    df_time = pd.read_csv(timing_path)
    df_time["dataset"] = dataset
    timing_data.append(df_time)
    
    # Essayer de trouver les fichiers de performances correspondants
    for fname in os.listdir(result_dir):
        if fname.startswith("results_") and fname.endswith(".csv"):
            df_perf = pd.read_csv(os.path.join(result_dir, fname), index_col=0)
            mean_score = df_perf.min().mean() if "RMSE" in fname else df_perf.max().mean()
            performance_data.append({
                "filename": fname,
                "dataset": dataset,
                "optimization_type": "progressive" if "progressive" in fname else "uniform",
                "mean_score": mean_score
            })

# Fusionner tous les résultats dans des DataFrames
df_timing = pd.concat(timing_data, ignore_index=True)
df_perf = pd.DataFrame(performance_data)

# Fusionner les deux (temps + perf)
df_merge = pd.merge(df_timing, df_perf, on=["dataset", "optimization_type"], how="left")

# Affichage statistique
summary = df_merge.groupby("optimization_type").agg({
    "time": ["mean", "std"],
    "mean_score": ["mean", "std"]
}).round(2)

print("\nRésumé des performances et des temps selon l'optimisation :\n")
print(summary)

# ──────────────────────────────
# Plotting the execution time
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_merge, x="optimization_type", y="time", palette="Set2")
plt.title("Temps d'exécution par stratégie d'optimisation")
plt.ylabel("Temps (secondes)")
plt.xlabel("Type d'optimisation")
plt.tight_layout()
plt.savefig("comparaison_temps_execution.png", dpi=300)
plt.show()

# ──────────────────────────────
# Facultatif : export CSV combiné
df_merge.to_csv("comparaison_optimisation.csv", index=False)
print("\n[INFO] Fichier exporté : comparaison_optimisation.csv")
