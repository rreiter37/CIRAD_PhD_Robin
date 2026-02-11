### script to compare the results of the tabpfn_best_preproc pipeline with the assoc pipeline et the other tabpfn wk

python scripts/Benchmark_tabpfn/compare_assoc_tabpfn_preproc.py \
  --tabpfn_workspaces wk_tabpfn_raw wk_tabpfn_rff \
  --tabpfn_labels "TabPFN raw" "TabPFN RFF" \
  --search_best_csvs Results/tabpfn_search_preproc_basic/best_tabpfn_per_dataset.csv \
  --search_best_labels "TabPFN best preproc" \
  --outdir Results/comp_assoc_tabpfn_mod7 \
  --strict_intersection

python scripts/Benchmark_tabpfn/compare_assoc_tabpfn_preproc.py \
  --search_best_csvs Results/tabpfn_search_preproc_basic/best_tabpfn_per_dataset.csv \
  --search_best_labels "TabPFN best preproc" \
  --outdir Results/comp_assoc_tabpfn_mod7 \
  --strict_intersection



### script to visualize the best preprocessings found across all datasets

python scripts/Benchmark_tabpfn/visualize_tabpfn_best_preproc.py \
  --csv Results/tabpfn_search_preproc_basic/best_tabpfn_per_dataset.csv \
  --outdir Results/tabpfn_best_preproc_viz \
  --metric final_test_nrmse



### script to get results on classif datasets with base models

python scripts/Benchmark_tabpfn/run_baselines_classif_on_all.py \
  --output_dir Results/baselines_classif \
  --catboost_iterations 200

### script to visualize the results for classification across models

python scripts/Benchmark_tabpfn/plot_classif_accuracy_heatmaps.py \
  --results_dirs Results/tabpfn_classif_raw Results/baselines_classif \
  --labels "TabPFN raw" "Baselines" \
  --keep_common_only \
  --output_dir Results/comp_classif_tabpfn_vs_baselines


### Script to rank the preprocessing families based on tabpfn results across datasets
python scripts/Benchmark_tabpfn/rank_simple_preproc_stage1.py \
  --csv_path Results/tabpfn_search_preproc_basic/tabpfn_search_results.csv \
  --out_dir Results/rank_simple_preproc_tabpfn_stage1 \
  --metric val_nrmse \
  --max_methods_plot 25 \
  --alpha 0.05



### Script to run the pipeline with pls and ridge on the regression datasets

python scripts/Benchmark_tabpfn/pipeline_pls_ridge.py \
  --database_detail_xlsx Data/DatabaseDetail.xlsx \
  --data_root Data/regression \
  --workspace wk_pls_ridge \
  --do_final_refit


### Script to run the final TabPFN pipeline on regression datasets

python scripts/Benchmark_tabpfn/run_tabpfn_final.py \
  --database_detail_xlsx Data/DatabaseDetail.xlsx \
  --data_root Data/regression \
  --output_dir Results/tabpfn_reg_final \
  --logs_dir Results/tabpfn_reg_final/logs \
  --summary_csv Results/tabpfn_reg_final/summary_runs.csv \
  --model_path "tabpfn-v2.5-regressor-v2.5_real.ckpt" \
  --n_estimators_search 4 \
  --n_estimators_final 4 \
  --parallel --n_jobs 12 --use_tmp_dir
