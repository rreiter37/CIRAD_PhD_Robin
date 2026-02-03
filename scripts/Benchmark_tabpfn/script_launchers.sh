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