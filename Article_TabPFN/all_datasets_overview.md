% ---------- Packages needed in preamble ----------
% \usepackage{longtable}
% \usepackage{booktabs}
% \usepackage{array}
% \usepackage{makecell}
% \usepackage{multirow}
% \usepackage{ragged2e}
% \usepackage{pdflscape}
% \usepackage{hyperref}
% \usepackage{xurl}
%
% Column types:
% \newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
% \newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\begingroup
\footnotesize
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.15}
\renewcommand{\cellalign}{tl}
\renewcommand{\theadalign}{tl}
\setlength{\LTpre}{4pt}
\setlength{\LTpost}{4pt}
\sloppy
\Urlmuskip=0mu plus 2mu
\urlstyle{same}

% ==================================================
% REGRESSION
% ==================================================


\begin{longtable}{L{3.5cm}L{7.9cm}L{2.7cm}L{1.7cm}L{1.95cm}L{1.05cm}R{0.70cm}R{0.70cm}R{0.70cm}R{0.70cm}R{0.85cm}R{0.70cm}}
\caption{\textbf{Regression.} Benchmark overview for regression datasets. Source URLs of the open access databases are given. \textit{p} stands for the number of variables in the dataset, \textit{Outl. test} for Outliers percentage test, \textit{Ext. test} for Extrapolation test. Outlier percentages are based on Hotelling $T^2$ detection on the test set, and extrapolation counts denote the number of test targets outside the support of the train set.}
\label{tab:dataset_description_long_regression} \\
\toprule
\makecell[l]{\bfseries Database\\ } &
\makecell[l]{\bfseries Dataset\\name} &
\makecell[l]{\bfseries Source\\URL} &
\makecell[l]{\bfseries Sample\\type} &
\makecell[l]{\bfseries Target\\trait} &
\makecell[l]{\bfseries Split\\type} &
\makecell[r]{\bfseries N\\total} &
\makecell[r]{\bfseries N\\train} &
\makecell[r]{\bfseries N\\test} &
\makecell[r]{\bfseries p} &
\makecell[r]{\bfseries Outl.\\test} &
\makecell[r]{\bfseries Ext.\\test} \\
\midrule
\endfirsthead

\toprule
\makecell[l]{\bfseries Database\\ } &
\makecell[l]{\bfseries Dataset\\name} &
\makecell[l]{\bfseries Source\\URL} &
\makecell[l]{\bfseries Sample\\type} &
\makecell[l]{\bfseries Target\\trait} &
\makecell[l]{\bfseries Split\\type} &
\makecell[r]{\bfseries N\\total} &
\makecell[r]{\bfseries N\\train} &
\makecell[r]{\bfseries N\\test} &
\makecell[r]{\bfseries p} &
\makecell[r]{\bfseries Outl.\\test} &
\makecell[r]{\bfseries Ext.\\test} \\
\midrule
\endhead

\midrule
\multicolumn{12}{r}{Continued on next page} \\
\midrule
\endfoot

\bottomrule
\endlastfoot

ALPINE
& ALPINE\_P
& \url{https://doi.org/10.18710/CXRCUW}
& Ground dried leaves & P & KS
& 291 & 247 & 44 & 2151 & - & 1 \\
\midrule
AMYLOSE
& Rice\_Amylose
& \url{https://doi.org/10.1016/j.dib.2017.09.077}
& Rice Flour & Amylose content & Y sorted
& 313 & 203 & 110 & 1154 & 3.6\% & 0 \\
\midrule
BEEFMARBLING
& Beef\_Marbling
& \url{https://doi.org/10.57745/FRDOJC}
& Fresh beef carcasse muscle & Marbling & Random
& 832 & 554 & 278 & 331 & 7.9\% & 0 \\
\midrule
\multirow{2}{*}{BEER}
& Beer\_OriginalExtract\_KS
& \url{https://github.com/nanxstats/OHPL/raw/master/data/beer.rda}
& Beer & Original extract & KS
& 60 & 40 & 20 & 576 & - & 0 \\
& Beer\_OriginalExtract\_YBasedSplit
&
& Beer & Original extract & Y sorted
& 60 & 40 & 20 & 576 & - & 0 \\
\midrule
\multirow{3}{*}{BERRY}
& Berry\_Brix
& \url{https://github.com/WongCYS/grapevine_RMI_2025}
& Fresh leaf & Winegrape berry brix & Stratified
& 2133 & 1434 & 699 & 2101 & 10.2\% & 2 \\
& Berry\_pH
&
& Fresh leaf & Winegrape berry pH & Stratified
& 1401 & 912 & 489 & 2101 & 18.4\% & 0 \\
& Berry\_TartaricAcid
&
& Fresh leaf & Winegrape berry tartaric acid content & Stratified
& 1401 & 912 & 489 & 2101 & 18.4\% & 0 \\
\midrule
\multirow{2}{*}{BISCUIT}
& Biscuit\_Fat
& \url{https://rdrr.io/cran/fds/man/Biscuit.html}
& Biscuit dough & Fat content & Random
& 72 & 40 & 32 & 700 & - & 2 \\
& Biscuit\_Sucrose
&
& Biscuit dough & Sucrose content & Random
& 72 & 40 & 32 & 700 & - & 1 \\
\midrule
\multirow{3}{*}{COLZA}
& Colza\_C
& \url{https://doi.org/10.57745/6VYUQN}
& Oilseed rape plant tissues & C content & not specified
& 2419 & 1210 & 1209 & 1154 & 2.6\% & 1 \\
& Colza\_N\_wOutlier
&
& Oilseed rape plant tissues & N content & not specified
& 2427 & 1220 & 1207 & 1154 & 2.9\% & 0 \\
& Colza\_N\_woOutlier
&
& Oilseed rape plant tissues & N content & not specified
& 2412 & 1205 & 1207 & 1154 & 3.1\% & 0 \\
\midrule
\multirow{2}{*}{CORN}
& Corn\_Oil
& \url{https://eigenvector.com/resources/data-sets/}
& Corn kernel & Oil content & Y sorted
& 80 & 64 & 16 & 700 & - & 0 \\
& Corn\_Starch
&
& Corn kernel & Starch content & Y sorted
& 80 & 64 & 16 & 700 & 6.2\% & 0 \\
\midrule
\multirow{3}{*}{DIESEL}
& Diesel\_bp50\_b-a
& \url{https://eigenvector.com/resources/data-sets/}
& Diesel fuel & Boiling point at 50\% recovery & not specified
& 226 & 113 & 113 & 401 & 7.1\% & 0 \\
& Diesel\_bp50\_hla-b
&
& Diesel fuel & Boiling point at 50\% recovery & not specified
& 246 & 133 & 113 & 401 & 8.8\% & 2 \\
& Diesel\_bp50\_hlb-a
&
& Diesel fuel & Boiling point at 50\% recovery & not specified
& 246 & 133 & 113 & 401 & 5.3\% & 0 \\
\midrule
\multirow{4}{*}{DarkResp}
& DarkResp\_SiteCB
& \url{https://doi.org/10.1111/nph.20267}
& Forest tree fresh leaf & Dark respiration & Site
& 470 & 324 & 146 & 2151 & 4.1\% & 1 \\
& DarkResp\_SiteGT
&
& Forest tree fresh leaf & Dark respiration & Site
& 470 & 297 & 173 & 2151 & 86.7\% & 1 \\
& DarkResp\_SiteXSBN
&
& Forest tree fresh leaf & Dark respiration & Site
& 470 & 319 & 151 & 2151 & 11.9\% & 0 \\
& DarkResp\_spxy
&
& Forest tree fresh leaf & Dark respiration & spxy
& 470 & 329 & 141 & 2151 & 1.4\% & 0 \\
\midrule
\multirow{3}{*}{ECOSIS\_LeafTraits}
& EcosisLeaf\_Carotenoid
& \url{https://github.com/UW-GCRL/PLSR_trait_models_evaluation}
& Fresh leaf & Total carotenoid content & spatial
& 4245 & 1016 & 3229 & 196 & 9.0\% & 0 \\
& EcosisLeaf\_Chlorophyll\_SpatialSplit
&
& Fresh leaf & Chla + b & spatial
& 6850 & 2925 & 3925 & 196 & 8.5\% & 0 \\
& EcosisLeaf\_Chlorophyll\_SpeciesSplit
&
& Fresh leaf & Chla + b & species
& 6850 & 3734 & 3116 & 196 & 14.0\% & 0 \\
\midrule
FUSARIUM
& Fusarium\_FvFm
& \url{https://doi.org/10.5281/zenodo.16217833}
& Fresh leaf & Photochemical potential (Fv/Fm) & Group stratified
& 518 & 351 & 167 & 2177 & 10.2\% & 1 \\
\midrule
GRAPEVINES
& Grapevines\_Chloride
& \url{https://github.com/diazgarcialab/grapevine-chloride-prediction}
& Fresh leaf & Leaf chloride content & ks
& 555 & 388 & 167 & 1023 & - & 0 \\
\midrule
\multirow{6}{*}{GRAPEVINE\_LeafTraits}
& GrapevineLeaf\_NetCO2\_ASD
& \url{https://doi.org/10.57745/WVAPOL}
& Dried leaf & Net CO2 assimilation & spxy
& 112 & 78 & 34 & 2101 & 11.8\% & 2 \\
& GrapevineLeaf\_NetCO2\_MicroNIR
&
& Fresh leaf & Net CO2 assimilation & spxy
& 116 & 81 & 35 & 125 & 5.7\% & 3 \\
& GrapevineLeaf\_NetCO2\_MicroNIR\_NeoSpectra
&
& Fresh leaf & Net CO2 assimilation & spxy
& 115 & 80 & 35 & 276 & 5.7\% & 3 \\
& GrapevineLeaf\_NetCO2\_NeoSpectra
&
& Fresh leaf & Net CO2 assimilation & spxy
& 119 & 82 & 37 & 257 & 8.1\% & 4 \\
& GrapevineLeaf\_LMA
&
& Dried leaf & Leaf mass per area & spxy
& 1564 & 1092 & 472 & 2101 & 9.1\% & 2 \\
& GrapevineLeaf\_WUE
&
& Fresh leaf & Water use efficiency & spxy
& 112 & 77 & 35 & 276 & 2.9\% & 0 \\
\midrule
IncombustibleMaterial
& Incombustible\_TIC
& \url{https://github.com/nevernervous78/nirpyresearch/tree/master/data}
& Incombustible material & Total incombustible content & spxy
& 62 & 43 & 19 & 254 & - & 0 \\
\midrule
\multirow{2}{*}{LUCAS}
& Lucas\_SOC
& \url{https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data#tabs-0-description=0}
& Dried topsoil & Soil organic content & ks
& 8731 & 6111 & 2620 & 4200 & 0.8\% & 0 \\
& Lucas\_pH
&
& Dried topsoil & Soil pH & random
& 1763 & 1175 & 588 & 4200 & 5.6\% & 0 \\
\midrule
\multirow{5}{*}{MANURE21}
& Manure\_CaO
& \url{https://doi.org/10.15454/JIGO8R}
& Cattle manure & CaO content & strat spxy
& 490 & 343 & 147 & 1003 & 1.4\% & 0 \\
& Manure\_K2O
&
& Cattle manure & K2O content & strat spxy
& 490 & 343 & 147 & 1003 & 2.0\% & 0 \\
& Manure\_MgO
&
& Cattle manure & MgO content & strat spxy
& 490 & 343 & 147 & 1003 & 3.4\% & 0 \\
& Manure\_P2O5
&
& Cattle manure & P2O5 content & strat spxy
& 490 & 343 & 147 & 1003 & 3.4\% & 0 \\
& Manure\_N
&
& Cattle manure & N content & strat spxy
& 490 & 343 & 147 & 1003 & 4.8\% & 0 \\
\midrule
\multirow{3}{*}{MILK}
& Milk\_Fat
& \url{https://zenodo.org/records/8263430}
& Milk & Fat content & ks
& 402 & 181 & 221 & 255 & 0.5\% & 0 \\
& Milk\_Lactose
& \url{https://zenodo.org/records/8263431}
& Milk & Lactose content & ks
& 1224 & 856 & 368 & 255 & 0.5\% & 0 \\
& Milk\_Urea
& \url{https://zenodo.org/records/8263432}
& Milk & Urea content & ks
& 1224 & 856 & 368 & 255 & 0.5\% & 1 \\
\midrule
\multirow{5}{*}{PHOSPHORUS}
& Phosphorus\_LP
& \url{https://doi.org/10.6084/m9.figshare.28675304}
& Fresh leaf & Lipid P content & spxy by species
& 257 & 169 & 88 & 2101 & 13.6\% & 9 \\
& Phosphorus\_MP
&
& Fresh leaf & Metabolite P content & spxy by species
& 257 & 169 & 88 & 2101 & 13.6\% & 4 \\
& Phosphorus\_NP
&
& Fresh leaf & Nucleic acid P content & spxy by species
& 257 & 169 & 88 & 2101 & 13.6\% & 6 \\
& Phosphorus\_Pi
&
& Fresh leaf & Orthophosphate P content & spxy by species
& 257 & 169 & 88 & 2101 & 13.6\% & 8 \\
& Phosphorus\_V25
&
& Fresh leaf & Photosynthetic capacity (Vcmax25) & spxy by species
& 250 & 168 & 82 & 2101 & 14.6\% & 3 \\
\midrule
PLUMS
& Plums\_Firmness
& \url{https://github.com/nevernervous78/nirpyresearch/blob/master/data}
& Plum & Firmness & spxy
& 40 & 28 & 12 & 600 & - & 0 \\
\midrule
QUARTZ
& Quartz_Content
& \url{https://github.com/nevernervous78/nirpyresearch/blob/master/data}
& Mineral & Quartz content & spxy
& 303 & 212 & 91 & 1500 & 4.4\% & 0 \\
\midrule
TABLET
& Tablet\_ActiveSubstance
& \url{https://ucphchemometrics.com/tablet/}
& Tablet & Active Substance & ks
& 310 & 207 & 103 & 404 & 44.7\% & 2 \\
\midrule
\multirow{2}{*}{WOOD\_density}
& Wood\_Density
& \url{https://doi.org/10.34725/DVN/24522}
& Ground dried wood auger cores & Wood density & ks
& 402 & 216 & 186 & 1038 & 7.5\% & 3 \\
& Wood\_N
&
& Ground dried wood auger cores & N content & ks
& 402 & 216 & 186 & 1038 & 7.5\% & 1 \\

\end{longtable}

% ==================================================
% CLASSIFICATION
% ==================================================

\begin{longtable}{L{3.2cm}L{6.0cm}L{2.7cm}L{1.75cm}L{1.85cm}L{1.05cm}R{0.72cm}R{0.72cm}R{0.72cm}R{0.82cm}R{0.68cm}R{0.82cm}R{0.82cm}}
\caption{\textbf{Classification.} Benchmark overview for classification datasets. Source URLs of the open access databases are given when available. \texit{p} stands for the number of variables in the dataset, \textit{Class imb}. for class imbalance, and \textit{Maj. class} for the proportion of the majority class in the dataset.}
\label{tab:dataset_description_long_classification} \\
\toprule
\makecell[l]{\bfseries Database\\ } &
\makecell[l]{\bfseries Dataset\\name} &
\makecell[l]{\bfseries Source\\URL} &
\makecell[l]{\bfseries Sample\\type} &
\makecell[l]{\bfseries Target\\trait} &
\makecell[l]{\bfseries Split\\type} &
\makecell[r]{\bfseries N\\total} &
\makecell[r]{\bfseries N\\train} &
\makecell[r]{\bfseries N\\test} &
\makecell[r]{\bfseries p} &
\makecell[r]{\bfseries N\\class} &
\makecell[r]{\bfseries Class\\imb.} &
\makecell[r]{\bfseries Maj.\\class} \\
\midrule
\endfirsthead

\toprule
\makecell[l]{\bfseries Database\\ } &
\makecell[l]{\bfseries Dataset\\name} &
\makecell[l]{\bfseries Source\\URL} &
\makecell[l]{\bfseries Sample\\type} &
\makecell[l]{\bfseries Target\\trait} &
\makecell[l]{\bfseries Split\\type} &
\makecell[r]{\bfseries N\\total} &
\makecell[r]{\bfseries N\\train} &
\makecell[r]{\bfseries N\\test} &
\makecell[r]{\bfseries p} &
\makecell[r]{\bfseries N\\class} &
\makecell[r]{\bfseries Class\\imb.} &
\makecell[r]{\bfseries Maj.\\class} \\
\midrule
\endhead

\midrule
\multicolumn{13}{r}{Continued on next page} \\
\midrule
\endfoot

\bottomrule
\endlastfoot

\multirow{2}{*}{ARABIDOPSIS\_CEFE}
& Arabidopsis\_Genotype & \url{https://doi.org/10.1038/s41597-023-02189-w} & Fresh leaf & Genotype group & random blocks & 2185 & 1530 & 655 & 2152 & 10 & 3.26 & 17.6\% \\
& Arabidopsis\_GrowingCondition &  & Fresh leaf & Growing indoor vs outdoor & random blocks & 1263 & 884 & 379 & 2152 & 2 & 1.44 & 59.0\% \\
\midrule
BEEF\_Impurity
& Beef\_Purity & - & Raw and cooked beef & Pure vs adulterated meat & not specified & 60 & 30 & 30 & 470 & 5 & 1.00 & 20.0\% \\
\midrule
COFFEE\_orig
& Coffee\_Origin & \url{https://nirpyresearch.com/analysis-ground-coffee-nir-spectroscopy/} & Aldi Expressi coffee capsules & Coffee origin & strat KS & 70 & 49 & 21 & 601 & 7 & 1.00 & 14.3\% \\
\midrule
COFFEE\_sp
& Coffee\_Species & - & Freeze-dried coffee & Coffee variety & not specified & 56 & 28 & 28 & 286 & 2 & 1.07 & 51.8\% \\
\midrule
\multirow{2}{*}{FUSARIUM}
& Fusarium\_Healthy\_FinalScore & \url{https://zenodo.org/records/16217833} & Fresh strawberry plant leaf & Healthy or diseased & Stratified & 935 & 646 & 289 & 2177 & 2 & 1.39 & 58.2\% \\
& Fusarium\_Healthy\_Score &  & Fresh strawberry plant leaf & Healthy or diseased & Stratified & 816 & 578 & 238 & 2177 & 2 & 3.27 & 76.6\% \\
\midrule
FruitPuree
& FruitPuree\_Strawberry & - & Fruit purée & Strawberry or non-strawberry & ks & 983 & 666 & 317 & 235 & 2 & 1.80 & 64.3\% \\
\midrule
\multirow{2}{*}{MALARIA}
& Malaria\_Oocyst & \url{https://doi.org/10.7910/DVN/YD34OX} & Mosquito & Infected (oocyst) & not specified & 333 & 227 & 106 & 2151 & 2 & 1.09 & 52.3\% \\
& Malaria\_Sporozoite &  & Mosquito & Infected (sporozoite) & not specified & 229 & 138 & 91 & 2151 & 2 & 1.52 & 60.3\% \\
\midrule
MILK
& Milk\_Ratio & \url{https://nirpyresearch.com/detecting-lactose-milk-spectroscopy/} & Milk/lactose-free milk & Ratio milk/lactose-free & strat KS & 450 & 315 & 135 & 601 & 9 & 1.00 & 11.1\% \\
\midrule
PISTACIA
& Pistacia\_Species & \url{https://doi.org/10.18167/DVN1/J1TZZN} & herbarium specimens of pistacia & pistacia species & not specified & 7323 & 5103 & 2220 & 1951 & 5 & 5.26 & 30.9\% \\
\midrule
\multirow{2}{*}{Wood\_Sustainability}
& Wood\_Sustainability\_Binary & - & wood & sustainability & not specified & 511 & 358 & 153 & 1050 & 2 & 1.05 & 51.3\% \\
& Wood\_Sustainability\_Multiclass & - & wood & sustainability & not specified & 511 & 358 & 153 & 1050 & 5 & 6.97 & 42.3\% \\

\end{longtable}
\endgroup