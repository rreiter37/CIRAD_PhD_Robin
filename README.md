# Thèse en informatique

### Exploration de méthodes d'assemblage de modèles pour la prédiction en spectroscopie proche infrarouge

> Doctorant : Robin Reiter
> 
> Encadrants : Grégory Beurier, CIRAD, AGAP Institut; Denis Cornet, CIRAD, AGAP Institut; Lauriane Rouan, CIRAD, AGAP Institut; Fabien Michel, LIRMM, Équipe SMILE
>
> Laboratoire d'accueil : [UMR AGAP Institut](https://umragap.cirad.fr/), Avenue Agropolis - 34398 Montpellier Cedex 5
> 
> Établissement de rattachement : Université de Montpellier, École Doctorale [I2S](https://edi2s.umontpellier.fr/)
>
> Contact : robin.reiter@cirad.fr  

---
**Champs scientifiques** : Informatique, mathématiques appliquées, statistiques, biologie, agronomie

### Contexte et problématique 

La spectroscopie proche infrarouge (NIRS) est une technique d'analyse rapide, non
destructive et à faible coût, très largement utilisée dans de nombreux domaines tels que la
santé, la chimie, l’agro-alimentaire et notamment l'agronomie. Elle permet de déterminer la
composition chimique et les propriétés fonctionnelles d'échantillons de produits tels que les
grains, fourrages, aliments, et tissus. Les données spectrales générées par NIRS sont riches en
informations mais nécessitent des traitements statistiques avancés pour des prédictions
précises. Des méthodes comme la régression PLS ont été historiquement utilisées, mais les
avancées en apprentissage machine (réseaux de neurones, SVM, random forest, etc.) et l'accès
à d'importantes bases de données NIRS ont permis l'adoption croissante de ces méthodes
d'intelligence artificielle, qui démontrent souvent de meilleures performances prédictives.


La démocratisation des spectromètres et l’augmentation croissante d’utilisateurs non
spécialistes, au Nord comme au Sud, renforce la nécessité de développer une approche
générique et performante de la calibration de modèles NIRS. Le stacking, méthode qui
combine les prédictions de multiples modèles, a démontré son potentiel pour exploiter les
forces complémentaires de différents algorithmes et améliorer les performances de prédiction.
Cependant, les stratégies de stacking restent sous-explorées pour l'analyse des données NIRS.
Dans ce contexte, le package Python [Pinard](https://pypi.org/project/pinard/) (a Pipeline for Nirs Analysis ReloadeD) développé par l'équipe encadrante fournit une base idéale
pour l'implémentation et le test des approches de prédiction à base de stacking.

### Objectifs

L'objectif principal de cette thèse est de développer et d'optimiser des stratégies de stacking
adaptées à la prédiction à partir de spectres NIRS en s'appuyant sur le package Pinard. Pinard
fournit déjà des outils pour le traitement et l'analyse des données NIRS, y compris des
modèles prédictifs individuels, mais ne propose pas actuellement de méthodologies
d'assemblage de modèles. Cette recherche vise à combler cette lacune en intégrant des
techniques avancées de stacking, permettant une amélioration significative des performances
prédictives.  
En particulier, le travail de thèse s’articulera autour des axes suivants (qui peuvent évoluer en
cours de doctorat et qui sont d’importances variées):  

• **Axe 1**: étudier et concevoir des méthodes de standardisation des données pour nourrir les
différentes classes de modèles de la stack ; en particulier en ce qui concerne les contraintes
des modèles de machine learning ou des sources différentes. Ce travail inclura également une
prise en main et une analyse poussée des jeux de données à disposition.  

• **Axe 2**: sélectionner, intégrer et hyperparamétrer des modèles de prédictions (existants ou
nouveaux) au sein d’une stack « traditionnelle » et étudier l’impact de chacun sur la précision
globale en fonction des jeux de données et des méthodes d’assemblage (sélection aléatoire,
sélection basée sur la performance, sur la diversité des algorithmes, sur la dissemblance des
prédictions, etc.).  

• **Axe 3**: Concevoir et explorer des stratégies efficaces afin d’améliorer les stratégies de
stacking de modèles en termes de précision, d’efficacité et de sobriété:
- Heuristiques issues de l’intelligence artificielle distribuée (systèmes multi-agents) ou
de l’optimisation (méthodes évolutionnistes),
- Calcul temps réel de la contribution et/ou de l’explicabilité des modèles,
- Organisation et sélection dynamique des prétraitements de données,
- Hyperparamétrisation partielle temps réel,
- Etc.
L’axe 3 est au cœur de la problématique de cette thèse et devrait légitimement représenter une
grande partie du travail du doctorant.

• **Axe 4**: Travailler sur la diffusion des résultats obtenus que ce soit en facilitant la
réutilisation de la stack ou l’accès aux outils et méthodes:
- Transfert de modèles à de nouveaux analytes / jeux de données / machines,
- Etude de l’explicabilité sous-jacente des modèles de la stack et identification des
composants du signal,
- Intégration des développements dans le package Pinard.


Ce travail fournira des approches innovantes et performantes pour exploiter la richesse des
données NIRS. Ainsi, il permettra d’améliorer la précision et la robustesse des analyses NIRS
pour des problématiques telles que l’identification rapide de variétés adaptées aux défis
climatiques, la détection et quantification de contaminants biotiques et abiotiques dans les
récoltes, l’optimisation de la qualité et la valeur nutritive des aliments transformés, etc.
contribuant de fait à des thématiques chères au CIRAD telles que la sécurité alimentaire, la
gestion durable des ressources et l'amélioration de la santé dans les pays du Sud.


---
- Lien Overleaf sur la prise de note d'articles, sur quelques idées éventuellement utilisables dans la thèse :  
🔗 [Lien Overleaf](https://fr.overleaf.com/read/jyjsqtfzbfrr#16f00e))

- Lien tableau des pistes explorées pendant la thèse :  
🔗 [Tableau](https://docs.google.com/spreadsheets/d/1C_fZ68UYlKwodlAuffRNj_JOBRbEiVaLZdlFl_1nukQ/edit?usp=sharing)

- Importer le projet et utiliser les scripts Python:  
> git clone https://github.com/rreiter37/CIRAD_PhD_Robin  
> cd CIRAD_PhD_Robin  
> pip install -r requirements.txt
