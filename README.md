
# Code for "Evolutionary dynamics under phenotypic uncertainty" (Mohanty and Sappington et al., RECOMB, 2026) and "Phenotypic noise drives evolutionary dynamics in adapting populations"

Implementation of [Evolutionary dynamics under phenotypic uncertainty](https://www.biorxiv.org/content/10.64898/2026.03.15.711953v1) by Vaibhav Mohanty*, Anna Sappington*, Eugene I. Shakhnovich** and Bonnie Berger**.

<img width="520" height="184" alt="Screenshot 2026-11-17 at 6 09 08 PM" src="https://github.com/user-attachments/assets/8d316b08-8f9e-42d0-ab60-91199c19d2f4" />

----------------------------------------------------------------------------------------------------------------------------------------

Included in this repo is code needed to run simulations and reproduce their results for the figures in the paper, outlined below:

**Figure 2**

- ProSed simulation, data, plotting files for buoy Fig 2.

**Figure 3**

- ProSed simulation, data, plotting files for mu / phi phase diagrams for Fig 3.

**Figure 4**

- ProSed simulation, data, plotting files for bridge Fig 4.

**Figure 5**

- ProSed simulation, data, plotting files for absolute fitness Fig 5 (DGP and PrGP).

**Figure 6**

- ProSed simulation, data, plotting files for mean fitness Fig 6.

**Figure 7**

- Analytical solution / plots for persister experiment

----------------------------------------------------------------------------------------------------------------------------------------

To run the code for each figure, follow these steps: 

``pip install -r requirements.txt``

``cd FigN && bash run_*.sh`` (replace ``N`` with the desired Figure number)

----------------------------------------------------------------------------------------------------------------------------------------

We also include an example notebook an example data files so users can run ProSeD simulations and our analytical theory on their own PrGP maps. This is located in the Example notebook folder.   

