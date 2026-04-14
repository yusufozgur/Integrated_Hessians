= Integrated Hessians

== Introduction

== API

== Implementation

There should be two modes for the api when we are done. Functional mode was added in hopes that it is faster. However, we have seen that calculating full hessian at once could run us out of memory. So we will a a .grad() based mode where we will be more mindful of memory limitations with cpu and gpu allocations, this should enable us to apply Integrated Hessians to much bigger models, or run in limited resource environments. Also, this should will enable calculating the subsetted regions of hessian matrix.

== Simulations

We have devised certain simulations to ascertain if the gradient based feature attribution and interaction attribution methods could be used to reveal underlying rules of the system that models have learned, specifically additive of individual features and interactive effects of feature pairs.

As our usecase is to apply this method to genomic sequence to function models, we have modelled a genomic system in our simulations, where certain motifs are inserted in a random nucleotide background, and each motif could have additive or phenotypic effect to a phenotype, where there is one phenotypic value per a sequence of 100 nucleotides. There are different simulations, each tests a different set of additive and interactive effect conditions.

=== XOR Model

This was created to show that our implementation works as expected for the xor example shown in janizek et al, and its results are consistent with the path_explain package.
