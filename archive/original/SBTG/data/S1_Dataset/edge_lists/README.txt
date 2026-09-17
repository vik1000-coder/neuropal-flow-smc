NOTES
-----

Each row of these CSV files represents an edge/link in a graph/network.

The columns are arranged in order: [source neuron, target neuron, transmitter, receptor].

Individual neurons are listed, rather than pairs/classes.

Neurons whose name includes a number are not zero-padded, for compatibility with community resources like OpenWorm and WormBase.

FILES
-----

*MA*.csv files contain monoamine edges, and *NP*.csv files contain neuropeptide edges.

*classes*.csv files contain links between neuron classes rather than individual neurons: these classes are defined based on the source expression data used in this paper.

*incl_dop-5_dop-6.csv files include edges whose receptor is dop-5 or dop-6.

FILE LIST
---------
edgelist_MA.csv
edgelist_MA_classes.csv
edgelist_MA_classes_incl_dop-5_dop-6.csv
edgelist_MA_incl_dop-5_dop-6.csv
edgelist_NP.csv
edgelist_NP_classes.csv
