Table of Contents
=================

 -  [Overview](#overview)
 -  [Requirements](#requirements)
 -  [Installation](#installation)
 -  [Documentation](#documentation)
 -  [Acknowledgements](#acknowledgements)
 -  [License](#license)

Overview
========

The Python script `split-freq.py` provides a command-line interface (CLI) for
comparing split frequencies between collections of trees.

NOTE: This script is not well tested, so double check your results and please
let me know if you have problems.

Requirements
============

The script requires Python (only tested under version 2.7) and the Python
package [`DendroPy`](http://pythonhosted.org/DendroPy/)
(<http://pythonhosted.org/DendroPy/>):

> Sukumaran, J. and M.T. Holder. 2010. DendroPy: A Python library for
> phylogenetic computint. Bioninformatics 26: 1569--1571.

Also, if you want to plot comparisons of split frequencies, you will also need
the Python plotting library [`matplotlib`](http://matplotlib.org/)
(<http://matplotlib.org/>).

Installation
============

Open a terminal window and navigate to where you would like to keep the `SplitFreq`
repository. Then, use `git` to clone the repository:

    git clone https://github.com/joaks1/SplitFreq.git

Move into the `SplitFreq` directory:
    
    cd SplitFreq

Call up the `split-freq.py` help menu:

    ./split-freq.py -h

If this does not work, try making the file executable:

    chmod +x split-freq.py

You can copy (or link) the file to your path using something like:

    sudo cp split-freq.py /usr/local/bin

Documentation
=============

If you have a collection of trees in file `TREE-FILE-A` and another collection
of trees in `TREE-FILE-B`, you can compare the split frequencies between these
two collections of trees using:

    split-freq.py -t TREE-FILE-A -t TREE_FILE-B

Note, all of the trees in both files must have the same set of taxa! If all
went well, the script should have output the split frequencies to the terminal
screen, and, if `matplotlib was found, the plot
`split-freq-plots/trees1-vs-trees2.pdf` was created.

Having all those split frequencies written to the screen is annoying, so let's
redirect all that standard output to a file named `freqs.txt`:

    split-freq.py -t TREE-FILE-A -t TREE_FILE-B > freqs.txt

If you want to ignore 100 trees at the beginning of each file you can use:

    split-freq.py --burnin 100 -t TREE-FILE-A -t TREE_FILE-B > freqs.txt

Let's move on to a slightly more involved example. Let's say we want to compare
the split frequencies of two different models, and we have run 4 independent
analyses under each model in MrBayes. So, we have the following eight files,
each with samples of trees

    model1-run1.t
    model1-run2.t
    model1-run3.t
    model1-run4.t
    model2-run1.t
    model2-run2.t
    model2-run3.t
    model2-run4.t

To compare the split frequencies of the models while removing the first 1000
sampled trees from each file, we can use:

    split-freq.py --burnin 1000 -t model2-run1.t model2-run2.t model2-run3.t model2-run4.t -t model2-run1.t model2-run2.t model2-run3.t model2-run4.t > freqs.txt

That's a lot of typing, but we can accomplish the same command a lot more
succinctly by taking advantage of shell wildcards:

    split-freq.py --burnin 1000 -t model1-run?.t -t model2-run?.t > freqs.txt

Having trees in separate files like this can speed things up considerably,
because `split-freq.py` can use as many processors as the number of files it is
provided.

Acknowledgements
================

This software benefited from funding provided to Jamie Oaks from the National
Science Foundation (DBI 1308885).

License
=======

Copyright (C) 2013 Jamie Oaks

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>.

See "LICENSE.txt" for full terms and conditions of usage.

