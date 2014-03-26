Table of Contents
=================

 -  [Overview](#overview)
 -  [Requirements](#requirements)
 -  [Installation](#installation)
 -  [Documentation](#documentation)
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

    split-freq.py -a TREE-FILE-A -b TREE_FILE-B

Note, all of the trees in both files must have the same set of taxa!

If you want to ignore 100 trees at the beginning of each file you can use:

    split-freq.py --burnin 100 -a TREE-FILE-A -b TREE_FILE-B

If you want to create a plot of the comparison named `split-freq-plot.pdf` use:

    split-freq.py --burnin 100 -a TREE-FILE-A -b TREE_FILE-B -p split-freq-plot.pdf

You must have `matplotlib` installed to create the plot.

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

