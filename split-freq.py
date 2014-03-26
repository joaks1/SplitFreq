#! /usr/bin/env python

import os
import sys
import copy
import argparse
import logging
import multiprocessing
import Queue
import time

import dendropy

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(os.path.basename(__file__))
_LOCK = multiprocessing.Lock()

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    _LOG.warning('matplotlib could not be imported; '
            'plotting functionality not supported')

_program_info = {
    'name': os.path.basename(__file__),
    'author': 'Jamie Oaks',
    'version': 'Version 0.1.0',
    'copyright': 'Copyright (C) 2014 Jamie Oaks.',
    'license': (
        'This is free software distributed under the GNU General Public '
        'License in the hope that it will be useful, but WITHOUT ANY '
        'WARRANTY. You are free to change and redistribute it in accord with '
        'the GPL. See the GNU General Public License for more details.'),}

if MATPLOTLIB_AVAILABLE:
    matplotlib.rcParams['pdf.fonttype'] = 42

    class ScatterPlot(object):
        def __init__(self, x, y,
                x_label = None,
                y_label = None,
                x_label_size = None,
                y_label_size = None,
                height = 6.0,
                width = 8.0,
                position = (1,1,1),
                xlim = (None, None),
                ylim = (None, None),
                marker = 'o',
                markerfacecolor = 'none',
                markeredgecolor = '0.35',
                markeredgewidth = 0.7,
                markersize = None,
                linestyle = '',
                perimeter_padding = 0.25,
                margin_left = 0,
                margin_right = 1,
                margin_bottom = 0,
                margin_top = 1,
                xticks = None,
                xtick_labels = None,
                xtick_label_size = 10.0,
                yticks = None,
                ytick_labels = None,
                ytick_label_size = 10.0,
                zorder = 100,
                **kwargs):
            self.x = list(x)
            self.y = list(y)
            self.x_label = x_label
            self.y_label = y_label
            self.x_label_size = x_label_size
            self.y_label_size = y_label_size
            self.width = width
            self.height = height
            self.fig = plt.figure(figsize = (self.width, self.height))
            self.ax = self.fig.add_subplot(*position)
            self.xlim_left = xlim[0]
            self.xlim_right = xlim[1]
            self.ylim_bottom = ylim[0]
            self.ylim_top = ylim[1]
            self.identity_color = '0.5'
            self.identity_style = '-'
            self.identity_width = 1.0
            self.marker = marker
            self.markerfacecolor = markerfacecolor
            self.markeredgecolor = markeredgecolor
            self.markeredgewidth = markeredgewidth
            self.markersize = markersize
            self.linestyle = linestyle
            self.perimeter_padding = perimeter_padding
            self.margin_left = margin_left
            self.margin_right = margin_right
            self.margin_bottom = margin_bottom
            self.margin_top = margin_top
            self.xticks = xticks
            self.xtick_labels = xtick_labels
            self.xtick_label_size = xtick_label_size
            self.yticks = yticks
            self.ytick_labels = ytick_labels
            self.ytick_label_size = ytick_label_size
            self.zorder = zorder
            self.kwargs = kwargs
            self._plot()
            self._reset_figure()
    
        def _plot(self):
            l = self.ax.plot(self.x, self.y)
            args = {'marker': self.marker,
                    'linestyle': self.linestyle,
                    'markerfacecolor': self.markerfacecolor,
                    'markeredgecolor': self.markeredgecolor,
                    'markeredgewidth': self.markeredgewidth,
                    'zorder': self.zorder,
                    }
            args.update(self.kwargs)
            if self.markersize != None:
                args['markersize'] = self.markersize
            plt.setp(l, **args)
    
            self.ax.set_xlim(left = self.xlim_left, right = self.xlim_right)
            self.ax.set_ylim(bottom = self.ylim_bottom, top = self.ylim_top)
            mn = self.get_minimum()
            mx = self.get_maximum()
            l = self.ax.plot([mn, mx], [mn, mx]) 
            plt.setp(l,
                    color = self.identity_color,
                    linestyle = self.identity_style,
                    linewidth = self.identity_width,
                    marker = '',
                    zorder = 0)
            self.ax.set_xlabel(
                    xlabel = self.x_label,
                    fontsize = self.x_label_size)
            self.ax.set_ylabel(
                    ylabel = self.y_label,
                    fontsize = self.y_label_size)
            self.ax.set_xlim(left = self.xlim_left, right = self.xlim_right)
            self.ax.set_ylim(bottom = self.ylim_bottom, top = self.ylim_top)
            if self.xticks:
                self.ax.set_xticks(self.xticks)
            if self.xtick_labels:
                self.ax.set_xticklabels(self.xtick_labels,
                        size = self.xtick_label_size)
            if self.yticks:
                self.ax.set_yticks(self.yticks)
            if self.ytick_labels:
                self.ax.set_yticklabels(self.ytick_labels,
                        size = self.ytick_label_size)
    
        def get_origin(self):
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            return xmin, ymin
    
        def get_minimum(self):
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            return min([xmin, ymin])
    
        def get_maximum(self):
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            return max([xmax, ymax])
    
        def _reset_figure(self):
            rect = (self.margin_left, self.margin_bottom, self.margin_right,
                    self.margin_top)
            self.fig.tight_layout(pad = self.perimeter_padding,
                    rect = rect) # available space on figure
    
        def savefig(self, *args, **kwargs):
            self.fig.savefig(*args, **kwargs)

class Manager(multiprocessing.Process):
    count = 0
    def __init__(self,
            work_queue,
            result_queue = None,
            get_timeout = 0.4,
            put_timeout = 0.2,
            log = None,
            lock = None):
        multiprocessing.Process.__init__(self)
        self.__class__.count += 1
        self.name = self.__class__.__name__ + '-' + str(self.count)
        if not result_queue:
            result_queue = multiprocessing.Queue()
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.get_timeout = get_timeout
        self.put_timeout = put_timeout
        if not log:
            log = _LOG
        self.log = log
        if not lock:
            lock = _LOCK
        self.lock = lock
        self.killed = False

    def compose_msg(self, msg):
        return '{0} ({1}): {2}'.format(self.name, self.pid, msg)

    def send_msg(self, msg, method_str='info'):
        self.lock.acquire()
        try:
            getattr(self.log, method_str)(self.compose_msg(msg))
        finally:
            self.lock.release()

    def send_debug(self, msg):
        self.send_msg(msg, method_str='debug')

    def send_info(self, msg):
        self.send_msg(msg, method_str='info')

    def send_warning(self, msg):
        self.send_msg(msg, method_str='warning')

    def send_error(self, msg):
        self.send_msg(msg, method_str='error')

    def _get_worker(self):
        worker = None
        try:
            self.send_debug('getting worker')
            worker = self.work_queue.get(block=True, timeout=self.get_timeout)
            self.send_debug('received worker {0}'.format(
                    getattr(worker, 'name', 'nameless')))
            # without blocking processes were stopping when the queue
            # was not empty, and without timeout, the processes would
            # hang waiting for jobs.
        except Queue.Empty:
            time.sleep(0.2)
            if not self.work_queue.empty():
                self.send_warning('raised Queue.Empty, but queue is '
                        'not empty... trying again')
                return self._get_worker()
            else:
                self.send_info('work queue is empty')
        return worker

    def _put_worker(self, worker):
        try:
            self.send_debug('returning worker {0}'.format(
                    getattr(worker, 'name', 'nameless')))
            self.result_queue.put(worker, block=True, timeout=self.put_timeout)
            self.send_debug('worker {0} returned'.format(
                    getattr(worker, 'name', 'nameless')))
        except Queue.Full, e:
            time.sleep(0.2)
            if not self.result_queue.full():
                self.send_warning('raised Queue.Full, but queue is '
                        'not full... trying again')
                self._put_worker(worker)
            else:
                self.send_error('result queue is full... aborting')
            self.killed = True
            raise e

    def run(self):
        self.send_debug('starting run')
        while not self.killed:
            worker = self._get_worker()
            if worker is None:
                break
            self.send_info('starting worker {0}'.format(
                    getattr(worker, 'name', 'nameless')))
            worker.start()
            self.send_info('worker {0} finished'.format(
                    getattr(worker, 'name', 'nameless')))
            self._put_worker(worker)
        if self.killed:
            self.send_error('manager was killed!')
        self.send_debug('end run')

    @classmethod
    def run_workers(cls,
            workers,
            num_processors,
            get_timeout = 0.4,
            put_timeout = 0.2,
            queue_max = 500):
        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        finished = []
        for w_list in list_splitter(workers, queue_max, by_size = True):
            assert work_queue.empty()
            assert result_queue.empty()
            for w in w_list:
                work_queue.put(w)
            managers = []
            for i in range(num_processors):
                m = cls(work_queue = work_queue,
                        result_queue = result_queue,
                        get_timeout = get_timeout,
                        put_timeout = put_timeout)
                managers.append(m)
            for i in range(len(managers)):
                managers[i].start()
            for i in range(len(w_list)):
                _LOG.debug('Manager.run_workers: getting result...')
                w_list[i] = result_queue.get()
                _LOG.debug('Manager.run_workers: got result {0}'.format(
                        getattr(w_list[i], 'name', 'nameless')))
            for i in range(len(managers)):
                managers[i].join()
            for w in w_list:
                if getattr(w, "error", None):
                    _LOG.error('Worker {0} returned with an error:\n{1}'.format(
                            getattr(w, 'name', 'nameless'),
                            w.trace_back))
                    raise w.error
            assert work_queue.empty()
            assert result_queue.empty()
            finished.extend(w_list)
        return finished

class SplitFreqWorker(object):
    count = 0
    def __init__(self, list_of_tree_file_paths,
            schema = 'nexus/newick',
            taxon_set = None,
            is_rooted = False,
            tree_offset = 0,
            tag = None):
        self.__class__.count += 1
        self.name = self.__class__.__name__ + '-' + str(self.count)
        self.paths = list_of_tree_file_paths
        self.schema = schema
        self.taxon_set = dendropy.TaxonSet()
        if isinstance(taxon_set, dendropy.TaxonSet):
            self.taxon_set = copy.deepcopy(taxon_set)
            self.taxon_set.sort(key=lambda taxonobject: taxonobject.label)
        self.is_rooted = is_rooted
        self.split_frequencies = {}
        self.tree_offset = tree_offset
        self.tag = tag
        self.n = 0
        self.finished = False

    def start(self):
        split_dist = dendropy.treesplit.SplitDistribution(
                taxon_set = self.taxon_set)
        for tree_file_num, tree_file in enumerate(self.paths):
            stream = open(tree_file, 'r')
            tree_iter = dendropy.dataio.ioclient.tree_source_iter(stream,
                    schema = self.schema,
                    tree_offset = self.tree_offset,
                    taxon_set = self.taxon_set,
                    as_rooted = self.is_rooted,
                    preserve_underscores=True)
            for tree_num, tree in enumerate(tree_iter):
                dendropy.treesplit.encode_splits(tree)
                split_dist.count_splits_on_tree(tree)
                self.n += 1
            stream.close()
        split_freqs = split_dist.split_frequencies
        for split, freq in split_freqs.iteritems():
            pretty_split = dendropy.treesplit.split_as_string_rev(split,
                    width = len(split_dist.taxon_set),
                    symbol1 = '0', symbol2 = '1')
            self.split_frequencies[pretty_split] = freq
        self.finished = True

def list_splitter(l, n, by_size=False):
    """
    Returns generator that yields list `l` as `n` sublists, or as `n`-sized
    sublists if `by_size` is True.
    """
    if n < 1:
        raise StopIteration
    elif by_size:
        for i in range(0, len(l), n):
            yield l[i:i+n]
    else:
        if n > len(l):
            n = len(l)
        step_size = len(l)/int(n)
        if step_size < 1:
            step_size = 1
        i = -step_size
        for i in range(0, ((n-1)*step_size), step_size):
            yield l[i:i+step_size]
        yield l[i+step_size:]

def merge_split_frequencies(split_freq_worker_1, split_freq_worker_2):
    assert (sorted(split_freq_worker_1.taxon_set.labels()) == 
            sorted(split_freq_worker_2.taxon_set.labels()))
    keys = list(set(split_freq_worker_1.split_frequencies.keys() +
            split_freq_worker_2.split_frequencies.keys()))
    splits = {}
    for k in keys:
        splits[k] = (split_freq_worker_1.split_frequencies.get(k, 0.0),
                split_freq_worker_2.split_frequencies.get(k, 0.0))
    return splits

def write_split_freqs(split_freqs, stream = sys.stdout):
    stream.write('split\tfreq1\tfreq2\n')
    for k, (freq1, freq2) in split_freqs.iteritems():
        stream.write('{0}\t{1}\t{2}\n'.format(k, freq1, freq2))

def expand_path(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))

def arg_is_file(path):
    try:
        if not os.path.isfile(path):
            raise
    except:
        msg = '{0!r} is not a file'.format(path)
        raise argparse.ArgumentTypeError(msg)
    return expand_path(path)

def arg_is_nonnegative_int(i):
    try:
        if int(i) < 0:
            raise
    except:
        msg = '{0!r} is not a non-negative integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)

def main_cli():
    description = '{name} {version}'.format(**_program_info)
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('-a', '--tree-files-a',
            metavar='INPUT-TREE-FILE-A',
            nargs = '+',
            type = arg_is_file,
            help = ('Input tree file(s) to be compared.'))
    parser.add_argument('-b', '--tree-files-b',
            metavar='INPUT-TREE-FILE-B',
            nargs = '+',
            type = arg_is_file,
            help = ('Input tree file(s) to be compared.'))
    parser.add_argument('--burnin',
            type = arg_is_nonnegative_int,
            default = 0,
            help = ('The number of trees to ignore from the beginning of each '
                    'input file.'))
    parser.add_argument('-p', '--plot-path',
            type = str,
            help = ('Output path for split-frequency plot. Requires '
                    'matplotlib.'))

    args = parser.parse_args()

    ##########################################################################
    ## handle args

    workers = []
    workers.append(SplitFreqWorker(args.tree_files_a,
            tree_offset = args.burnin,
            tag = 'a'))
    workers.append(SplitFreqWorker(args.tree_files_b,
            tree_offset = args.burnin,
            tag = 'b'))
    _LOG.info("Counting split frequencies from tree files...")
    workers = Manager.run_workers(workers, num_processors = 2)
    sfwa, sfwb = None, None
    for w in workers:
        if w.tag == 'a':
            sfwa = w
        elif w.tag == 'b':
            sfwb = w
        else:
            raise Exception('Unexpected worker tag')

    _LOG.info("Merging split frequences...")
    splits = merge_split_frequencies(sfwa, sfwb)
    write_split_freqs(splits)

    if args.plot_path:
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning(
                    '`matplotlib` could not be imported, so the plot can not\n'
                    'be produced. The data to create the plot was written to\n'
                    'standard output.')
            sys.exit(0)
        ticks = [x / 10.0 for x in range(11)]
        tick_labels = []
        for i, t in enumerate(ticks):
            if i % 2 == 0:
                tick_labels.append(t)
            else:
                tick_labels.append('')
        keys = sorted(splits.keys())
        x = [splits[k][0] for k in keys]
        y = [splits[k][1] for k in keys]
        sp = ScatterPlot(
                x = x,
                y = y,
                x_label = 'Split frequencies a',
                y_label = 'Split frequencies b',
                x_label_size = 16.0,
                y_label_size = 16.0,
                height = 4.0,
                width = 6.0,
                xlim = (0, 1.0),
                ylim = (0, 1.0),
                perimeter_padding = 0.2,
                margin_left = 0,
                margin_right = 1,
                margin_bottom = 0,
                margin_top = 1,
                xticks = ticks,
                xtick_labels = tick_labels,
                xtick_label_size = 12.0,
                yticks = ticks,
                ytick_labels = tick_labels,
                ytick_label_size = 12.0)
        sp.savefig(args.plot_path)

if __name__ == '__main__':
    main_cli()

