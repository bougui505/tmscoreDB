#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2021-02-17 08:40:18 (UTC+0100)

import gzip
import io
import numpy as np
from scipy.sparse import coo_matrix
import sys
import os


class TMscoreDB(object):
    def __init__(self, recfilename: str = None):
        """__init__.

        :param recfilename:
        :type recfilename: str
        """
        self.recfile = recfilename
        self.train_ind = dict()
        self.test_ind = dict()
        self.rows = []
        self.cols = []
        self.data = []
        if self.recfile is not None:
            self.build()

    def build(self):
        with io.TextIOWrapper(io.BufferedReader(gzip.open(self.recfile, 'r'))) as recfile:
            datapoint = []
            for line in recfile:
                splitout = line.split(':')
                if splitout[0] == 'train':
                    train = splitout[1].strip()
                    datapoint.append(train)
                if splitout[0] == 'test':
                    test = splitout[1].strip()
                    datapoint.append(test)
                if splitout[0] == 'tmscore':
                    tmscore = float(splitout[1].strip())
                    datapoint.append(tmscore)
                if splitout[0] == '\n':
                    self.add(datapoint)
                    datapoint = []
        self.format()

    def add(self, datapoint: list):
        """add.

        :param datapoint:
        :type datapoint: list
        """
        train, test, tmscore = datapoint
        train = os.path.split(train)[-1]
        test = os.path.split(test)[-1]
        if len(self.train_ind) > 0:
            train_ind_max = max(self.train_ind.values())
        else:
            train_ind_max = -1
        if len(self.test_ind) > 0:
            test_ind_max = max(self.test_ind.values())
        else:
            test_ind_max = -1
        if train not in self.train_ind:
            self.train_ind[train] = train_ind_max + 1
        if test not in self.test_ind:
            self.test_ind[test] = test_ind_max + 1
        sys.stdout.write(f"Append data point: ({self.train_ind[train]}, {self.test_ind[test]})              \r")
        sys.stdout.flush()
        self.rows.append(self.train_ind[train])
        self.cols.append(self.test_ind[test])
        self.data.append(tmscore)

    def format(self):
        n, p = max(self.rows) + 1, max(self.cols) + 1
        self.data = coo_matrix((self.data, (self.rows, self.cols)), shape=(n, p)).toarray()
        del self.rows, self.cols
        print(f'Shape of the data array: {self.data.shape}')

    def reformat_keys(self) -> None:
        """reformat_keys.

        :rtype: None

        Reformat the keys by removing the leading path
        HD-database/1/5/c/8/15c8-LH-P01837-P01869-H.pdb -> 15c8-LH-P01837-P01869-H.pdb
        """
        train_ind = dict()
        for train in self.train_ind:
            train_new = os.path.split(train)[-1]
            train_ind[train_new] = self.train_ind[train]
        test_ind = dict()
        for test in self.test_ind:
            test_new = os.path.split(test)[-1]
            test_ind[test_new] = self.test_ind[test]
        self.train_ind = train_ind
        self.test_ind = test_ind

    def filter(self, train_keys: list, test_keys: list) -> None:
        """filter.

        :param train_keys:
        :type train_keys: list
        :param test_keys:
        :type test_keys: list

        Filter the database with the given keys
        """
        train_ind = dict()
        test_ind = dict()
        train_keys = np.unique(train_keys)
        test_keys = np.unique(test_keys)
        data = np.zeros((len(train_keys), len(test_keys)))
        for i, train in enumerate(train_keys):
            train_ind[train] = i
            for j, test in enumerate(test_keys):
                test_ind[test] = j
                data[i, j] = self.get(train, test)
        self.train_ind = train_ind
        self.test_ind = test_ind
        self.data = data

    def get(self, train: str, test: str) -> float:
        """get.

        :param train:
        :type train: str
        :param test:
        :type test: str
        :rtype: float

        Get the given data
        """
        train = os.path.split(train)[-1]
        test = os.path.split(test)[-1]
        i = self.train_ind[train]
        j = self.test_ind[test]
        return self.data[i, j]

    def print(self, getmax: bool = False, print_keys: bool = False):
        """print.

        :param getmax:

        Print all the data on stdout
        """
        def sort_keys(mydict):
            kv = mydict.items()
            kv = sorted(kv, key=lambda x: x[1])
            keys = np.asarray([e[0] for e in kv])
            return keys
        if not getmax:
            toprint = self.data.flatten()
        else:
            toprint = self.data.max(axis=1).flatten()
        if print_keys:
            train_keys = sort_keys(self.train_ind)
            test_keys = sort_keys(self.test_ind)
            inds = self.data.argmax(axis=1)
            test_keys = test_keys[inds]
            toprint = np.vstack(((train_keys, test_keys, toprint.astype(np.str)))).T
        np.savetxt(sys.stdout, toprint, fmt='%s')


if __name__ == '__main__':
    import argparse
    import pickle
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-r', '--rec', type=str, default=None, help='Build the DB from a recfile containing the tmscore data')
    parser.add_argument('-o', '--out', default=None, help='Output database filename', type=str)
    parser.add_argument('-d', '--db', type=str, default=None, help='Load the given db pickle file')
    parser.add_argument('-p', '--print', action='store_true', help='Print all the data in the db')
    parser.add_argument('-m', '--max', action='store_true', help='Print the max over axis 1 of the data in the db')
    parser.add_argument('-g', '--get', nargs=2, type=str, help='Get the data for the given train, test couple')
    parser.add_argument('-f', '--filter', type=str, help='Filter the database with the given test file containing one couple "train_key test_key" per line')
    parser.add_argument('-k', '--keys', action='store_true', help='Print the keys along with the data')
    parser.add_argument('--format', action='store_true', help='Reformat the keys such as HD-database/1/5/c/8/15c8-LH-P01837-P01869-H.pdb -> 15c8-LH-P01837-P01869-H.pdb')
    args = parser.parse_args()

    def test_out():
        if args.out is None:
            print("Please give on output filename with the --out option e.g.: '--out tmscore.pickle'")
            sys.exit(1)

    if args.rec is not None:
        test_out()
        tmscoredb = TMscoreDB(recfilename=args.rec)
        pickle.dump(tmscoredb, open(args.out, 'wb'))
    if args.db is not None:
        tmscoredb = pickle.load(open(args.db, 'rb'))
        if args.format:
            test_out()
            tmscoredb.reformat_keys()
            pickle.dump(tmscoredb, open(args.out, 'wb'))
        if args.print:
            tmscoredb.print()
        if args.max:
            tmscoredb.print(getmax=True, print_keys=args.keys)
        if args.get is not None:
            data = tmscoredb.get(train=args.get[0], test=args.get[1])
            print(f'{data:.4f}')
        if args.filter is not None:
            test_out()
            traintest = np.genfromtxt(args.filter, dtype=str)
            train = traintest[:, 0]
            test = traintest[:, 1]
            tmscoredb.filter(train, test)
            pickle.dump(tmscoredb, open(args.out, 'wb'))
