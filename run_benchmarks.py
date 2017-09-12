import os
import inspect
import timeit
from statistics import mean, stdev
from math import log10
import sys
import argparse
from datetime import datetime

modules = []
results = []

parser = argparse.ArgumentParser(description='Run all benchmarks')
parser.add_argument('--out', type=str, help='Output csv file')
parser.add_argument('--append', action='store_true',
                    help='Append to previous results')
args = parser.parse_args()


for f in os.walk('benchmarks'):
    if f[0] == 'benchmarks':
        for file in f[2]:
            if file.startswith('benchmark_') and file.endswith('.py'):
                modules.append(file.strip('.py'))

now = datetime.now().isoformat()

for this_module in modules:
    print('File:', this_module)
    exec('from benchmarks import {}'.format(this_module))
    for mem in inspect.getmembers(locals()[this_module],
                                  predicate=inspect.isclass):
        if mem[0].startswith('Time'):
            print(mem[0])
            t = mem[1]()
            t.setup()
            for method in inspect.getmembers(t, predicate=inspect.ismethod):
                if method[0].startswith('time_'):
                    try:
                        # Run a single test to determine the number of
                        # repetitions
                        test = timeit.timeit(
                            "t.{}()".format(method[0]),
                            globals=globals(),
                            number=1)

                        # Cap the number of repetitions
                        fac = 10**round(log10(0.5/test))
                        if fac > 10000:
                            fac = 10000
                        if fac < 1:
                            fac = 1

                        stats = timeit.repeat(
                            "t.{}()".format(method[0]),
                            globals=globals(),
                            number=fac,
                            repeat=11)

                        mean_t = mean(stats[1:]) / fac * 1000
                        stdev_t = stdev(stats[1:]) / fac * 1000
                        sys.stdout.write('\033[94m' + method[0] + ': ' +
                                         '\033[0m')
                        sys.stdout.write(
                            str(round(mean_t, 6)) + ' ± (std) ' +
                            str(round(stdev_t/mean_t*100)) + '% [ms]')
                        sys.stdout.write(os.linesep)

                        results.append({
                            'when': now,
                            'module': this_module,
                            'class': mem[0],
                            'benchmark': method[0],
                            'mean': mean_t,
                            'std': stdev_t,
                            'unit': 'ms'
                        })
                    # Do not break the benchmarks due to buggy code.
                    except:
                        print(method[0], 'F')

if args.append:
    spec = 'a'
else:
    spec = 'w'

if args.out:
    with open(args.out, spec) as f:
        for i, c in enumerate(results):
            if not args.append:
                if i == 0:
                    f.write(','.join(c))
                    f.write(os.linesep)

            f.write(','.join([str(x) for x in c.values()]))
            f.write(os.linesep)
