import os
import inspect
import timeit
from statistics import mean, stdev
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
    print(this_module)
    exec('from benchmarks import {}'.format(this_module))
    for mem in inspect.getmembers(locals()[this_module],
                                  predicate=inspect.isclass):
        if mem[0].startswith('Time'):
            print(mem[0])
            t = mem[1]()
            t.setup()
            for method in inspect.getmembers(t, predicate=inspect.ismethod):
                if method[0].startswith('time_'):
                    stats = timeit.repeat(
                        "t.{}()".format(method[0]),
                        globals=globals(),
                        number=100,
                        repeat=10)

                    mean_t = mean(stats[1:]) * 10
                    stdev_t = stdev(stats[1:]) * 10
                    sys.stdout.write(method[0] + ': ')
                    sys.stdout.write(str(mean_t) + ' ± (std) ' +
                                     str(stdev_t) + ' [ms]')
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
