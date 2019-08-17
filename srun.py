import subprocess
import argparse
import textwrap


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='srun')
    parser.add_argument('--partition', type=str, default='16gV100')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--nodelist', type=str, default='')
    known_args, unknown_args = parser.parse_known_args()

    command = textwrap.dedent(f'''\
        srun \
        --mpi=pmi2 \
        --partition={known_args.partition} \
        --nodes={known_args.num_nodes} \
        --ntasks-per-node={known_args.num_gpus} \
        --ntasks={known_args.num_nodes * known_args.num_gpus} \
        --gres=gpu:{known_args.num_gpus} \
        --nodelist={known_args.nodelist} \
        python -u {' '.join(unknown_args)}
    ''')

    subprocess.call(command.split())
