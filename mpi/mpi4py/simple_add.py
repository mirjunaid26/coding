from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Each process works on its own independent data
a = rank + 1       # Example value for process rank
b = (rank + 1) * 2 # Another value
local_sum = a + b

print(f"Rank {rank}: {a} + {b} = {local_sum}")

# If you want to collect all results to rank 0:
all_sums = comm.gather(local_sum, root=0)

if rank == 0:
    print("Collected sums from all ranks:", all_sums)
    print("Total of all local sums:", sum(all_sums))
