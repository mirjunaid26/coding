from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# Split into two groups: first half (color=0), second half (color=1)
# Reverse order inside each group by making key = -rank
if rank < size / 2:
    color = 0
else:
    color = 1

key = -rank  # reverse order

# Create subcommunicator
sub_comm = comm.Split(color=color, key=key)
sub_rank = sub_comm.Get_rank()

# Sum *global* ranks in this subcommunicator
sub_sum = sub_comm.allreduce(rank, op=MPI.SUM)

# Print as requested
print(f"I am rank {rank}, sub_rank {sub_rank}, "
      f"and the sum of ranks in my sub_comm is {sub_sum}")

