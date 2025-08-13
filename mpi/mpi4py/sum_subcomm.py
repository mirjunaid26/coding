from mpi4py import MPI

# ================================================================
# Exercise 1.1 - Summing the ranks in subcommunicator
#
# Description:
# - Split MPI_COMM_WORLD into two equal-sized subcommunicators 
#   with reverse rank order.
# - Each subcommunicator should compute the sum of the *global*
#   ranks of all processes in that subcommunicator.
# - All processes in a subcommunicator should carry the summed value.
# - Each process should print:
#     "I am rank m, sub_rank n, and the sum of ranks in my sub_comm is i"
#     where:
#         m = rank in MPI_COMM_WORLD (global rank)
#         n = rank in subcommunicator
#         i = sum of all global ranks in that subcommunicator
#
# Testing:
# - With 8 processes → the sum should be 6
# - With 12 processes → the sum should be 28
#
# Hint:
# - Preferably use MPI_Comm_split
# ================================================================

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Determine which group (color) the process belongs to, and
# set reverse rank ordering (key) inside the group
half_size = size // 2
if rank < half_size:
    color = 0
    key = half_size - rank - 1  # reverse order in first group
else:
    color = 1
    key = size - rank - 1       # reverse order in second group

# Create the subcommunicator
sub_comm = comm.Split(color=color, key=key)
sub_rank = sub_comm.Get_rank()

# Calculate sum of *global* ranks in the subcommunicator
sum_ranks = sub_comm.allreduce(rank, op=MPI.SUM)

# Print the required message
print(f"I am rank {rank}, sub_rank {sub_rank}, "
      f"and the sum of ranks in my sub_comm is {sum_ranks}")
# Finalize the MPI environment
MPI.Finalize()