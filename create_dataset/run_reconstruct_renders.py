import reconstruct_renders

reconstruct_renders.reconstruct_all(
	input_root='../run_on_cluster/run/results',
	temp_root='../run_on_cluster/spoisson_tmp', # Note: This should be on an SSD drive for improved performance.
	poisson_exe='../bin/poisson.exe'
)
