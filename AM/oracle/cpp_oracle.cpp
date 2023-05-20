#include "cpp_oracle.hpp"

float cpp_oracle_(vector<bool> map, vector<int> agents, vector<int> sep, int num_of_agents, int num_of_rows, int num_of_cols) {
	vector<vector<int>> locations;
	locations.resize(num_of_agents);

	int pos = 0;
	for (int i = 0; i < num_of_agents; i++)
	{
		vector<int> locs;
		locs.resize(sep[i]);
		for (int j = 0; j < sep[i]; j++) {
			locs[j] = agents[pos + j];
		}
		locations[i] = locs;
		pos += sep[i];
	}

	Instance instance(map, locations, num_of_agents,  num_of_rows, num_of_cols);

    PBS pbs(instance, true, 0);

    float cost = pbs.solve(60);
    pbs.savePaths("./latest_pbs_instance_paths");
	pbs.clearSearchEngines();

	return cost;
}

float cpp_oracle(bool* map, int* agents, int* sep, int num_of_agents, int num_of_locs, int num_of_rows, int num_of_cols) {
    vector<bool> vmap(map, map + (num_of_rows * num_of_cols));
    vector<int> vagents(agents, agents + num_of_locs);
    vector<int> vsep(sep, sep + num_of_agents);
    return cpp_oracle_(vmap, vagents, vsep, num_of_agents, num_of_rows, num_of_cols);
}
