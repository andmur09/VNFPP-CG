    # def find_cover_cut(self, function):
    #     """
    #     For a given LP solution, uses the requiredinstances constraint for each VNF to add cover cuts if found. This problem finds a cover, which must be satisfied by
    #     the integer solution and is not satisfied by the LP solution.
    #     """
    #     nodes = [n for n in self.network.locations if isinstance(n, Node) == True]
    #     self.nodes = nodes
    #     bound = sum([s.throughput for s in self.services if function.description in s.vnfs])
    #     m = gp.Model(function.description + "_cover", env=env)
    #     lp_vals = []
    #     coeffs = []
    #     for node in nodes:
    #         if isinstance(node, Dummy):
    #             # Gets the LP value associated with the variable.
    #             lp_vals.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment").x)
    #             # Adds the binary assignment variable.
    #             m.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name= node.description + "_" + function.description + "_assignment")
    #             # Gets the coefficients associated with the cover constraint.
    #             coeffs.append(bound)
    #         else:
    #             max_instances = min(8, node.cpu // function.cpu)
    #             for i in range(1, max_instances + 1):
    #                 # Gets the LP value associated with the variable.
    #                 lp_vals.append(self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment").x)
    #                 # Adds the binary assignment variable.
    #                 m.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name= node.description + "_" + function.description + "_" + str(i) + "_assignment")
    #                 # Gets the coefficients associated with the cover constraint.
    #                 coeffs.append(i * function.throughput)
    #     m.update()
    #     vars = m.getVars()
    #     # Adds the cover constraint. If all assignment variables taking a value of 1, were to be used in the MIP, this constraint would be violated. As such,
    #     # For this constraint to be satisfied at least one of the other variables (those with current value 0) must be active.
    #     m.addConstr(gp.quicksum(coeffs[i]*vars[i] for i in range(len(coeffs))) <= bound - 1, name = "cover")

    #     for node in nodes:
    #         vars_used = []
    #         for v in vars:
    #             node_name = v.varName.split("_")[0]
    #             if node_name == node.description:
    #                 vars_used.append(v)
    #         # Adds the constraint that says that only one assignment var for each function/node pair must be active.
    #         m.addConstr(gp.quicksum(vars_used[i] for i in range(len(vars_used))) <= 1, name = node.description + "_assignment")

    #     # This objective finds the value of all LP variables whose integer solution are zero. This should be less than 1 for the cover to be satisfied by the fractional solution.
    #     m.setObjective(gp.quicksum(lp_vals[i] * (1 - vars[i]) for i in range(len(vars))), GRB.MINIMIZE)
    #     # Updates and optimizes.
    #     m.update()
    #     m.optimize()
    #     m.write("{}.lp".format(m.getAttr("ModelName")))
    #     if m.status == GRB.OPTIMAL:
    #         logging.info(" Optimisation terminated successfully.") if self.verbose > 0 else None
    #         logging.info(' Ojective: {}'.format(m.objVal)) if self.verbose > 0 else None
    #         logging.info(' Vars:')
    #         vars = m.getVars()
    #         for i in range(len(vars)):
    #             if vars[i].x != 0:
    #                 logging.info(" Description: {}, Value: {}".format(vars[i].varName, vars[i].x)) if self.verbose > 0 else None
    #         return m
    #     else:
    #         logging.error(" Optimisation Failed - consult .ilp") if self.verbose > 0 else None
    #         m.computeIIS()
    #         m.write("{}.ilp".format(m.getAttr("ModelName")))
    #         raise ValueError("Optimisation failed")   

    # def find_cover_cut_node(self, node):
    #     """
    #     For a given LP solution, uses the requiredinstances constraint for each VNF to add cover cuts if found. This problem finds a cover, which must be satisfied by
    #     the integer solution and is not satisfied by the LP solution.
    #     """
        
    #     m = gp.Model(node.description + "_cover", env=env)
    #     lp_vals = []
    #     coeffs = []
    #     for function in self.vnfs:
    #         if function.throughput == None:
    #             # Gets the LP value associated with the variable.
    #             lp_vals.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment").x)
    #             # Adds the binary assignment variable.
    #             m.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name= node.description + "_" + function.description + "_assignment")
    #             # Gets the coefficients associated with the cover constraint.
    #             coeffs.append(function.cpu)
    #         else:
    #             max_instances = min(8, node.cpu // function.cpu)
    #             for i in range(1, max_instances + 1):
    #                 # Gets the LP value associated with the variable.
    #                 lp_vals.append(self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment").x)
    #                 # Adds the binary assignment variable.
    #                 m.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name= node.description + "_" + function.description + "_" + str(i) + "_assignment")
    #                 # Gets the coefficients associated with the cover constraint.
    #                 coeffs.append(i * function.cpu)
    #     m.update()
    #     vars = m.getVars()
    #     # Adds the cover constraint. If all assignment variables taking a value of 1, were to be used in the MIP, this constraint would be violated. As such,
    #     # For this constraint to be satisfied at least one of the other variables (those with current value 0) must be active.
    #     m.addConstr(gp.quicksum(coeffs[i]*vars[i] for i in range(len(coeffs))) >= node.cpu + 1, name = "cover")

    #     # This objective finds the value of all LP variables whose integer solution are zero. This should be less than 1 for the cover to be satisfied by the fractional solution.
    #     m.setObjective(gp.quicksum((1 - lp_vals[i]) * vars[i] for i in range(len(vars))), GRB.MINIMIZE)
    #     # Updates and optimizes.
    #     m.update()
    #     m.optimize()
    #     m.write("{}.lp".format(m.getAttr("ModelName")))
    #     if m.status == GRB.OPTIMAL:
    #         logging.info(" Optimisation terminated successfully.") if self.verbose > 0 else None
    #         logging.info(' Ojective: {}'.format(m.objVal)) if self.verbose > 0 else None
    #         logging.info(' Vars:')
    #         vars = m.getVars()
    #         for i in range(len(vars)):
    #             if vars[i].x != 0:
    #                 logging.info(" Description: {}, Value: {}".format(vars[i].varName, vars[i].x)) if self.verbose > 0 else None
    #         return m
    #     else:
    #         logging.error(" Optimisation Failed - consult .ilp") if self.verbose > 0 else None
    #         m.computeIIS()
    #         m.write("{}.ilp".format(m.getAttr("ModelName")))
    #         raise ValueError("Optimisation failed")   


            # # Solves the cover cut problem for each function and adds the constraints.
            # for function in self.vnfs:
            #     if function.throughput != None:
            #         logging.info(" SOLVING COVER CUT PROBLEM FOR {}\n".format(function.description)) if self.verbose > 0 else None
            #         cov = self.find_cover_cut(function)
            #         if cov.objVal < 1:
            #             logging.info(" New cut for VNF {} found so adding.\n".format(function.description)) if self.verbose > 0 else None
            #             function.cuts_generated += 1
            #             self.add_cut_from_model(function, cov)
            #         else:
            #             logging.info(" No cut found for VNF {}.\n".format(function.description)) if self.verbose > 0 else None
            
            # # Solves the cover cut problem for each node and adds the constraints.
            # for node in self.nodes:
            #     if not isinstance(node, Dummy):
            #         logging.info(" SOLVING COVER CUT PROBLEM FOR {}\n".format(node.description)) if self.verbose > 0 else None
            #         cov = self.find_cover_cut_node(node)
            #         if cov.objVal < 1:
            #             logging.info(" New cut for VNF {} found so adding.\n".format(node.description)) if self.verbose > 0 else None
            #             node.cuts_generated += 1
            #             self.add_cut_from_model(node, cov)
            #         else:
            #             logging.info(" No cut found for VNF {}.\n".format(function.description)) if self.verbose > 0 else None     
            # Updates and resolves RMP.



            
    # def solve_heuristic(self):
    #     """
    #     Heuristically generates a low cost configuration which can be used to generate promising initial paths.
    #     """
    #     # Initialises everything.
    #     assignments = {f.description: [] for f in self.vnfs}
    #     tp_count = {}
    #     curr_avail = {s.description: 0 for s in self.services}
    #     curr_flow = {s.description: 0 for s in self.services}
    #     cpu_left = {n.description: n.cpu for n in self.nodes}
    #     ram_left = {n.description: n.ram for n in self.nodes}
    #     bw_left = {e.get_description(): e.bandwidth for e in self.network.links}
    #     slas = {s.description: False for s in self.services}

    #     # Keep going until all nodes are used up or all sla's are satisfied.
    #     while min([cpu_left[n.description] for n in self.nodes]) >= min([f.cpu for f in self.vnfs]) and min([ram_left[n.description] for n in self.nodes]) >= min([f.ram for f in self.vnfs]) and any(item == False for item in [slas[s.description] for s in self.services]):
    #         for s in self.services:
    #             if not slas[s.description]:
    #                 # Finds the vnf's that we need to assign for another path for service s.
    #                 to_place = []
    #                 for f in s.get_vnfs(self.vnfs):
    #                     # If there are not current assignments
    #                     if f.description not in tp_count:
    #                         to_place.append(f)
    #                     # If there are no assignments with any available throughput left.
    #                     elif all(item == 0 for item in [tp_count[f.description][n] for n in tp_count[f.description]]):
    #                         to_place.append(f)

    #                 # Greedily places the vnf's
    #                 while to_place:
    #                     current = to_place.pop(0)
    #                     for node in self.nodes:
    #                         # If it's possible to place the VNF.
    #                         if current.cpu <= cpu_left[node.description] and current.ram <= ram_left[node.description] and node.description not in assignments[current.description]:
    #                             logging.info(" Adding {} to {}".format(current.description, node.description)) if self.verbose > 0 else None
    #                             assignments[current.description].append(node.description)
    #                             if current not in tp_count:
    #                                 tp_count[current.description] = {}
    #                             tp_count[current.description][node.description] = current.throughput
    #                             # Since we have a new assignment we should be able to route another path.
    #                             cpu_left[node.description] -= current.cpu
    #                             ram_left[node.description] -= current.ram
    #                             break

    #                 # Finds new path using assignments:
    #                 heuristic = {"assignments": {}, "edges": {}}
    #                 # Only considers the assignments whose throughput has not been exhausted.
    #                 for key in assignments:
    #                     for node in assignments[key]:
    #                         if key in tp_count and node in tp_count[key] and tp_count[key][node] > 0:
    #                             if key not in heuristic["assignments"].keys():
    #                                 heuristic["assignments"][key] = []
    #                             heuristic["assignments"][key].append(node)

    #                 heuristic["edges"] = copy.deepcopy(bw_left)
    #                 logging.info(tp_count) if self.verbose > 0 else None
    #                 logging.info(bw_left) if self.verbose > 0 else None
    #                 try:
    #                     cg = self.pricing_problem(s, initial = True, heuristic_solution = heuristic)
    #                     if cg.status == GRB.OPTIMAL:
    #                         # Adds the path if it has a feasible solution.
    #                         path = self.get_path_from_model(s, cg)
    #                         s.graph.add_path(path)                     
    #                         # Gets the max possible flow we can send down the path.
    #                         times_traversed, components_assigned = path.get_params()["times traversed"], path.get_params()["components assigned"]
    #                         components_assigned = {s.vnfs[i]: components_assigned[i] for i in range(len(s.vnfs))}
    #                         flow_poss = [bw_left[k]/times_traversed[k] for k in times_traversed] + [tp_count[k][components_assigned[k]] for k in components_assigned]
    #                         logging.info(" Flow possible {}".format(flow_poss)) if self.verbose > 0 else None
    #                         max_flow = min(flow_poss)
    #                         # If the flow required for the service is less than the max flow then we can send it all, else send the max.
    #                         if s.throughput - curr_flow[s.description] <= max_flow:
    #                             flow_to_send = s.throughput - curr_flow[s.description] 
    #                         else:
    #                             flow_to_send = max_flow
    #                         # Changes available flows.
    #                         curr_flow[s.description] += flow_to_send
    #                         for edge in times_traversed:
    #                             bw_left[edge] -= flow_to_send * times_traversed[edge]
    #                         for f in components_assigned:
    #                             tp_count[f][components_assigned[f]] -= flow_to_send
    #                         # Calculates the availability.
    #                         curr_avail[s.description] = 1
    #                         for i in range(len(s.vnfs)):
    #                             nodes_assigned = set()
    #                             for path in s.graph.paths:
    #                                 nodes_assigned.add(path.get_params()["components assigned"][i])
    #                             k = len(nodes_assigned)
    #                             curr_avail[s.description] *= (1 - (1 - self.node_availability * self.get_vnf_by_description(s.vnfs[i]).availability)**k)
    #                         logging.info(" Params used: {}".format(path.get_params())) if self.verbose > 0 else None
    #                         logging.info(" Path {}: ".format(path.description) + path.__str__() + " added.") if self.verbose > 0 else None
    #                         logging.info(" Flow {} out of {}".format(curr_flow[s.description], s.throughput)) if self.verbose > 0 else None
    #                         logging.info(" Availability {} out of {}\n".format(curr_avail[s.description], s.availability)) if self.verbose > 0 else None
    #                         # If the slas are satisfied, set to true
    #                         if curr_avail[s.description] >= s.availability and curr_flow[s.description] >= s.throughput:
    #                             slas[s.description] == True
    #                 except:
    #                     continue


    # def solve_heuristic2(self):
    #     """
    #     Heuristically generates a low cost configuration which can be used to generate promising initial paths.
    #     """
    #     # Makes list of replica count for required VNF's using availability
    #     n_replicas = {v.description: 1 for v in self.vnfs}
    #     function_availability = min([v.availability for v in self.vnfs])
    #     n_paths_needed = {s.description: 1 for s in self.services}

    #     # if the number of vnfs cannot handle the required throughput, then we increase the replica count so that it can.
    #     for v in self.vnfs:
    #         n_instances = ceil(sum([s.throughput for s in self.services if v in s.get_vnfs(self.vnfs)])/v.throughput)
    #         n_replicas[v.description] = max(n_instances, n_replicas[v.description])

    #     for s in self.services:
    #         if s.availability != None:
    #             k = len(s.vnfs)
    #             # Starting with one, check if availability can be satisfied using one path, else increase 1 until it can be satisfied
    #             i = 1
    #             while (1 - (1 - self.node_availability * function_availability)**i)**k < s.availability:
    #                 i += 1
    #             # Updates the number of paths needed for each service based on the availability.
    #             n_paths_needed[s.description] = max(n_paths_needed[s.description], i)
    #             for v in s.vnfs:
    #                 # If the required relicas for this service is greater than that previously calculated we update it.
    #                 n_replicas[v] = max(i, n_replicas[v])

    #     # Sorts the nodes according to cost
    #     nodes = [n for _, n in sorted(zip([n.cost for n in self.nodes], self.nodes), key=lambda pair: pair[0])]
    #     # Sorts vnfs according to cores:
    #     vnfs = [v for _, v in sorted(zip([v.cpu for v in self.vnfs], self.vnfs), key=lambda pair: pair[0], reverse=True)]
    #     # Start with the cheapest node
    #     i = 0
    #     assignments = {f.description: [] for f in vnfs}
    #     try:
    #         while any(x != 0 for x in n_replicas.values()) and i < len(nodes):
    #             current_node = nodes[i]
    #             cpu_remaining, ram_remaining = nodes[i].cpu, nodes[i].ram
    #             to_place = vnfs[:]
    #             while to_place:
    #                 v = to_place.pop(0)
    #                 # If it's possible to place it, and an instance still needs to be placed then place it.
    #                 if v.cpu <= cpu_remaining and v.ram <= ram_remaining and n_replicas[v.description] > 0:
    #                     assignments[v.description].append(current_node.description)
    #                     n_replicas[v.description] -= 1
    #                     cpu_remaining -= v.cpu
    #                     ram_remaining -= v.ram
    #             # Move onto next cheapest node.
    #             i += 1
    #         logging.info(" HEURISTIC FOUND THE FOLLOWING ASSIGNMENTS {}\n".format(assignments)) if self.verbose > 0 else None
    #         # Uses heuristic to add paths for each service.
    #         for service in self.services:
    #             logging.info(" USING HEURISTIC SOLUTION TO GENERATE COLUMNS FOR {}\n".format(service.description)) if self.verbose > 0 else None
    #             assignments_to_use = copy.deepcopy(assignments)
    #             for i in range(n_paths_needed[s.description]):
    #                 cg = self.pricing_problem(service, initial = True, heuristic_solution = assignments_to_use)
    #                 if cg.status == GRB.OPTIMAL:
    #                     # Adds the path if it has a feasible solution.
    #                     path = self.get_path_from_model(service, cg)
    #                     service.graph.add_path(path)
    #                     # Removes the nodes used in the solution from the assignments_to_use dictionary.
    #                     assignments_used = path.get_params()["components assigned"]
    #                     logging.info(" Params used: {}".format(path.get_params()))
    #                     for i in range(len(service.vnfs)):
    #                         # If the service uses the same VNF twice, then should only remove that VNF once.
    #                         if assignments_used[i] in assignments_to_use[service.vnfs[i]]:
    #                             assignments_to_use[service.vnfs[i]].remove(assignments_used[i])
    #                     logging.info(" Path {}: ".format(path.description) + path.__str__() + " added.\n") if self.verbose > 0 else None
                        
    #     except IndexError:
    #         logging.info(" Not possible to assign all services\n".format(assignments)) if self.verbose > 0 else None
