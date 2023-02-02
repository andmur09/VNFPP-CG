import sys
import gurobipy as gp
from gurobipy import GRB
from math import sqrt, log
import numpy as np
from service_class import service
import graphviz as gvz

def masterProblem(topology, services, lp_relaxation=True):
    m = gp.Model(topology.name)
    nodes = topology.getLocationsByType("node")
    n_nodes = len(nodes)

    components = set()
    # Adds flow vector for each service where each element represents a path associated with that service
    # Also makes set of components for all service so that no duplicate components are considered
    for service in services:
        graph = service.graphs[topology.name]
        m.addMVar(shape=len(graph.getPaths()), vtype=GRB.CONTINUOUS, name= service.description + "_flows")
        for component in service.components:
            components.add(component)
    components = list(components)

    if lp_relaxation == False:
        # Adds binary variables vector where each element takes a value of 1 if the component is assigned to a particular node or 0 otherwise
        for component in components:
            m.addMVar(shape=n_nodes, vtype=GRB.BINARY, name=component.description + "_assignment")
        m.update()
    else:
        # For LP relaxation makes variables continuous
        for component in components:
            m.addMVar(shape=n_nodes, lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name=component.description + "_assignment")
        m.update()
    
    # # Adds constraint that the sum of flows for each service must be greater than the required throughput for the service
    for service in services:
        flows = [v for v in m.getVars() if service.description + "_flows" in v.varName]
        m.addConstr(gp.quicksum(flows) >= service.required_throughput, name="throughput_{}".format(service.description))

    # Adds a constraint that says that the sum of all flows through the edge must be less than the bandwidth:
    for i in range(len(topology.links)):
        path_vars = []
        coefficients = []
        for service in services:
            path_vars.append([v for v in m.getVars() if service.description + "_flows" in v.varName])
            graph = service.graphs[topology.name]
            coefficients.append([path.times_traversed[i][1] for path in graph.getPaths()])
        m.addConstr(gp.quicksum(coefficients[s][p]*path_vars[s][p] for s in range(len(services)) for p in range(len(path_vars[s]))) <= topology.links[i].bandwidth, name="bandwidth_{}".format(topology.links[i].description))
    
    # Adds constraint that forces flows to be equal to zero for any path not containing a node that a required component is assigned to
    for n in range(len(nodes)):
        for s in services:
            for c in s.components:
                #print([c.description for c in s.components])
                y = m.getVarByName(c.description + "_assignment"+"[{}]".format(n))
                x = [v for v in m.getVars() if s.description + "_flows" in v.varName]
                graph = s.graphs[topology.name]
                assignments = [p.component_assignment for p in graph.paths]
                #print(assignments)
                alpha = [assignments[g][c.description][nodes[n].description] for g in range(len(x))]
                m.addConstr(gp.quicksum(alpha[i]*x[i] for i in range(len(x))) <= s.required_throughput * y, name="assignmentflow_{}_{}_{}".format(s.description, c.description, nodes[n].description))

    # Adds a constraint that says that a component must be assigned to x different nodes where x is the replica count
    for component in components:
        assignment_vars = [v for v in m.getVars() if component.description + "_assignment" in v.varName]
        m.addConstr(gp.quicksum(assignment_vars) == component.replica_count, name = "replicas_{}".format(component.description))

    # Adds a constraint that says that the sum of component requirements running on a node must not exceed the capacity.
    for i in range(len(nodes)):
        assignment_variables = [m.getVarByName(component.description + "_assignment"+"[{}]".format(i)) for component in components]
        # For CPU
        requirements = [component.requirements["cpu"] for component in components]
        m.addConstr(gp.quicksum([assignment_variables[i]*requirements[i] for i in range(len(components))]) <= nodes[i].cpu, name="capacity_{}_{}".format("cpu", nodes[i].description))
        # For RAM
        requirements = [component.requirements["ram"] for component in components]
        m.addConstr(gp.quicksum([assignment_variables[i]*requirements[i] for i in range(len(components))]) <= nodes[i].ram, name="capacity_{}_{}".format("ram", nodes[i].description))
        # Adds a constraint that fixes all assignment variables to zero whenever a nodle is not active (used to simulate node failure)
        if nodes[i].active == False:
            for v in assignment_variables:
                m.addConstr(v == 0)

    #Sets objective to minimise node rental costs
    node_rental_costs = []
    node_assignments = []
    for i in range(len(nodes)):
        node_rental_costs.append(nodes[i].cost)
        node_assignments.append([m.getVarByName(component.description + "_assignment"+"[{}]".format(i)) for component in components])
    m.setObjective(gp.quicksum(node_rental_costs[i] * node_assignments[i][j] for i in range(len(nodes)) for j in range(len(components))), GRB.MINIMIZE)
    
    m.update()
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        m.write("{}.lp".format(m.ModelName))
        for service in services:
            print("\n Service {} required components {}".format(service.description, [c.description for c in service.components]))
        print('\n Objective: ', m.objVal)
        print('\n Vars:')
        for component in components:
            for n in range(n_nodes):
                variable = m.getVarByName(component.description + "_assignment[{}]".format(n))
                if variable.x != 0:
                    print("Description: {} assigned to {}".format(component.description, nodes[n].description), "Value: {}".format(variable.x))
        for service in services:
            for p in range(len(service.graphs[topology.name].paths)):
                variable = m.getVarByName(service.description + "_flows[{}]".format(p))
                if variable.x != 0:
                    print("Description: {} flow through path {}".format(service.description, p), "Value: {}".format(variable.x))
        print("\n")
    else:
        m.computeIIS()
        m.write("{}.ilp".format(m.ModelName))
        m.write("{}.lp".format(m.ModelName))
        m.write("{}.mps".format(m.ModelName))
        
    return m


def columnGeneration(topology, service):
    m = gp.Model(topology.name + "_" + service.description)
    graph = service.graphs[topology.name]

    # Adds variables representing whether an edge has been used in a path or not:
    links = m.addMVar(shape=len(graph.links), vtype=GRB.BINARY, name="links")
    weights = np.array([l.cost for l in graph.links])

    # Gets indexes of links corresponding to ones leaving the source
    start = graph.getStartNode()
    outgoing = graph.outgoingEdge(start)
    start_indexes = [i for i in range(len(graph.links)) if graph.links[i] in outgoing]
    # Adds constraint that exactly one link leaving the source must be active
    m.addConstr(gp.quicksum([links[i] for i in start_indexes]) == 1)

    # Gets indexes of links corresponding to ones entering the sink
    end = graph.getEndNode()
    incoming = graph.incomingEdge(end)
    end_indexes = [i for i in range(len(graph.links)) if graph.links[i] in incoming]
    # Adds constraint that exactly one link entering the sink must be active
    m.addConstr(gp.quicksum([links[i] for i in end_indexes]) == 1)

    # Adds constraint that the sum of the flow into and out of every other edge must be conserved
    source_and_sink = [graph.locations.index(i) for i in [start, end]]
    for i in range(len(graph.locations)):
        if i not in source_and_sink:
            incoming = graph.incomingEdge(graph.locations[i])
            incoming_i = [i for i in range(len(graph.links)) if graph.links[i] in incoming]
            outgoing = graph.outgoingEdge(graph.locations[i])
            outgoing_i = [i for i in range(len(graph.links)) if graph.links[i] in outgoing]
            m.addConstr(gp.quicksum([links[i] for i in incoming_i]) == gp.quicksum([links[o] for o in outgoing_i]))
    
    m.setObjective(weights @ links, GRB.MINIMIZE)
    m.update()
    m.optimize()

    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        vars = m.getVars()
        for i in range(len(vars)):
            if vars[i].x != 0:
                print("Description: {}".format(graph.links[i].description), "Value: {}".format(str(vars[i].x)))
        print("\n")
    else:
        m.computeIIS()
        m.write("{}.ilp".format(m.ModelName))
        m.write("{}.lp".format(m.ModelName))
        m.write("{}.mps".format(m.ModelName))

    # From solution gets set of links
    links_i = [i for i in range(len(graph.links)) if links[i].x == 1]
    used_links = [graph.links[i] for i in range(len(graph.links)) if i in links_i]

    # From links gets set of nodes and adds path
    used_nodes = set()
    for link in used_links:
        if link.source not in used_nodes:
            used_nodes.add(link.source)
        if link.sink not in used_nodes:
            used_nodes.add(link.sink)
    used_nodes = list(used_nodes)

    # Counts each time an edge in the original topology has been traversed by the path
    times_traversed = []
    for link1 in topology.links:
        z = 0
        for link2 in used_links:
            if link1.source.description == link2.source.description and link1.sink.description == link2.sink.description:
                z += 1   
            elif link1.sink.description == link2.source.description and link1.source.description == link2.sink.description:
                z += 1
        times_traversed.append((link1, z))
    
    # Makes binary vector representing assignment of component to node for the path:
    component_assignment = {}
    for component in service.components:
        component_assignment[str(component.description)] = {}
        for node in topology.getLocationsByType("node"):
            if node.description + "[" + component.description + "]" in [i.description for i in used_nodes]:
                component_assignment[str(component.description)][str(node.description)] = 1
            else:
                component_assignment[str(component.description)][str(node.description)] = 0
    path = service_path(topology.name + "_" + service.description, used_nodes, used_links, times_traversed, component_assignment)

    # Checks for duplicate paths and doesn't add if already added
    no_duplicates = True
    for other_path in graph.paths:        
        if path.check_if_same(other_path) == True:
            no_duplicates = False
            print("Duplicate exists with name {}: not adding path".format(other_path.name))
            break
    if no_duplicates == True:
        print("No duplicates: adding path")
        graph.addPath(path)
        path_added = True
    else:
        path_added = False
    return m, path_added

def optimiser(topology, services):
    # Kickstarts the algorithm by finding at least one feasible path for every service. This just uses a dummy node with infinite capacity connected to the gateway. The cost of this node is arbitrarily large. For the algorithm to select
    # this path we set the cost of the edges to 0. After this the cost is set high so that these edges are no longer used.
    print("\n######################### INITIALISING ##########################\n")
    for s in services:
        if topology.name not in s.graphs:
            s.addGraph(topology)
            graph = s.graphs[topology.name]
            for link in graph.links:
                if "Dummy" in link.description:
                    link.setLinkCost(0.0001)
            print("\n######################### SOLVING CG FOR {} PATH 0 ##########################\n".format(s.description))
            columnGeneration(topology, s)
            for link in graph.links:
                if "Dummy" in link.description:
                    link.setLinkCost(10000)
        else:
            graph = s.graphs[topology.name]
    
    k = 0
    terminate = False
    while terminate == False:
        # Solves master problem and sets lower bound
        print("\n######################### SOLVING MP ITERATION {} ##########################\n".format(k))
        m = masterProblem(topology, services)
        k += 1
        LB = m.objVal

        # Prints dual solution
        print("\nDual solution")
        for constraint in m.getConstrs():
            print(constraint.getAttr("ConstrName"))
            print(constraint.getAttr("Pi"))

        # Updates the edge weights in the service graph according to the current dual values from the LP relaxation of the restricted master problem
        objs = []
        for s in services:
            pi = m.getConstrByName("throughput_{}".format(s.description)).getAttr("Pi")
            graph = s.graphs[topology.name]
            for link in graph.links:
                #print("\n", link.description)
                if "Component" not in link.description:
                    if link.source.description != link.sink.description:
                        try:
                            ##print(m.getConstrByName("bandwidth_{}".format(link.description)))
                            link.setLinkCost(-m.getConstrByName("bandwidth_{}".format(link.description)).getAttr("Pi"))
                        except:
                            #print(m.getConstrByName("bandwidth_{}".format(link.getOpposingEdgeDescription())))
                            link.setLinkCost(-m.getConstrByName("bandwidth_{}".format(link.getOpposingEdgeDescription())).getAttr("Pi"))
                elif "Component" in link.description:
                    # Gets the description of the component assigned
                    component = [n.getComponentAssigned() for n in [link.source, link.sink] if n.getComponentAssigned() != None]
                    assert len(component) == 1
                    component = component[0]
                    # Gets the description of the node that the component is assigned on
                    node = [n.description for n in [link.source, link.sink] if "Component" not in n.description]
                    assert len(node) == 1
                    node = node[0]
                    # Gets the constraint using the service, component and node description
                    constraint = m.getConstrByName("assignmentflow_{}_{}_{}".format(s.description, component, node))
                    link.setLinkCost(-constraint.getAttr("Pi")/2) 
            print("\n######################### SOLVING CG FOR {} PATH {} ##########################\n".format(s.description, len(graph.paths)))
            print([l.cost for l in graph.links])
            m_cg, path_added = columnGeneration(topology, s)
            #if path_added == True:
            print(pi, m_cg.objVal)
            objs.append(-pi + m_cg.objVal)
        
        # Checks if none of the paths have improving columns - if not sets terminate to True and breaks out of loop
        terminate = True
        print("OBJS = ", objs)
        for obj in objs:
            if obj < 0:
                terminate = False
        print("Terminate = ", terminate)
    
    # Once column generaiton has converged it solves master problem with integrality enforced to obtain an upper bound
    print("\n######################### ALL COLUMNS ADDED SOLVING FINAL MP ##########################\n")
    m = masterProblem(topology, services, lp_relaxation=False)
    UB = m.objVal

    print("\nFinished optimisation.")
    print("\nObjective: ", m.objVal)
    print("Optimality gap: ", (UB-LB)/LB)
    return topology, services, m

    





    

    