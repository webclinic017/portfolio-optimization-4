
from collections import defaultdict



def fast_nondominated_sorting(individuals, k, first_front_only=False):
    """ Fast non-dominated.
    Sort the first *k* *individuals* into different nondomination levels
    Complexity O(MN^2) where M is the number of objectives and N the number of individuals.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param first_front_only: If :obj:`True` sort only the first front and exit.
    :returns: A list of Pareto fronts (lists), the first list includes nondominated individuals.
    """
    if k == 0:
        return []
    mapFitInd = defaultdict(list)
    for ind in individuals:
        mapFitInd[ind.fitness].append(ind)
    fits = list(mapFitInd.keys())
    currentFront = []
    nextFront = []
    dominatingFits = defaultdict(int)
    dominatedFits = defaultdict(list)
    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if fit_i.dominates(fit_j):
                dominatingFits[fit_j] += 1
                dominatedFits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominatingFits[fit_i] += 1
                dominatedFits[fit_j].append(fit_i)
        if dominatingFits[fit_i] == 0:
            currentFront.append(fit_i)
    fronts = [[]]
    for fit in currentFront:
        fronts[-1].extend(mapFitInd[fit])
    paretoSorted = len(fronts[-1])
    if not first_front_only:
        N = min(len(individuals), k)
        while paretoSorted < N:
            fronts.append([])
            for fit_p in currentFront:
                for fit_d in dominatedFits[fit_p]:
                    dominatingFits[fit_d] -= 1
                    if dominatingFits[fit_d] == 0:
                        nextFront.append(fit_d)
                        paretoSorted += len(mapFitInd[fit_d])
                        fronts[-1].extend(mapFitInd[fit_d])
            currentFront = nextFront
            nextFront = []
    return fronts