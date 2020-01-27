'''
Implementation of the Relative Convergence Speed (RCS) measure.

Proposed in: Almuth Meier, Oliver Kramer: "Prediction with Recurrent Neural 
Networks in Evolutionary Dynamic Optimization". In: K. Sim and P. Kaufmann: 
Applications of Evolutionary Computation (EvoApplications), 2018.

Created on Jan 24, 2020

@author: ameier
'''
import numpy as np


def rel_conv_speed(generations_of_chgperiods, global_opt_fit_per_chgperiod, best_found_fit_per_gen_and_alg,
                   only_for_preds, first_chgp_idx_with_pred_per_alg, with_abs):
    '''
    Measure of relative convergence speed of algorithms for one run of a 
    specific problem. Depends on the worst fitness value any algorithm achieved.
    Disadvantage: results not comparable to results in other papers if other 
    algorithms are employed.

    Proposed in: Almuth Meier and Oliver Kramer: "Prediction with Recurrent 
    Neural Networks in Evolutionary Dynamic Optimization", EvoApplications 2018.

    Function is called in statistical_tests.py

    @param generations_of_chgperiods:  dictionary containing for each change 
    period (the real ones, not only those the optimizer has detected) a list 
    with the generation numbers
    @param global_opt_fit_per_chgperiod: 1d numpy array: for each change period
    the global optimum fitness (for all changes stored in the dataset 
    file, not only for those used in the experiments, i.e. 
    len(global_opt_fit_per_chgperiod) may be larger than len(generations_of_chgperiods)
    @param best_found_fit_per_gen_and_alg: dictionary containing for each 
    algorithm a 1d numpy array that contains the best found fitness of each
    generation 
    @param only_for_preds: if True, RCS is computed based only on those change
    periods for that a prediction was made (might be different change periods
    for each algorithm)
    @param first_chgp_idx_with_pred_per_alg: dictionary containing for each 
    algorithm the index if the first change period for that a prediction was made
    @param with_abs: if True, RCS is computed with absolute differences as in
    the definition, otherwise with signed differences
    @return: dictionary containing one score for each algorithm:
        Score 0 -> best algorithm
        Score 1 -> worst algorithm
        None: if the respective run was not executed for the algorithm
    '''
    n_chgperiods = len(generations_of_chgperiods)

    # -------------------------------------------------------------------------
    # compute worst fitness per change achieved by any algorithm

    # 2d list: one row for each algorithm
    evals_per_alg_list = list(best_found_fit_per_gen_and_alg.values())
    # worst fitness per generation achieved by any algorithm
    worst_fit_evals = np.max(evals_per_alg_list, axis=0)
    # compute worst fitness per change
    worst_fit_per_chgperiod = {}
    for chgperiod_nr, gens in generations_of_chgperiods.items():
        worst_fit_per_chgperiod[chgperiod_nr] = np.max(worst_fit_evals[gens])
    # test whether worst fitness is larger than global best fitness (should be
    # the case in minimization problems)
    try:
        all_idcs = np.arange(n_chgperiods)
        bools = np.array(global_opt_fit_per_chgperiod)[all_idcs]
        assert np.all(list(worst_fit_per_chgperiod.values(
        )) >= bools), "global fitness worse than worst fitness"
    except Exception as e:
        print(e, flush=True)
        print("worst-fit-per-change-period: ")
        print(list(worst_fit_per_chgperiod.values()))
        print()
        print("global-opt-fit-per-change-period: ")
        print(global_opt_fit_per_chgperiod)
        raise  # throw the exception

    # -------------------------------------------------------------------------
    # compute convergence speed for each algorithm
    speed_per_alg = {}
    algs = list(best_found_fit_per_gen_and_alg.keys())
    for alg in algs:
        speed_per_alg[alg] = __convergence_speed__(generations_of_chgperiods,
                                                   global_opt_fit_per_chgperiod,
                                                   best_found_fit_per_gen_and_alg[alg],
                                                   worst_fit_per_chgperiod,
                                                   only_for_preds, first_chgp_idx_with_pred_per_alg[alg], with_abs)
    return speed_per_alg


def __convergence_speed__(generations_of_chgperiods,
                          global_opt_fit_per_chgperiod,
                          best_found_fit_per_gen,
                          worst_fit_per_chgperiod,
                          only_for_preds, first_chgp_idx_with_pred, with_abs):
    '''
    Internal method, called by rel_conv_speed().

    Computes convergence speed for one specific algorithm.

    Works only for minimization problems. (this implementation; but the measure 
    as formally defined should be able to handle maximization problems as well)
    Between 0 and 1. Best case: 0, worst case: 1.

    @param generations_of_chgperiods:  dictionary containing for each change 
    period (the real ones, not only those the EA has detected) a list with the 
    generation numbers
    @param global_opt_fit_per_chgperiod: 1d numpy array: for each change period
    the global optimum fitness (for all changes stored in the dataset 
    file, not only for those used in the experiments, i.e. 
    len(global_opt_fit_per_chgperiod) may be larger than len(generations_of_chgperiods))
    @param best_found_fit_per_gen: 1d numpy array containing for each generation 
    the best fitness value achieved by this algorithm.
    @param worst_fit_per_chgperiod: dictionary containing for each change period 
    the worst fitness value achieved by any algorithm.
    @param only_for_preds: if True, RCS is computed based only on those change
    periods for that a prediction was made (might be different change periods
    for each algorithm)
    @param first_chgp_idx_with_pred_per_alg: dictionary containing for each 
    algorithm the index if the first change period for that a prediction was made
    @param with_abs: if True, RCS is computed with absolute differences as in
    the definition, otherwise with signed differences
    @return: scalar: convergence speed for this algorithm
             None: if for the respective algorithm this run was not executed
    '''
    sum_norm_areas = 0
    n_summed_chgps = 0
    for chgperiod_nr, gens in generations_of_chgperiods.items():
        if only_for_preds and chgperiod_nr < first_chgp_idx_with_pred:
            continue
        n_summed_chgps += 1

        optimal_fit = global_opt_fit_per_chgperiod[chgperiod_nr]
        worst_fit = worst_fit_per_chgperiod[chgperiod_nr]
        if with_abs:
            # TODO actually, in the RCS definition the differences are reverse
            # due to abs() it should not make a difference
            best_worst_fit_diff = abs(optimal_fit - worst_fit)
        else:
            best_worst_fit_diff = (optimal_fit - worst_fit)
        # compute area for this change
        area_for_change = 0
        max_area_for_change = 0
        gen_in_chg = 0
        for gen in gens:
            found_fit = best_found_fit_per_gen[gen]
            if found_fit is None:
                return None
            assert optimal_fit <= found_fit, "opt-fit " + str(
                optimal_fit) + " fit " + str(found_fit)
            if with_abs:
                diff = abs(optimal_fit - found_fit)
            else:
                diff = (optimal_fit - found_fit)
            area_for_change += (gen_in_chg + 1) * diff  # +1, otherwise first 0
            max_area_for_change += (gen_in_chg + 1) * best_worst_fit_diff
            gen_in_chg += 1

        if max_area_for_change == 0:
            # means RCS=0 for this change period, since all algorithms always
            # had the global optimum fitness
            pass
        else:
            # normalize area so that it lies between 0 and 1
            norm_area_for_change = area_for_change / max_area_for_change
            sum_norm_areas += norm_area_for_change

    return sum_norm_areas / n_summed_chgps


def rcs_example():
    '''
    Example with two algorithms (a and b), and three change periods.
    A minimization problem is assumed.
    '''
    generations_of_chgperiods = {0: [0, 1, 2, 3],
                                 1: [4],
                                 2: [5, 6]}
    global_opt_fit_per_chgperiod = np.array([-12, 5, -4])
    best_found_fit_per_gen_and_alg = {
        'a': [10, 8, 7, 6, 11, 9, 4],
        'b': [15, 12, 12, 5, 10, 13, 7]}
    only_for_preds = False
    first_chgp_idx_with_pred_per_alg = {'a': 0, 'b': 0}
    with_abs = True

    speed_per_alg = rel_conv_speed(generations_of_chgperiods, global_opt_fit_per_chgperiod, best_found_fit_per_gen_and_alg,
                                   only_for_preds, first_chgp_idx_with_pred_per_alg, with_abs)
    print("a: ", speed_per_alg['a'])
    print("b: ", speed_per_alg['b'])


if __name__ == '__main__':
    rcs_example()
